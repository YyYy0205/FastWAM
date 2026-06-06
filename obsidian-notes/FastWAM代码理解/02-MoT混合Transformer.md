---
tags:
  - FastWAM
  - MoT
  - 注意力机制
  - 代码理解
created: 2026-06-06
---

# MoT 混合 Transformer

**源文件**：`src/fastwam/models/wan22/mot.py`
**相关笔记**：[[00-索引]] | [[01-ActionDiT动作专家]] | [[03-训练Loss全流程]] | [[04-推理流程与KV缓存]]

---

## 1. 核心直觉

标准 Transformer 中，每个 token 只与**自己序列内**的 token 做注意力。

MoT 打破这个边界：**每一层**，把视频 token 和动作 token 合并成一个大序列，共同做一次注意力，然后再拆回各自序列继续处理。

```
标准做法：
  video tokens → [Video Attn] → video tokens
  action tokens → [Action Attn] → action tokens

MoT 做法（每层）：
  video tokens ──┐
                 ├─→ [concat] → [Mixed Attn] → [split]
  action tokens ─┘                               ├─→ video tokens
                                                 └─→ action tokens
```

---

## 2. 训练时 forward（第 447 行）

```python
for layer_idx in range(30):
    q_chunks, k_chunks, v_chunks = [], [], []

    # ── Step 1：每个专家独立计算本层的 Q/K/V ──
    for name in ["video", "action"]:
        expert = self.mixtures[name]
        block  = expert.blocks[layer_idx]
        x      = tokens_all[name]
        # video:  [B, S_v, 3072]
        # action: [B, T,   1024]

        # _build_expert_attention_io 内部：
        #   1. t_mod.chunk(6) → shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        #   2. attn_input = LayerNorm(x) * (1 + scale_msa) + shift_msa
        #   3. q = RoPE(norm_q(Linear_q(attn_input)))
        #   4. k = RoPE(norm_k(Linear_k(attn_input)))
        #   5. v = Linear_v(attn_input)
        q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, use_ckpt = \
            self._build_expert_attention_io(expert, block, x, freqs, t_mod)

        # video   q/k/v: [B, S_v, 24×128=3072]
        # action  q/k/v: [B, T,   24×128=3072]  ← hidden_dim 不同但 q/k/v 维度相同
        q_chunks.append(q)
        k_chunks.append(k)
        v_chunks.append(v)

    # ── Step 2：拼接 ──
    q_cat = torch.cat(q_chunks, dim=1)  # [B, S_v+T, 3072]
    k_cat = torch.cat(k_chunks, dim=1)  # [B, S_v+T, 3072]
    v_cat = torch.cat(v_chunks, dim=1)  # [B, S_v+T, 3072]

    # ── Step 3：一次混合注意力 ──
    mixed = self._mixed_attention(q_cat, k_cat, v_cat, attention_mask)
    # flash_attention 内部：
    #   [B, S, H*D] → reshape → [B, H, S, D]
    #   → scaled_dot_product_attention
    #   → [B, S, H*D]
    # mixed: [B, S_v+T, 3072]

    # ── Step 4：切回各自部分，各自过 FFN ──
    start = 0
    for name, seq_len in zip(["video", "action"], [S_v, T]):
        end = start + seq_len
        mixed_slice = mixed[:, start:end, :]

        # post_block 做：
        #   x = x + gate_msa ⊙ o_proj(mixed_slice)    ← self-attn 残差
        #   x = x + cross_attn(x, context)             ← 交叉注意力（文本条件）
        #   x = x + gate_mlp ⊙ FFN(LN(x)*scale+shift) ← FFN 残差
        tokens_all[name] = _apply_expert_post_block(
            block, residual_x, mixed_slice, gate_msa, shift_mlp, scale_mlp, gate_mlp, context
        )
        start = end
```

> [!question] 为什么两路 Q/K/V 能拼接？
> Video DiT `hidden_dim=3072`，Action DiT `hidden_dim=1024`，但注意力计算维度是：
> `num_heads(24) × attn_head_dim(128) = 3072`
> Q/K/V 来自线性投影，投影到的是**注意力头空间**，与 hidden_dim 无关，所以维度一致，可以拼接。

---

## 3. 注意力掩码（`fastwam.py` 第 386 行）

```python
def _build_mot_attention_mask(video_seq_len, action_seq_len, video_tokens_per_frame):
    total = video_seq_len + action_seq_len
    mask = torch.zeros((total, total), dtype=torch.bool)
    # False = 不可见（SDPA 中 False 位置被屏蔽）

    # video → video：因果掩码
    #   第一帧可看所有帧
    #   后续帧只能看自身之前的帧
    mask[:video_seq_len, :video_seq_len] = video_expert.build_video_to_video_mask(...)

    # action → action：全部可见
    mask[video_seq_len:, video_seq_len:] = True

    # action → 第一帧视频 token：可见
    first_frame_tokens = video_tokens_per_frame  # 例如 14×14 = 196
    mask[video_seq_len:, :first_frame_tokens] = True

    # video → action：保持 False（视频不看动作）
    return mask
```

### 掩码可视化

以 `S_v=392`（2帧×196），`T=16`，`first_frame=196` 为例：

```
          [frame0: 0..195] [frame1: 196..391] [action: 0..15]
frame0 →  [      ✓       ] [        ✓       ] [      ✗      ]
frame1 →  [      ✓       ] [        ✓       ] [      ✗      ]
action →  [      ✓       ] [        ✗       ] [      ✓      ]
```

> [!important] 关键设计：为什么动作只看第一帧？
> - 如果动作能看到所有未来视频帧，模型会"作弊"——直接从视频中读取动作，而不是真正学会预测。
> - 只允许看**第一帧**（初始观察），动作专家必须自己推断未来，迫使它内化世界模型知识。
> - 这也是推理时可以去掉视频生成的根本原因。

---

## 4. 梯度检查点（Gradient Checkpointing）

```python
# mot.py: _mixed_attention()
if self.mot_checkpoint_mixed_attn and self.training:
    return torch.utils.checkpoint.checkpoint(
        _forward, q_cat, k_cat, v_cat, use_reentrant=False
    )
```

混合注意力的序列长度是 `S_v + T`（可达 1000+），开启检查点以重计算代替存储中间激活，节省显存，代价是多一次前向计算。

| 配置 | `mot_checkpoint_mixed_attn` | 场景 |
|------|---------------------------|------|
| `configs/model/fastwam.yaml` | `true` | 大 batch 训练，省显存 |
| `configs/task/libero_uncond_2cam224_1e-4.yaml` | `false` | 评测任务，追求速度 |

---

## 5. 推理时的 KV 缓存（`prefill_video_cache`）

见 [[04-推理流程与KV缓存]] 详细讲解。核心思路：

```
prefill_video_cache():
  视频专家跑完 30 层，每层缓存 K/V
  → kv_cache: [30 × {"k": [B,Sv,3072], "v": [B,Sv,3072]}]

forward_action_with_video_cache():
  动作 Q 与 (cached video K + action K) 做注意力
  → 视频 K/V 不重新计算，节省 19×30 次 Transformer 前向
```

---

## 6. Tensor 形状速查

| 变量 | 形状 | 说明 |
|------|------|------|
| `q_cat` / `k_cat` / `v_cat` | `[B, S_v+T, 3072]` | 混合注意力输入 |
| `mixed` | `[B, S_v+T, 3072]` | 混合注意力输出 |
| `mixed_slice["video"]` | `[B, S_v, 3072]` | 视频部分切片 |
| `mixed_slice["action"]` | `[B, T, 3072]` | 动作部分切片 |
| `attention_mask` | `[S_v+T, S_v+T]` | bool，True=可见 |
| `video_kv_cache[l]["k"]` | `[B, S_v_first, 3072]` | 第 l 层视频 K 缓存 |

---

## 关联笔记

- [[01-ActionDiT动作专家]]：Action Expert 的结构，理解 Q/K/V 是怎么准备的
- [[03-训练Loss全流程]]：MoT forward 在训练中的调用位置
- [[04-推理流程与KV缓存]]：KV 缓存的完整使用流程
