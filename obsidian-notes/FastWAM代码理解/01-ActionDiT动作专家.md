---
tags:
  - FastWAM
  - ActionDiT
  - 扩散模型
  - 代码理解
created: 2026-06-06
---

# ActionDiT 动作专家

**源文件**：`src/fastwam/models/wan22/action_dit.py`
**相关笔记**：[[00-索引]] | [[02-MoT混合Transformer]] | [[03-训练Loss全流程]]

---

## 1. 在整体架构中的位置

```
FastWAM
├── VAE          → 视频编解码
├── T5 Encoder   → 文字编码
├── Video Expert → 视频生成（训练用）
├── Action Expert（ActionDiT）← 本文件
└── MoT          → 两个专家共享注意力
```

ActionDiT 的职责：对**带噪声的动作序列**做去噪，预测流匹配目标（flow target）。

---

## 2. 模块结构（`__init__`）

```python
self.action_encoder  = nn.Linear(action_dim, hidden_dim=1024)
# 把原始动作维度（如 14）映射到 hidden_dim

self.text_embedding  = nn.Sequential(
    nn.Linear(4096, 1024), nn.GELU(), nn.Linear(1024, 1024)
)
# T5 context（4096维）→ hidden_dim，作为交叉注意力的 context

self.time_embedding  = nn.Sequential(
    nn.Linear(freq_dim=256, 1024), nn.SiLU(), nn.Linear(1024, 1024)
)
# 正弦位置编码(t) → 时间嵌入

self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(1024, 1024 * 6))
# 时间嵌入 → 6 个调制参数（shift/scale/gate × 2：self-attn 和 FFN 各一组）

self.blocks = nn.ModuleList([DiTBlock(...) for _ in range(30)])
# 30 层 DiTBlock，与 Video DiT 结构完全一致

self.head = ActionHead(hidden_dim=1024, out_dim=action_dim)
# 输出头：LayerNorm（带调制）→ Linear 预测去噪目标
```

> [!info] 关键约束：为什么头数必须和 Video DiT 一致？
> Video DiT `hidden_dim=3072`，Action DiT `hidden_dim=1024`，但两者的注意力计算维度是：
> `num_heads(24) × attn_head_dim(128) = 3072`
> Q/K/V 是注意力头维度，不是 hidden_dim，**MoT 拼接的是 Q/K/V，所以维度必须相同**。

---

## 3. pre_dit 详解（第 226 行）

`pre_dit` 和 `post_dit` 的拆分是为了让 MoT 能插入中间做混合注意力。

```python
def pre_dit(self, action_tokens, timestep, context, context_mask):
    # action_tokens: [B, T, action_dim]  e.g. [2, 16, 14]
    # timestep:      [B]                 e.g. [2]

    # Step 1: 正弦编码时间步 → 时间调制张量
    t = self.time_embedding(sinusoidal_embedding_1d(freq_dim=256, timestep))
    # t: [B, 1024]

    t_mod = self.time_projection(t).unflatten(1, (6, 1024))
    # t_mod: [B, 6, 1024]
    # 这 6 个向量分别是：
    #   [0] shift_msa   [1] scale_msa   [2] gate_msa
    #   [3] shift_mlp   [4] scale_mlp   [5] gate_mlp
    # 每一层 DiTBlock 都用它们对 LayerNorm 输出做仿射变换

    # Step 2: 动作 token 线性映射
    tokens = self.action_encoder(action_tokens)
    # [B, T, action_dim] → [B, T, 1024]

    # Step 3: 文本 context 映射（T5 4096 → hidden 1024）
    context_emb = self.text_embedding(context)
    # [B, L, 4096] → [B, L, 1024]

    # Step 4: 交叉注意力 mask 扩展
    context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
    # [B, L] → [B, T, L]  每个动作 token 都能看全部文本 token

    # Step 5: RoPE 频率（1D，按动作序列位置）
    freqs = self.freqs[:seq_len]
    # [T, 1, rope_dim]

    return {
        "tokens": tokens,           # [B, T, 1024]  ← MoT 的输入
        "freqs":  freqs,            # [T, 1, rope_dim]
        "t_mod":  t_mod,            # [B, 6, 1024]
        "context": context_emb,     # [B, L, 1024]
        "context_mask": context_attn_mask,  # [B, T, L]
        "meta": {"batch_size": B, "seq_len": T},
    }
```

---

## 4. post_dit 详解（第 301 行）

```python
def post_dit(self, tokens, pre_state):
    return self.head(tokens)
    # tokens: [B, T, 1024]  ← MoT 30 层后的输出
    # ActionHead 内部：
    #   shift, scale = (modulation + t).chunk(2)
    #   out = Linear(LayerNorm(tokens) * (1 + scale) + shift)
    # 输出: [B, T, action_dim]  ← 预测的去噪方向
```

> [!tip] pre_dit / post_dit 分割的意义
> - `pre_dit`：准备 embedding、时间调制、RoPE 频率
> - 中间 30 层：交给 MoT 统一调度（混合注意力）
> - `post_dit`：只做最后的线性投影
>
> 这样 MoT 可以"劫持"中间的注意力计算，把两个专家的 token 合并处理。

---

## 5. 初始化策略

ActionDiT 的主干权重从 Video DiT 线性插值得到，而非随机初始化。

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

> [!note] 为什么不用随机初始化？
> 从 Video DiT 插值初始化后，ActionDiT 已经具备对时间步、文本条件的响应能力，
> 类似于知识蒸馏的起点，比随机初始化收敛更快、效果更好。

---

## 6. Tensor 形状速查

| 变量 | 形状 | 说明 |
|------|------|------|
| `action_tokens`（输入） | `[B, T, action_dim]` | 原始/带噪声动作序列 |
| `tokens`（pre_dit 输出） | `[B, T, 1024]` | 映射到 hidden_dim |
| `t_mod` | `[B, 6, 1024]` | 时间调制，6 个参数 |
| `context_emb` | `[B, L, 1024]` | 文本 embedding |
| `tokens`（MoT 输出） | `[B, T, 1024]` | 经过混合注意力后 |
| `pred_action`（post_dit 输出） | `[B, T, action_dim]` | 预测去噪方向 |

---

## 关联笔记

- [[02-MoT混合Transformer]]：ActionDiT 的 30 层如何与 Video DiT 共享注意力
- [[03-训练Loss全流程]]：`pre_dit` 和 `post_dit` 在训练中的调用位置
- [[04-推理流程与KV缓存]]：推理时 ActionDiT 独立运行的完整流程
