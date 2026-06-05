# FastWAM 学习文档

## 目录

- [核心思想](#核心思想)
- [整体架构](#整体架构)
- [逐模块详解](#逐模块详解)
  - [1. VAE 视频编解码器](#1-vae-视频编解码器)
  - [2. T5 文本编码器](#2-t5-文本编码器)
  - [3. ActionDiT 动作专家](#3-actiondit-动作专家)
  - [4. Video DiT 视频专家](#4-video-dit-视频专家)
  - [5. MoT 混合 Transformer](#5-mot-混合-transformer)
  - [6. 流匹配调度器](#6-流匹配调度器)
- [训练流程详解](#训练流程详解)
- [推理流程详解](#推理流程详解)
- [数据流详解](#数据流详解)
- [关键设计决策](#关键设计决策)
- [推荐学习路径](#推荐学习路径)

---

## 核心思想

### 背景问题

传统的 World Action Model（WAM）思路：

```
观察帧(t=0) → [生成未来视频帧] → 根据视频预测动作
```

这种做法在**测试时**需要先完整生成未来视频（几十步扩散推理），再根据视频预测动作，推理速度慢。

### FastWAM 的解答

**训练时**：视频专家和动作专家通过 MoT 联合训练，动作专家通过混合注意力"偷看"视频专家的表示，学会了利用世界模型知识预测动作。

**推理时**：只运行动作专家，视频专家完全跳过。动作专家已在训练中内化了足够的视觉世界知识，不再需要显式生成视频。

```
训练：观察帧 → [Video Expert + Action Expert（MoT 联合）] → 视频预测 + 动作预测
推理：观察帧 → [Action Expert only] → 动作预测（无需视频生成）
```

**核心结论**：通过 MoT 联合训练，动作专家可以从视频专家的表示中蒸馏世界知识，测试时无需"想象未来"。

---

## 整体架构

```
输入
  ├── 视频帧序列（包括第一帧条件帧）
  ├── 文字指令（如 "pick up the mug"）
  └── 本体感知状态（可选，proprio）

FastWAM
  ├── VAE Encoder          → 视频帧压缩为 latent（空间 ÷16，时间 ÷4）
  ├── T5 Encoder           → 文字指令 → token 序列（dim=4096）
  │
  ├── Video Expert (WanVideoDiT)
  │     ├── Patch Embed    → latent patch 化为 token
  │     ├── 30× DiTBlock   → 时空注意力 + 交叉注意力（条件于文本）
  │     └── 输出头          → 预测去噪 latent
  │
  ├── Action Expert (ActionDiT)
  │     ├── Action Encoder → 动作序列线性映射到 hidden_dim
  │     ├── 30× DiTBlock   → （与 Video Expert 共享注意力）
  │     └── 输出头 (ActionHead) → 预测去噪动作
  │
  └── MoT                  → 每层拼接两路 Q/K/V，执行一次混合注意力

输出
  ├── 训练时：video_loss + action_loss
  └── 推理时：去噪后的动作序列 [B, T, action_dim]
```

---

## 逐模块详解

### 1. VAE 视频编解码器

**文件**：`src/fastwam/models/wan22/wan_video_vae.py`

**作用**：将高分辨率视频帧压缩为低维 latent，再从 latent 重建视频。

**压缩倍率**：
- 空间：÷16（224px → 14 个 patch）
- 时间：÷4（4 帧压缩为 1 个 latent 步，第 1 帧单独保留）
- 通道：原始 3 通道 → latent 16 通道

**在训练中的作用**：
```python
# fastwam.py: _encode_video_latents()
z = vae.encode(video_tensor)   # [B, 3, T, H, W] → [B, 16, T', H', W']
# VAE 编码后，latent 作为扩散训练的目标
```

**注意**：VAE 权重在训练中**冻结**，不参与梯度更新。

---

### 2. T5 文本编码器

**文件**：`src/fastwam/models/wan22/wan_video_text_encoder.py`

**作用**：将自然语言指令编码为 token 序列，作为视频和动作专家的交叉注意力条件。

**规格**：
- 模型：`Wan-AI/Wan2.1-T2V-1.3B` 的 T5 模块
- 最大序列长度：128 tokens（训练时）
- 输出维度：4096（`text_dim`）

**优化技巧**：训练前用 `precompute_text_embeds.py` 把所有 embedding 缓存到磁盘，训练时直接读取，避免每次前向传播调用 T5。

```python
# fastwam.py: encode_prompt()
ids, mask = tokenizer(prompt)
prompt_emb = text_encoder(ids, mask)  # [B, L, 4096]
```

---

### 3. ActionDiT 动作专家

**文件**：`src/fastwam/models/wan22/action_dit.py`

**作用**：对带噪声的动作序列进行去噪，预测动作流场（flow matching target）。

**架构细节**：

```
输入：noisy_action [B, T, action_dim]
  ↓
action_encoder: Linear(action_dim → hidden_dim=1024)
  ↓
time_embedding: 正弦编码(t) → MLP → [B, hidden_dim]
time_projection: SiLU → Linear → [B, hidden_dim * 6]  (生成 6 个调制参数)
text_embedding: MLP(text_dim=4096 → hidden_dim=1024)
  ↓
30 × DiTBlock（与 Video Expert 结构相同，但 hidden_dim 更小）
  ↓ （在 MoT 中与 video tokens 混合注意力）
ActionHead: LayerNorm(调制) → Linear(hidden_dim → action_dim)
  ↓
输出：predicted_noise [B, T, action_dim]
```

**与 Video DiT 的关键约束**：

```python
# fastwam.py: from_wan22_pretrained()
# 两个专家必须满足：
assert action_expert.num_heads == video_expert.num_heads        # 24
assert action_expert.attn_head_dim == video_expert.attn_head_dim  # 128
assert len(action_expert.blocks) == len(video_expert.blocks)    # 30
```

注意力头数和每头维度必须相同，因为 MoT 将两路 Q/K/V 拼接后一起做注意力。

**初始化策略**：ActionDiT 的主干权重从 Video DiT 的对应层线性插值而来（`preprocess_action_dit_backbone.py`），而非随机初始化，提供更好的起点。

---

### 4. Video DiT 视频专家

**文件**：`src/fastwam/models/wan22/wan_video_dit.py`

**作用**：对带噪声的视频 latent 进行去噪，同时通过 MoT 为动作专家提供世界知识。

**架构细节**：

```
输入：noisy_latent [B, 16, T', H', W']
  ↓
3D Patch Embed：patch_size=[1,2,2] → [B, S_video, hidden_dim=3072]
  ↓
RoPE 频率编码（时空位置编码）
  ↓
30 × DiTBlock：
  ├── Pre-Norm（AdaLayerNorm，由 timestep 调制 shift/scale/gate）
  ├── Self-Attention（在 MoT 中与 action tokens 混合）
  ├── Cross-Attention（条件于 T5 文本 embedding）
  └── FFN（门控激活）
  ↓
输出头：Linear(hidden_dim=3072 → out_dim=48)
  ↓
输出：pred_video [B, 48, T', H', W'] → 重整回 [B, 16, T', H', W']
```

**注意力掩码模式**（`video_attention_mask_mode: "first_frame_causal"`）：
- 第一帧可以看所有帧
- 后续帧只能看到自身之前的帧（因果掩码）

---

### 5. MoT 混合 Transformer

**文件**：`src/fastwam/models/wan22/mot.py`

这是 FastWAM 最核心的创新模块。

**直觉理解**：

标准 Transformer 中，每个 token 只与自己序列内的 token 做注意力。MoT 打破了这个边界——在每一层，视频 token 和动作 token 合并成一个大序列，共同做一次注意力，然后再拆回各自的序列继续处理。

**逐层前向传播**（`mot.py: forward()`）：

```
第 l 层：

1. 视频专家：video_tokens → Q_v, K_v, V_v（含 RoPE 位置编码）
2. 动作专家：action_tokens → Q_a, K_a, V_a

3. 拼接：
   Q_cat = [Q_v | Q_a]   shape: [B, S_v + S_a, H * D_head]
   K_cat = [K_v | K_a]
   V_cat = [V_v | V_a]

4. 混合 Flash Attention（带注意力掩码）：
   attn_out = flash_attention(Q_cat, K_cat, V_cat, mask=attention_mask)

5. 切分回各自序列：
   video_attn_out = attn_out[:, :S_v]
   action_attn_out = attn_out[:, S_v:]

6. 各自过 FFN（残差连接 + 交叉注意力）：
   video_tokens = video_expert.post_block(video_attn_out, ...)
   action_tokens = action_expert.post_block(action_attn_out, ...)
```

**注意力掩码设计**（`fastwam.py: _build_mot_attention_mask()`）：

```
           video_tokens        action_tokens
           [S_v tokens]        [S_a tokens]

video  →   [  √ 全看  ]        [  ✗ 不看  ]
action →   [√ 仅第1帧]         [  √ 全看  ]
```

关键约束：
- 视频 token 不看动作 token（防止动作信息污染视频生成）
- 动作 token 只看**第一帧**的视频 token（获取初始状态，不依赖未来帧）
- 动作 token 之间全部可见

**推理时的 KV 缓存加速**（`mot.py: prefill_video_cache()`）：

推理时，视频分支（仅包含第一帧）只需运行一次，缓存每层的 K/V：
```python
video_kv_cache = mot.prefill_video_cache(first_frame_tokens, ...)
# 之后每次扩散去噪步只运行动作分支
for t in timesteps:
    pred_action = mot.forward_action_with_video_cache(
        action_tokens, video_kv_cache=video_kv_cache, ...
    )
```

---

### 6. 流匹配调度器

**文件**：`src/fastwam/models/wan22/schedulers/scheduler_continuous.py`

**作用**：FastWAM 使用流匹配（Flow Matching）而非标准 DDPM 做扩散，视频和动作各有独立的调度器。

**核心概念**：

```
训练目标（flow target）：u_t = x_1 - x_0
  其中 x_1 = 原始数据（干净视频/动作）
      x_0 = 标准正态噪声
      
加噪过程：x_t = (1-t) * x_0 + t * x_1   （线性插值）

模型学习：在时间步 t，给定 x_t，预测 x_1 - x_0
```

**shift 参数**的作用：通过 `shift=5.0` 将采样时间步分布向高噪声端偏移，让模型更多地学习粗粒度结构。

**训练时独立采样时间步**：视频和动作的 `timestep` 独立采样，不要求相同，增加训练数据多样性。

---

## 训练流程详解

**入口**：`src/fastwam/models/wan22/fastwam.py: training_loss()`

```
Step 1: 数据准备
  sample = {video_frames, action, language_instruction, proprio, ...}

Step 2: 编码
  latents = vae.encode(video_frames)    # [B, 16, T', H', W']
  context = text_encoder(instruction)   # [B, L, 4096]（或从缓存读取）

Step 3: 分别加噪（独立 timestep）
  noise_video ~ N(0,1)
  t_video ~ Uniform(0,1)  # 经 shift 变换
  latents_noisy = flow_interpolate(latents, noise_video, t_video)

  noise_action ~ N(0,1)
  t_action ~ Uniform(0,1)
  action_noisy = flow_interpolate(action, noise_action, t_action)

Step 4: 预处理（pre_dit）
  video_pre = video_expert.pre_dit(latents_noisy, t_video, context)
    → tokens: [B, S_v, 3072]
    → freqs（RoPE）, t_mod（时间调制）, context（交叉注意力用）

  action_pre = action_expert.pre_dit(action_noisy, t_action, context)
    → tokens: [B, T, 1024]
    → freqs, t_mod, context

Step 5: MoT 混合注意力（30层）
  attention_mask = build_mot_mask(S_v, T, tokens_per_frame)
  tokens_out = mot.forward(
      video_tokens, action_tokens,
      freqs_all, context_all, t_mod_all,
      attention_mask
  )

Step 6: 后处理（post_dit）& 损失计算
  pred_video = video_expert.post_dit(tokens_out["video"])
  pred_action = action_expert.post_dit(tokens_out["action"])

  target_video = noise_video - latents  # flow target
  target_action = noise_action - action

  loss_video = MSE(pred_video, target_video) * weight(t_video)
  loss_action = MSE(pred_action, target_action) * weight(t_action)
  loss = λ_video * loss_video + λ_action * loss_action

Step 7: 反向传播
  只有 MoT（含两个专家的 DiT blocks）和 proprio_encoder 更新梯度
  VAE 和 T5 完全冻结
```

---

## 推理流程详解

**入口**：`src/fastwam/models/wan22/fastwam.py: infer_action()`

```
Step 1: 编码第一帧
  first_frame_latents = vae.encode(input_image)   # [B, 16, 1, H', W']
  context = text_encoder(prompt)                  # 或从缓存

Step 2: 初始化动作噪声
  action_noisy = torch.randn([B, action_horizon, action_dim])

Step 3: 视频 KV 缓存预填充（只做一次）
  video_kv_cache = mot.prefill_video_cache(
      video_expert.pre_dit(first_frame_latents, t=0)
  )
  # 缓存 30 层的 K/V，每层 shape: [B, S_v_first_frame, H*D_head]

Step 4: 扩散去噪循环（默认 20 步）
  for t in scheduler.timesteps:   # t: 1.0 → 0.0
      pred_action_noise = mot.forward_action_with_video_cache(
          action_expert.pre_dit(action_noisy, t),
          video_kv_cache=video_kv_cache,
      )
      action_noisy = scheduler.step(pred_action_noise, t, action_noisy)

Step 5: 反归一化
  action = dataset_stats.denormalize(action_noisy)
  # 输出: [B, action_horizon, action_dim]
```

**与联合推理（infer_joint）的区别**：`infer_action` 只运行动作分支；`infer_joint` 同时生成视频和动作，用于可视化或调试，但推理速度慢很多。

---

## 数据流详解

**文件**：`src/fastwam/datasets/processors/fastwam_processor.py`

```
原始数据（LeRobot 格式）
  ├── 摄像头图像：{cam_0, cam_1, ...} × T 帧
  ├── 末端执行器位姿（action）：[T, 7]（3 位移 + 3 旋转 + 1 抓手）
  └── 本体感知（proprio）：关节角等

         ↓ FaswWAMProcessor

处理后输入
  ├── video: [B, 3, T, H, W]  → 送 VAE 编码
  │     （多摄像头图像拼接为一个时间序列）
  ├── action: [B, T_act, action_dim]  → 归一化到 [-1, 1]
  │     （相对动作表示或绝对动作，取决于配置）
  ├── action_is_pad: [B, T_act]  → 标记哪些时间步是 padding
  ├── image_is_pad: [B, T]       → 标记哪些帧是 padding
  ├── context / context_mask     → T5 embedding（预计算或在线计算）
  └── proprio（可选）            → 归一化后的关节状态
```

**归一化统计**：首次训练时自动计算 `dataset_stats.json`，记录动作和状态的均值/标准差，推理时用于反归一化输出动作。

---

## 关键设计决策

### 1. 为什么 ActionDiT 的 hidden_dim 比 Video DiT 小？

Video DiT：`hidden_dim=3072`，Action DiT：`hidden_dim=1024`。

动作序列比视频 latent 简单得多（维度低，语义清晰），不需要那么大的容量。同时保持注意力头数（24）和每头维度（128）相同，确保 MoT 混合注意力的维度兼容。

### 2. 为什么动作 token 只能看第一帧视频 token？

这是关键的**因果约束**：
- 如果动作 token 能看到所有未来视频帧，模型会"作弊"——直接从视频帧中读取动作，而不是真正学会预测动作。
- 只允许看第一帧（初始观察），动作专家必须自己推断未来，迫使它内化世界模型知识。

### 3. 为什么视频和动作使用独立的 timestep？

联合采样（同一 timestep）会造成训练不平衡：两个任务难度相同时才梯度对齐。独立采样让两个专家各自处于不同的去噪难度，增加训练多样性，避免一个任务主导梯度。

### 4. ActionDiT 权重初始化为什么不用随机？

从 Video DiT 插值初始化，ActionDiT 的主干已经具备对时间步、文本条件的响应能力，比随机初始化收敛更快，效果更好（类似知识蒸馏的起点）。

### 5. 梯度检查点（gradient checkpointing）的使用

```python
mot_checkpoint_mixed_attn: true  # 训练时默认开启
```

MoT 的混合注意力计算量大，开启梯度检查点以重计算代替存储中间激活，节省 GPU 显存。评测任务配置（如 `libero_uncond_2cam224_1e-4.yaml`）关闭以提升速度。

---

## 推荐学习路径

建议按以下顺序阅读源码：

```
第 1 步：理解数据结构
  configs/data/libero_2cam.yaml          → 了解输入数据规格
  configs/model/fastwam.yaml             → 了解模型超参数全貌

第 2 步：理解动作专家（最简单的子模块）
  src/fastwam/models/wan22/action_dit.py → ActionDiT 完整实现
  重点：pre_dit()、ActionHead、DiTBlock

第 3 步：理解视频 DiT 的基础组件
  src/fastwam/models/wan22/wan_video_dit.py  → DiTBlock、RoPE、flash_attention
  重点：DiTBlock 的 AdaLayerNorm 调制机制

第 4 步：理解 MoT（核心创新）
  src/fastwam/models/wan22/mot.py
  重点：forward()、_mixed_attention()、prefill_video_cache()

第 5 步：理解完整模型组装
  src/fastwam/models/wan22/fastwam.py
  重点：
    - from_wan22_pretrained()：模型构建流程
    - _build_mot_attention_mask()：注意力掩码设计
    - training_loss()：完整训练步骤
    - infer_action()：推理流程

第 6 步：理解训练循环
  src/fastwam/trainer.py
  重点：train_step()、checkpointing 逻辑、DeepSpeed 集成

第 7 步：理解数据处理
  src/fastwam/datasets/processors/fastwam_processor.py
  src/fastwam/datasets/robot_video_dataset.py
```

### 阅读时的辅助问题

- MoT 中，action token 是如何"看到"video token 信息的？哪行代码实现了这个？
- `_build_mot_attention_mask` 返回的 mask 形状是什么？`True` 代表可见还是不可见？
- 推理时 `prefill_video_cache` 缓存了什么？为什么只需调用一次？
- `pre_dit` 和 `post_dit` 的分割点在哪里？为什么要这样分割？
- `training_loss` 中视频和动作的 `timestep` 是否相同？
