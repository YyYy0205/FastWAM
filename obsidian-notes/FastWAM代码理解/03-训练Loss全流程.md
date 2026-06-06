---
tags:
  - FastWAM
  - 训练
  - 流匹配
  - Loss
  - 代码理解
created: 2026-06-06
---

# 训练 Loss 全流程

**源文件**：`src/fastwam/models/wan22/fastwam.py`，`training_loss()` 第 448 行
**相关笔记**：[[00-索引]] | [[01-ActionDiT动作专家]] | [[02-MoT混合Transformer]]

---

## 1. 训练数据假设

以 LIBERO 任务为例：
- Batch size：2
- 分辨率：224 × 224
- 视频帧数：16
- 动作步长（action_horizon）：16
- 摄像头数：2（拼成时间序列处理）

---

## 2. 完整流程（带 Tensor 形状）

### 阶段 1：编码输入

```python
inputs = self.build_inputs(sample)

# VAE 编码视频
input_latents = inputs["input_latents"]
# [2, 3, 16, 224, 224] → [2, 16, 5, 14, 14]
#   时间：(16-1)//4 + 1 = 5 个 latent 步
#   空间：224 // 16 = 14
#   通道：16（VAE latent 通道数）

first_frame_latents = inputs["first_frame_latents"]
# [2, 16, 1, 14, 14]  ← 第一帧单独保存，训练时作为条件帧，不加噪

action       = inputs["action"]           # [2, 16, action_dim]
context      = inputs["context"]          # [2, L, 4096]  T5 文本 embedding
context_mask = inputs["context_mask"]     # [2, L]
```

> [!note] VAE 权重冻结
> VAE 编码只做推理，不参与梯度计算。训练时只有 MoT（含两个专家 DiT blocks）和 `proprio_encoder`（可选）更新梯度。

---

### 阶段 2：视频加噪（流匹配）

```python
noise_video    = torch.randn_like(input_latents)    # [2, 16, 5, 14, 14]
timestep_video = train_video_scheduler.sample_training_t(batch_size=2)
# timestep_video: [2]，每个样本独立采样，范围 [0,1]，经 shift=5.0 偏移

# 流匹配加噪：x_t = (1-t) * noise + t * data
latents_noisy = scheduler.add_noise(input_latents, noise_video, timestep_video)
# [2, 16, 5, 14, 14]

# 流匹配目标：u_t = data - noise（预测从噪声指向数据的方向）
target_video = scheduler.training_target(input_latents, noise_video, timestep_video)
# [2, 16, 5, 14, 14]

# 第一帧替换为干净帧（条件帧不加噪）
latents_noisy[:, :, 0:1] = first_frame_latents   # [2, 16, 1, 14, 14]
```

> [!info] 流匹配 vs DDPM
> | 对比项 | 流匹配（FastWAM） | DDPM |
> |--------|-----------------|------|
> | 加噪公式 | `x_t = (1-t)*noise + t*data` | `x_t = √ᾱ·data + √(1-ᾱ)·noise` |
> | 预测目标 | `data - noise`（流场方向） | `noise`（预测噪声） |
> | 轨迹 | 直线 | 弯曲 |
> | 推理步数 | 通常更少 | 通常更多 |

---

### 阶段 3：动作加噪（独立 timestep）

```python
noise_action    = torch.randn_like(action)          # [2, 16, action_dim]
timestep_action = train_action_scheduler.sample_training_t(batch_size=2)
# 独立采样！与 timestep_video 不同

noisy_action  = scheduler.add_noise(action, noise_action, timestep_action)
target_action = scheduler.training_target(action, noise_action, timestep_action)
```

> [!important] 为什么要独立采样时间步？
> - 联合采样（同一 t）会造成训练不平衡：两个任务在同一去噪难度下梯度对齐，但难度未必一致。
> - 独立采样让两个专家各自处于不同去噪难度，增加训练多样性，避免一个任务主导梯度。

---

### 阶段 4：pre_dit 预处理

```python
video_pre = video_expert.pre_dit(
    x=latents_noisy,           # [2, 16, 5, 14, 14]
    timestep=timestep_video,   # [2]
    context=context,           # [2, L, 4096]
    ...
)
# video_pre["tokens"]: [2, S_v, 3072]
#   S_v = 5 × 14 × 14 = 980（时空 patch 化后的 token 数）
# video_pre["freqs"]:  [S_v, 1, rope_dim]（3D RoPE 编码）
# video_pre["t_mod"]:  [2, 6, 3072]（时间调制 6 个参数）

action_pre = action_expert.pre_dit(
    action_tokens=noisy_action,  # [2, 16, action_dim]
    timestep=timestep_action,    # [2]
    context=context,             # [2, L, 4096]
)
# action_pre["tokens"]: [2, 16, 1024]
# action_pre["t_mod"]:  [2, 6, 1024]
```

---

### 阶段 5：MoT 混合注意力（30 层）

```python
attention_mask = _build_mot_attention_mask(
    video_seq_len=980,
    action_seq_len=16,
    video_tokens_per_frame=196
)
# attention_mask: [996, 996]（bool，True=可见）

tokens_out = mot.forward(
    embeds_all={"video": [2,980,3072], "action": [2,16,1024]},
    attention_mask=[996, 996],
    freqs_all=..., context_all=..., t_mod_all=...
)
# 每层内部：concat → [2, 996, 3072] → flash_attention → split
# tokens_out["video"]:  [2, 980, 3072]
# tokens_out["action"]: [2, 16,  1024]
```

详见 [[02-MoT混合Transformer]]

---

### 阶段 6：post_dit + 损失计算

```python
pred_video  = video_expert.post_dit(tokens_out["video"],  video_pre)
# [2, 48, 4, 14, 14]（去掉条件帧后 → 4 个 latent 时间步）

pred_action = action_expert.post_dit(tokens_out["action"], action_pre)
# [2, 16, action_dim]
```

#### 视频损失

```python
# 逐 token 计算 MSE，在通道和空间维度上平均
video_loss_token = F.mse_loss(pred_video, target_video, reduction="none")
                   .mean(dim=(1, 3, 4))
# [2, 4]  ← 每个样本每个 latent 时间步的损失

# 排除 padding 帧（不同 episode 长度 pad 到相同长度）
valid_video  = (~image_is_pad_latent).float()   # [2, 4]
valid_sum    = valid_video.sum(dim=1).clamp(min=1.0)
loss_video   = (video_loss_token * valid_video).sum(1) / valid_sum  # [2]

# 流匹配时间步权重：高 t（高噪声）损失权重更大
weight_video     = train_video_scheduler.training_weight(timestep_video)  # [2]
loss_video_final = (loss_video * weight_video).mean()  # 标量
```

#### 动作损失

```python
action_loss_token = F.mse_loss(pred_action, target_action, reduction="none")
                    .mean(dim=2)
# [2, 16]  ← 每个样本每个动作时间步的损失

valid_action      = (~action_is_pad).float()  # [2, 16]
valid_sum         = valid_action.sum(dim=1).clamp(min=1.0)
action_loss_final = (action_loss_token * valid_action).sum(1) / valid_sum
action_loss_final = (action_loss_final * weight_action).mean()
```

#### 总损失

```python
loss = λ_video * loss_video_final + λ_action * loss_action_final
# 默认 λ_video=0（只训练动作，见 fastwam.yaml：loss.lambda_action=1.0，无 lambda_video）
# 或同时训练时两者都有权重
```

---

## 3. Padding Mask 详解

不同 episode 长度不同，短的会 pad 到最大长度。

```python
# image_is_pad: [B, num_frames]  True = 这一帧是填充的
# 对应到 latent 时间步：
tail_is_pad          = image_is_pad[:, 1:]      # 去掉第一帧
latent_tail_is_pad   = tail_is_pad.view(B, -1, temporal_factor).all(dim=2)
# temporal_factor=4：4 个原始帧 → 1 个 latent 步，全是 pad 才标记为 pad

# 计算 loss 时，pad 位置不计入
valid_sum = valid.sum(dim=1).clamp(min=1.0)  # 避免除零
loss = (loss_token * valid).sum(1) / valid_sum
```

---

## 4. 可训练参数

```python
# trainer.py: __init__
trainable_params = list(model.dit.parameters())   # MoT（包含两个专家 DiT blocks）
if model.proprio_encoder is not None:
    trainable_params += list(model.proprio_encoder.parameters())

# 冻结的：
# - vae（编解码器）
# - text_encoder（T5）
# - tokenizer
```

---

## 5. 完整数据流汇总

```
原始数据 [B, 3, 16, 224, 224]
    ↓ VAE encode（冻结）
latents [B, 16, 5, 14, 14]
    ↓ 流匹配加噪（独立 timestep）
noisy_latents [B, 16, 5, 14, 14]  +  noisy_action [B, T, action_dim]
    ↓ pre_dit（各自）
video_tokens [B, 980, 3072]        +  action_tokens [B, T, 1024]
    ↓ MoT 30层混合注意力
video_tokens_out [B, 980, 3072]    +  action_tokens_out [B, T, 1024]
    ↓ post_dit（各自）
pred_video [B, 48, 4, 14, 14]     +  pred_action [B, T, action_dim]
    ↓ MSE vs target（加权）
loss_video（标量）                 +  loss_action（标量）
    ↓
total_loss = λ_v × loss_video + λ_a × loss_action
```

---

## 关联笔记

- [[01-ActionDiT动作专家]]：`action_expert.pre_dit()` 和 `post_dit()` 的内部细节
- [[02-MoT混合Transformer]]：`mot.forward()` 的 30 层混合注意力
- [[04-推理流程与KV缓存]]：推理时如何只用动作专家
