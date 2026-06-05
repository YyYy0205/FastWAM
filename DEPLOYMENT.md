# FastWAM 部署文档

## 目录

- [环境要求](#环境要求)
- [环境安装](#环境安装)
- [模型准备](#模型准备)
- [数据集准备](#数据集准备)
- [训练部署](#训练部署)
- [推理部署](#推理部署)
- [评测部署](#评测部署)
- [目录结构说明](#目录结构说明)
- [常见问题](#常见问题)

---

## 环境要求

### 硬件要求

| 场景 | 最低配置 | 推荐配置 |
|------|---------|---------|
| LIBERO 训练 | 8× GPU (≥40GB each) | 8× A100 80GB |
| RoboTwin 训练 | 32× GPU | 64× A100 80GB |
| 单卡推理评测 | 1× GPU (≥24GB) | 1× A100/H100 |
| 多卡评测 | 4× GPU | 8× A100 80GB |

### 软件要求

| 软件 | 版本 |
|------|------|
| Python | 3.10 |
| CUDA | 12.8 |
| PyTorch | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| OS | Linux (推荐 Ubuntu 20.04/22.04) |

---

## 环境安装

### 步骤 1：创建 conda 环境

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
```

### 步骤 2：安装 PyTorch（必须指定 CUDA 版本）

```bash
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

### 步骤 3：安装项目依赖

```bash
cd /path/to/FastWAM
pip install -e .
```

### 步骤 4（可选）：安装评测环境

**LIBERO 评测：**
```bash
# 从官方仓库安装 LIBERO 环境
# https://github.com/Lifelong-Robot-Learning/LIBERO
pip install mujoco==3.3.2
```

**RoboTwin 评测：**
```bash
# 参考官方文档安装 RoboTwin 环境
# https://github.com/RoboTwin-Platform/RoboTwin
# 安装完成后创建策略软链接
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" \
  "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
```

---

## 模型准备

此步骤训练和推理都需要。

### 步骤 1：设置模型目录

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

> 可将该环境变量写入 `~/.bashrc` 或在每次训练脚本前设置。

### 步骤 2：预生成 ActionDiT 主干权重

ActionDiT 的初始权重由 Wan2.2 视频 DiT 线性插值得到，需要提前生成：

```bash
# 标准 FastWAM（uncond）
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

运行成功后，输出文件路径为：
```
checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

> 此步骤会自动从 Hugging Face 下载 `Wan-AI/Wan2.2-TI2V-5B` 和 `Wan-AI/Wan2.1-T2V-1.3B`。
> 如网络受限，可提前手动下载至 `checkpoints/` 目录。

---

## 数据集准备

### LIBERO 数据集

```bash
mkdir -p data/libero_mujoco3.3.2
cd data/libero_mujoco3.3.2

# 从 https://huggingface.co/datasets/yuanty/LIBERO-fastwam 下载全部 tar.gz 文件
# 下载完成后解压所有文件
for f in *.tar.gz; do
  tar -xzf "$f"
done
```

解压后目录结构：
```
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

### RoboTwin 数据集

```bash
mkdir -p data/robotwin2.0
cd data/robotwin2.0

# 从 https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam 下载所有分卷
# 合并并解压
cat robotwin2.0.tar.gz.part-* | tar -xzf -
```

解压后目录结构：
```
data/robotwin2.0/
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

---

## 训练部署

### 步骤 1：预计算 T5 文本 Embedding 缓存

训练前必须提前计算文本 embedding，避免训练中反复调用 T5 模型：

```bash
# LIBERO（单卡）
python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4

# LIBERO（多卡加速）
torchrun --standalone --nproc_per_node=8 \
  scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4

# RoboTwin（单卡）
python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4
```

### 步骤 2：首次训练时关闭 norm 统计复用

**编辑对应的 `configs/data/*.yaml`，将 `pretrained_norm_stats` 设为 `null`：**

```yaml
# configs/data/libero_2cam.yaml（首次训练前）
pretrained_norm_stats: null
```

### 步骤 3：启动训练

**单节点 LIBERO 训练（8 GPU）：**

```bash
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4
```

**多节点 RoboTwin 训练（示例：4 节点，每节点 16 GPU）：**

```bash
# 节点 0（主节点）
NNODES=4 NODE_RANK=0 MASTER_ADDR=<主节点IP> \
  bash scripts/train_zero1.sh 16 task=robotwin_uncond_3cam_384_1e-4

# 节点 1-3（工作节点）
NNODES=4 NODE_RANK=<1|2|3> MASTER_ADDR=<主节点IP> \
  bash scripts/train_zero1.sh 16 task=robotwin_uncond_3cam_384_1e-4
```

### 训练输出

训练产物保存在：
```
runs/{task_name}/{run_id}/
├── checkpoints/          # 模型权重（每 save_every 步保存一次）
├── dataset_stats.json    # 动作/状态归一化统计（首次训练后生成）
└── logs/                 # 训练日志
```

### 步骤 4：后续训练复用 norm 统计

首次训练后更新 `pretrained_norm_stats` 路径：
```yaml
pretrained_norm_stats: ./runs/libero_uncond_2cam224_1e-4/<run_id>/dataset_stats.json
```

### 关键训练超参数

| 参数 | LIBERO 默认值 | 说明 |
|------|-------------|------|
| `batch_size` | 16 | 每 GPU batch size |
| `learning_rate` | 1e-4 | 初始学习率（cosine 衰减） |
| `num_epochs` | 10 | 训练轮数 |
| `gradient_accumulation_steps` | 1 | 梯度累积步数 |
| `mixed_precision` | bf16 | 混合精度（推荐 bf16） |
| `weight_decay` | 1e-2 | AdamW 权重衰减 |
| `max_grad_norm` | 1.0 | 梯度裁剪阈值 |

---

## 推理部署

### 下载官方预训练权重

```bash
pip install -U huggingface_hub

huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release
```

下载后目录结构：
```
checkpoints/fastwam_release/
├── libero_uncond_2cam224.pt
├── libero_uncond_2cam224_dataset_stats.json
├── robotwin_uncond_3cam_384.pt
└── robotwin_uncond_3cam_384_dataset_stats.json
```

### 在代码中加载模型进行推理

```python
from fastwam.runtime import create_fastwam
from omegaconf import OmegaConf
import torch

# 加载模型配置
cfg = OmegaConf.load("configs/model/fastwam.yaml")

# 创建模型
model = create_fastwam(cfg, device="cuda", torch_dtype=torch.bfloat16)

# 加载微调权重
ckpt = torch.load("checkpoints/fastwam_release/libero_uncond_2cam224.pt", map_location="cpu")
model.load_state_dict(ckpt, strict=False)
model.eval()

# 推理（仅动作专家，无需视频生成）
result = model.infer_action(
    prompt="pick up the mug",
    input_image=first_frame_tensor,   # [1, 3, H, W], 归一化到 [0,1]
    action_horizon=16,                # 预测动作步数
    num_inference_steps=20,           # 扩散去噪步数
    seed=42,
)
# result["action"]: [1, action_horizon, action_dim]
```

---

## 评测部署

### LIBERO 评测

```bash
# 使用官方权重
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus=8

# 使用自训练权重
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./runs/libero_uncond_2cam224_1e-4/<run_id>/checkpoints/step_xxxxx.pt \
  MULTIRUN.num_gpus=4
```

### RoboTwin 评测

```bash
# 使用官方权重（unseen 指令，默认）
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8

# 使用 seen 指令（理论上提升 1-2 点）
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.instruction_type=seen \
  MULTIRUN.num_gpus=8
```

### 评测参数说明

| 参数 | 说明 |
|------|------|
| `MULTIRUN.num_gpus` | 并行评测 GPU 数量，默认 8，可改为 4 |
| `EVALUATION.instruction_type` | `unseen`（默认）或 `seen` |
| `EVALUATION.skip_get_obs_within_replan` | `true` 加速评测但降低视频质量，默认 `true` |

---

## 目录结构说明

```
FastWAM/
├── checkpoints/              # 预训练权重存放处
│   ├── ActionDiT_*.pt        # 预生成的 ActionDiT 主干
│   └── fastwam_release/      # 官方发布权重
├── configs/
│   ├── data/                 # 数据集配置（摄像头数、分辨率、归一化等）
│   ├── model/                # 模型架构配置（维度、层数、调度器参数等）
│   ├── task/                 # 任务配置（组合 data+model+训练超参）
│   ├── train.yaml            # 基础训练配置
│   ├── sim_libero.yaml       # LIBERO 评测配置
│   └── sim_robotwin.yaml     # RoboTwin 评测配置
├── data/                     # 数据集存放处
├── experiments/
│   ├── libero/               # LIBERO 评测管理器
│   └── robotwin/             # RoboTwin 评测管理器
├── runs/                     # 训练输出（权重、日志、dataset_stats）
├── scripts/
│   ├── train.py              # 训练入口（Hydra）
│   ├── train_zero1.sh        # DeepSpeed ZeRO-1 启动脚本
│   ├── preprocess_action_dit_backbone.py
│   └── precompute_text_embeds.py
├── src/fastwam/              # 核心代码包
└── third_party/RoboTwin/     # RoboTwin 评测环境代码
```

---

## 常见问题

### Q1：`DIFFSYNTH_MODEL_BASE_PATH` 未设置导致模型下载到错误位置

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```
建议写入 `.bashrc` 或训练脚本顶部。

### Q2：多节点训练 run_id 不同步

`train_zero1.sh` 通过 `TCPStore` 在节点间同步 `run_id`，确保 `MASTER_ADDR` 和 `RUN_ID_SYNC_PORT`（默认 `MASTER_PORT+11`）端口可达。超时时间由 `RUN_ID_SYNC_TIMEOUT`（默认 180s）控制。

### Q3：首次训练报找不到 `dataset_stats.json`

将 `configs/data/*.yaml` 中的 `pretrained_norm_stats` 设为 `null`，首次训练会自动生成该文件。

### Q4：评测时 GPU 数量不够 8 张

传入 `MULTIRUN.num_gpus=4`（或更小值）即可：
```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=<ckpt_path> \
  MULTIRUN.num_gpus=4
```

### Q5：RoboTwin 评测视频帧率很低

这是正常现象。默认开启了 `EVALUATION.skip_get_obs_within_replan=true` 以加速评测。如需完整渲染视频，设为 `false`（会显著降低评测速度）。

### Q6：想用 seen 指令提升 RoboTwin 性能

```bash
EVALUATION.instruction_type=seen
```
论文默认使用 unseen 指令，seen 指令理论上提升 1-2 个百分点。
