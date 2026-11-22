# 机器与数据

本文件回答三个问题：每台机器做什么、东西放在哪、某个数据或权重在哪里。

- 运行记录：在项目仓库里用 `lab runs`（先自动 git pull）
- 各机器做过什么：`lab journal`
- 同步：`lab sync`
- 方法细节：skill `lab-machines`；项目结构：skill `research-project`

## 机器

| 标识 | 角色 | 连接（从 Mac） | 总台仓库路径 | 跑什么 |
|---|---|---|---|---|
| mac | 指挥台 | 本机 | `/Users/max/codebase/coding_agent_repro` | 规划、写代码、分析；只跑 L0 和 ≤5 分钟的 CPU smoke |
| pc | 数据中心 + RTX 3060（实际可用约 6 GB） | `ssh pc`（WSL，2222 端口） | `~/lab/coding_agent_repro` | L1/L2，用 tmux 运行 |
| cluster | 大算力（Slurm） | `ssh bme_login2`；传输走 `bme_xfer` | `/public_bme/home/maxquan/Projects/coding_agent_repro` | L3，遵循 bme-cluster |

项目仓库：Mac `/Users/max/codebase/<项目>`；pc `~/lab/<项目>`；集群 `/public_bme/home/maxquan/Projects/<项目>`。运行产物默认在项目的 `runs/` 下（被 git 忽略），也可以用环境变量 `LAB_RUNS_ROOT` 指到别处。

总仓库 `research_workplace`（git@github.com:CristinaAsuna/research_workplace.git，子项目在 `paper_repro/`、`idea_verify/` 下）：Mac `/Users/max/codebase/research_workplace`；pc `~/lab/research_workplace`；集群 `/public_bme2/bme-cuizhm/maxquan/research_iter/research_workplace`。

## 路由规则

1. 无 GPU 且 ≤5 分钟 → mac。
2. 单卡、显存 ≤6 GB、耗时 ≤数小时 → pc。运行前用 `nvidia-smi` 确认空闲显存，Zotero 翻译模型会占用一部分。
3. 更大的显存、多卡或长时间训练 → cluster。
4. 数据已经在哪台机器上，优先在那台跑。

## 路径约定

| 机器 | 运行产物 | 数据集（`LAB_DATA_DIR`） | 权重（`LAB_MODELS_DIR`） | 冷归档 |
|---|---|---|---|---|
| mac | `<项目>/runs/`（只放 smoke） | 不放 | 不放 | — |
| pc | `<项目>/runs/`（或 `LAB_RUNS_ROOT=~/lab/runs`） | `~/lab/assets/datasets/<名称>/`（ext4，读写快） | `/mnt/d/vscode_workplace/brainstorm/models`（HF 缓存在其下的 `huggingface/`） | `/mnt/d/lab-archive` |
| cluster | `<项目>/runs/`（或 `LAB_RUNS_ROOT`） | `/public_bme/home/maxquan/Datasets/Data/<名称>/`（CBCT 旧数据在 `Datasets/bme/`） | `/public_bme/home/maxquan/Datasets/models/` | — |

- Python 环境：
  - pc：`~/lab/envs/main`（uv，torch CUDA）。
  - cluster：`lab` 用 uv 安装的 Python 3.12；实验环境按项目用 conda 或 uv 另建。
- 下载：pc 和 cluster 都设置 `HF_ENDPOINT=https://hf-mirror.com`，只对当次命令生效。

## 传输方式

- PC ↔ 集群：在 WSL 上 `rsync -aH --partial <源>/ bme_xfer:<目标>/`，不经过 Mac。
- Mac → PC：只用于迁移旧数据，用 `/opt/homebrew/bin/rsync`。
- 小结果和图：一律通过 git。

## 资产表

新下载或迁移的数据和权重都要登记在这里。

| 名称 | 来源 / revision | 位置（机器:路径） | 大小 | 使用的项目 |
|---|---|---|---|---|
| PC 原有权重（dit、sit 的 `.pt`，以及 huggingface/、torchhub/ 中原有的部分） | 历史下载 | pc:`/mnt/d/vscode_workplace/brainstorm/models` | 18 GB | — |
| HF hub 缓存（dinov2 base/large/with-registers、siglip、clip b16/b32、rad-dino、mae、PubMedCLIP、sd-vae-ft-ema、SD-x4-upscaler、SD1.5、RAE、SiT 等） | 2026-11-14 从 Mac 迁入，revision 见 brainstorm `assets/registry.json` | pc:`…/models/huggingface/`（HF_HOME 指向此目录，不要加 `hub/`） | 14 GB | — |
| irepa-collections | 同上 | pc:`…/models/irepa-collections/` | 22 GB | — |
| torch hub（checkpoints、pyiqa） | 同上 | pc:`…/models/torchhub/` | 0.8 GB | — |
| diffusion-mechanisms、jit-back-to-basics、unet-v3（adm、sd15）、torchxrayvision | 同上 | pc:`…/models/<同名目录>/` | 14 GB | — |
| 数据集：aariz、montgomery、scikit-image、spair-71k、stable_sr_testsets | 同上 | pc:`~/lab/assets/datasets/<名称>/` | 8.3 GB | — |
| MNIST（IDX 原始训练集与测试集，含 gzip 副本） | PC 原有数据，下载来源及 revision 未记录 | pc:`/mnt/d/Datasets/MNIST/raw/` | 63.5 MiB | 2026-11-16-sra-selfflow-mnist |
| 集群共享权重目录（`LAB_MODELS_DIR`；`huggingface/` 是 HF_HOME，`torch/` 是 TORCH_HOME，其他权重各占一个子目录） | 2026-11-21 新建 | cluster:`/public_bme/home/maxquan/Datasets/models/`（下文用 `⟨M⟩` 表示） | 21 GB | 所有项目 |
| SD1.5（diffusers 格式，sha256 见目录内 `model_manifest.json`） | 来源和 revision 未记录 | cluster:`⟨M⟩/stable-diffusion-v1-5/` | 3.8 GB | CBCT sd15_* |
| SD3.5 medium | ModelScope，revision fe81289a07ee5ebe2410fa7bcf275cb6910145a5 | cluster:`⟨M⟩/stable-diffusion-3.5-medium-modelscope-fe81289a/` | 16 GB | CBCT pre_trained_sft |
| sd-vae-ft-ema、sd-vae-ft-mse、sdxl-vae | stabilityai，revision f04b2c4b、31f26fde、6f5909a7（完整值见 CBCT `shared/pretrained_vae/weights_manifest.json`） | cluster:`⟨M⟩/sd-vae-ft-ema/`、`sd-vae-ft-mse/`、`sdxl-vae/` | 各 320 MB | CBCT |
| DINOv2 ViT-B/14（`dinov2_vitb14_pretrain.pth`） | torch.hub 官方权重 | cluster:`⟨M⟩/torch/hub/checkpoints/` | 331 MB | CBCT representation_analysis |
| ADM ImageNet-256 参考 batch（`VIRTUAL_imagenet256_labeled.npz`） | OpenAI guided-diffusion 官方参考 batch，SHA-256 `b32732719497e42660a9affb4a966068cba0855ac449b82015e34ec376d20758` | cluster:`/public_bme/home/maxquan/Datasets/Data/adm-ref-in256/` | 2,037,122,530 B | paper_repro/repa |
| ADM Inception 图（`classify_image_graph_def.pb`） | OpenAI guided-diffusion 官方评测图，SHA-256 `009d6814d1bc560d4e7b236e170e9b2d5ca6f4b57bd8037f6db05776204415c6` | cluster:`⟨M⟩/adm/` | 95,673,916 B | paper_repro/repa |
| DINOv2 torch.hub 源码 | facebookresearch/dinov2，revision `85a24602099d397264d5b30461ad7f3bfd726ca1` | cluster:`⟨M⟩/torch/hub/facebookresearch_dinov2_85a24602099d397264d5b30461ad7f3bfd726ca1/` | 2.3 MB | paper_repro/repa |
| REPA SiT-XL/2 4M 官方 checkpoint | sihyun-yu/REPA `67f7145` 的 `utils.py` 所指 Dropbox 文件，SHA-256 `bd62ad8b9ec5b12393b07745187bdd2f638af87d03236ee98faa5bc66ef0d0bc` | cluster:`⟨M⟩/repa/sit-xl2-4m.pt` | 2,733,044,564 B | paper_repro/repa（D-003） |
| ImageNet-1k（ILSVRC2012，Kaggle 版 `raw/ILSVRC` 原始数据；`train/`、`val/` 是各 1000 类的 ImageFolder 软链接视图） | 2026-08-16 下载，08-18 校验 1331167 张，5 张坏图已隔离到 `quarantine_invalid_links/`，清单 `invalid_images_4062340.tsv`；脚本在 `Datasets/scripts/` | cluster:`/public_bme/home/maxquan/Datasets/Data/imagenet/` | raw 157 GB（2026-11-22 du）；train 可读 1,281,162 张 | paper_repro/repa |
| CBCT 数据（`1017_raw_clear_no_landmark_50`、`cbct1175_mask`、`landmark3d`） | 来源未记录 | cluster:`/public_bme/home/maxquan/Datasets/bme/` | 未测 | CBCT |

2026-11-21 从 CBCT 项目 `CBCT_decouping_diffusion/artifacts/development/` 下的 `shared/pretrained_sd15/models/sd15`、`shared/pretrained_sft/models/modelscope_fe81289a…`、`shared/pretrained_vae/{ema,mse,sdxl}` 和 representation_analysis study 的 `exports/pretrained/` 移入，原位置留有指向新位置的软链接，CBCT 代码不用改。SD1.5 和三个 VAE 移动后按 manifest 重算 sha256 全部一致。

迁移记录（2026-11-14）：Mac → PC，`registry.json` 中登记的 648 个文件 sha256 全部一致；rsync 预演确认没有缺失或大小不一致的文件。Mac 上的原件（`/Users/max/codebase/brainstorm/assets/`）仍保留，由用户确认后自行删除。`…` 表示 `/mnt/d/vscode_workplace/brainstorm`。
