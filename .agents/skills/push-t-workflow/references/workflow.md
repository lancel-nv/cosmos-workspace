# 复现步骤

## 1. 获取代码与选择环境

```bash
git clone https://github.com/lancel-nv/cosmos-workspace.git
cd cosmos-workspace
export WORKSPACE="$PWD"
```

固定本仓库 commit；`upstream-lock.json` 记录四个上游版本。不要在这个 monorepo 内执行上游的
`git submodule update` 来代替依赖安装，使用 `scripts/bootstrap_dependencies.py`。

处理已有 raw 不需要 Isaac Sim。需要 Linux、ffmpeg/ffprobe，以及三个独立环境：

| 环境 | 验证过的关键版本/入口 |
| --- | --- |
| LeRobot 转换 | Python 3.11，lerobot 0.4.4，torch 2.7.0 CPU，torchvision 0.22.0 CPU，av 15.1.0，numpy 2.2.6，pyarrow 26.0.1 |
| Curator | 仓库 `pixi.toml` / `pixi.lock` 的 default 环境，含 Ray、Lance、PyArrow、PyAV |
| Framework | 仓库 `pyproject.toml`，Python 3.13，训练依赖及对应 CUDA group |

LeRobot 示例安装：

```bash
uv venv --python 3.11 RoboDojo/.venv-lerobot
uv pip install --python RoboDojo/.venv-lerobot/bin/python \
  torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python RoboDojo/.venv-lerobot/bin/python \
  lerobot==0.4.4 av==15.1.0 numpy==2.2.6 pyarrow==26.0.1
python scripts/bootstrap_dependencies.py --component cosmos-curator
(cd cosmos-curator && pixi install -e default)
(cd cosmos-framework && uv sync --all-extras --group=cu130-train)
```

框架 CUDA 组须匹配新集群驱动；不要升级或替换其他任务的系统驱动。
使用集群提供的容器也可以，保证整个 work 目录在 Ray driver/worker 上使用同一绝对路径。
直接运行 Curator Python 时需要它的 ffmpeg 等工具在 PATH，必要时在 `pixi shell -e default` 中运行。
运行安装命令前核对上游 `cosmos-framework/docs/setup.md` 和集群策略；本仓库不自动安装系统驱动。

## 2. 下载并校验 raw

HF 仓库是私有的 `lancel-nv/robodojo-push-t-raw`。使用有访问权限的账号认证，Token 不写进脚本。
使用发布完成后记录在 `data-release.json` 的 HF revision，不使用浮动 main 做正式实验。

```bash
export PATH="$WORKSPACE/RoboDojo/.venv-lerobot/bin:$PATH"
hf auth login
python scripts/download_raw.py --output "$WORKSPACE/data/push-t-raw"
RoboDojo/.venv-lerobot/bin/python scripts/raw_release.py verify \
  --root "$WORKSPACE/data/push-t-raw" --decode
```

预期502条：200成功、301超时、1任务失败；317训练、185验证，分组单位为 layout。
每条只有 metadata、trajectory.npz 和三路 RGB。HF 不包含 LeRobot、Curator、训练 bundle 或 checkpoint。

## 3. 准备 tokenizer 和机器人校准资产

处理数据需要 tokenizer 文件和 ARX CAD 校准源，不需要先下载整个视频模型权重：

```bash
hf download nvidia/Cosmos3-Nano --revision 45666784803b3a7ad577e0def405f9afcc3bddb2 \
  --include 'tokenizer*' 'vocab.json' 'merges.txt' 'special_tokens_map.json' \
  --local-dir "$WORKSPACE/data/tokenizer"
hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
  --revision 1127bda94f172d757386945af3eb3a6754d1a4e6 \
  --include 'Assets/Robots/x5/**' --local-dir "$WORKSPACE/data/robot-assets"
```

检查 tokenizer 目录中有 `vocab.json`、`merges.txt`、`tokenizer_config.json`。
机器人目录需要 `robot_config.yml`、`X5A.urdf`、`ARX.usd`、`meshes/link7.STL` 和 `meshes/link8.STL`。
不要手改这些文件来绕过校准检查。

## 4. raw → LeRobot → Curator → 单主相机 bundle

```bash
export PATH="$WORKSPACE/cosmos-curator/.pixi/envs/default/bin:$PATH"
python scripts/process_raw.py \
  --raw "$WORKSPACE/data/push-t-raw" \
  --work "$WORKSPACE/data/processed" \
  --lerobot-python "$WORKSPACE/RoboDojo/.venv-lerobot/bin/python" \
  --curator-python "$WORKSPACE/cosmos-curator/.pixi/envs/default/bin/python" \
  --framework-python "$WORKSPACE/cosmos-framework/.venv/bin/python" \
  --tokenizer "$WORKSPACE/data/tokenizer" \
  --robot-assets "$WORKSPACE/data/robot-assets/Assets/Robots/x5"
```

默认只打印命令；检查路径后在同一命令末尾加 `--execute`。各输出目录必须不存在。
脚本执行8步，日志分别在 work 下：export、curate、augment、staging、bundle、audit-short、windows、audit-final。
遇错停止，不修改下载的 raw，不启动模型训练。检查失败日志后用新 work 名重试，避免把 partial 当成成功。

最终产物为 `data/processed-head-var593/`，含 train、validation、export_receipt 和 readiness。
短窗口 bundle 是构建中间产物；最终窗口覆盖所有合法起点，最长592动作/593RGB，动作数为16的倍数，不补RGB帧。
每条样本只读取主相机；训练原生预处理按480档位保持比例缩放/反射填充，实际尺寸由输入比例决定，不固定为旧拼图640×640。
任务 prompt 保留原文，采用纯文本和0.1 CFG文本 dropout；没有重新生成论文中的初始场景描述。

保留 `.npz` transition 回填：`next.*` 标记动作后的转移，终止观察的 `action_valid=false`。
成功事件可能发生在最后控制步内的物理子步，不能仅用最后一张图重新定义标签。

## 5. Stage 0（仅在另行要求训练时）

准备原始 `nvidia/Cosmos3-Nano` 权重、Wan2.2 VAE，以及 Framework 原生 DCP 格式 initializer。
按 `cosmos-framework/docs/training.md` 和 `cosmos_framework/scripts/convert_model_to_dcp.py --help` 转换；
不能直接把 Hugging Face 模型目录当 DCP，也不要使用旧的拼图训练 checkpoint。

```bash
export BUNDLE_ROOT="$WORKSPACE/data/processed-head-var593"
export COSMOS_SIM_BIMANUAL_GA_CHECKPOINT=/your/checkpoints/Cosmos3-Nano-dcp
export VIDEO_VAE_PATH=/your/checkpoints/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=/your/outputs
bash scripts/train_stage0.sh dryrun
```

使用官方 `sim_bimanual_stage0_bidir.toml`：双向 Stage0、首帧视觉条件、动作条件、仅 RGB 目标；
新 ARX 域为33，配置34个域，保持官方新动作投影初始化/EMA逻辑。
入口默认8卡，CP4/FSDP8、400000 token budget、BS上限1/pack；先显式小样训练再测容量。
默认35000步来自公开代码，不代表此小数据集的最佳步数；不直接沿用之前16小时的估算。
公开代码Stage0使用480档位，论文表2为256/480混合；我们跟随固定commit的公开代码，
此外25FPS、593帧上限、commanded link6 EEF和原任务文本是自定义适配，不是论文完整复现。
尚未验证新集群的模型加载/反向传播时，不要把数据 readiness 当作训练可用性或效果证明。

只有用户授权后，才执行例如 `MAX_ITER=2 SAVE_ITER=2 RUN_NAME=head-smoke bash scripts/train_stage0.sh train`。
正式训练使用新的 run name；不能同时运行或自动停止其他GPU任务。

## 6. 可选：生成新 raw 数据

现有数据迁移无需执行本节。生成新数据时：

```bash
python scripts/bootstrap_dependencies.py --component RoboDojo
HF_REVISION=1127bda94f172d757386945af3eb3a6754d1a4e6 bash RoboDojo/scripts/init_assets.sh
```

单独配置 Isaac Sim 环境；原采集验证版本为 Python3.12.16、IsaacSim6.0.1.0、IsaacLab0.54.3、
torch2.11.0+cu128、numpy2.3.1、warp-lang1.13.0。IsaacLab补丁由 bootstrap 应用。
上游 RoboDojo 的一般安装器可能选择别的 Isaac Sim 版本，不能把它的默认环境当成此采集版本。
修改 `RoboDojo/configs/push_t_collection.json` 中的环境路径与输出根目录；建议复制为新的配置，
设置新的seed/start/run名称。不要改写已有raw或复用旧run指纹。

```bash
cd RoboDojo
bash scripts/push_t_pipeline.sh --config configs/push_t_collection.json --run my-push-t doctor
bash scripts/push_t_pipeline.sh --config configs/push_t_collection.json --run my-push-t smoke
```

检查成功/失败、末帧、三相机、曝光、状态与动作对齐。人工确认后才 `approve --reviewed`，然后 `collect`、`validate`。
配额是至少500条、至少200成功、至少201失败，可能超过500条；不复制轨迹、不修改成功阈值凑配额。
采集代码保留房间/灯光清理、相机同步、成功子步证据和曝光门禁。异常 partial 不计入训练数据。
停止在 raw 即可；处理使用第4节的独立入口，不需要执行旧机器的冻结工作流。

## 7. 验证与发布

源码测试：

```bash
python -m unittest discover -s tests -v
cd cosmos-framework
COSMOS_INTERNAL=0 COSMOS_TRAINING=1 .venv/bin/python -m pytest \
  tests/robodojo_eef_data_test.py tests/robodojo_window_policy_test.py -q
```

发布前检查 `git ls-files`：禁止数据、权重、`.env`、密码/Token、运行日志进入Git。
数据发布默认私有；用户明确同意公开后才能上传到公开HF仓库。
原始数据不覆盖：发布新的数据commit并记录校验和；正式训练绑定代码commit、数据commit、bundle收据和训练配置。
若 `data-release.json` 的 revision 为空，表示发布仍待确认，下载入口会拒绝回退到 main。
