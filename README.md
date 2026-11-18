# Cosmos Workspace

用于复现 RoboDojo ARX-X5 Push-T 数据采集、raw 数据处理和单主相机 Bimanual Stage 0 的代码仓库。
仅保存维护中的代码、配置、测试和工作流；不包含数据集、模型权重、旧实验输出或机器凭据。

## 入口

| 任务 | 入口 |
| --- | --- |
| 给 coding agent 的完整工作流 | [.agents/skills/push-t-workflow/SKILL.md](.agents/skills/push-t-workflow/SKILL.md) |
| 人工逐步复现 | [工作流步骤](.agents/skills/push-t-workflow/references/workflow.md) |
| 下载的数据 | [HF raw 数据集](https://huggingface.co/datasets/lancel-nv/robodojo-push-t-raw) |
| 校验 raw | `python scripts/raw_release.py verify --root data/push-t-raw` |
| 处理计划（不执行） | `python scripts/process_raw.py --help` |
| Stage 0 配置检查 | `bash scripts/train_stage0.sh dryrun` |

## 结构

- `RoboDojo/`：仿真源码、最终采集修复、raw→LeRobot 和 transition 回填代码。
- `cosmos-curator/`：固定版本的 Curator 源码，使用原生 `robot-action-split`。
- `cosmos-framework/`：固定版本的 Framework 源码、ARX EEF 适配和单相机 bundle 导出/验证。
- `cosmos-evaluator/`：固定版本的评估工具源码；其模型和示例数据按需从上游获取，不在 Git 中存放。
- `scripts/`：跨集群入口；路径由命令参数/环境变量指定，不绑定原机器。
- `upstream-lock.json`、`patches/`：上游 commit、可选依赖和兼容补丁。

四个主项目直接纳入此仓库；没有复制各自的 Git 历史。保留上游运行时代码以避免裁剪隐藏依赖，
省去大模型、媒体、benchmark 数据、上游 agent 技能库和本项目中间探索脚本。
RoboDojo 的 IsaacLab、cuRobo、XPolicyLab 和 Curator 的 Xenna 是固定版本外部依赖：

```bash
python scripts/bootstrap_dependencies.py --component RoboDojo
python scripts/bootstrap_dependencies.py --component cosmos-curator
```

仅在使用 Evaluator 对应模型时执行 `--component cosmos-evaluator`，会下载上游资源。
四个项目保留各自 LICENSE/ATTRIBUTIONS；本仓库不改变上游代码或资产的许可证。

## 数据与训练边界

HF 发布范围仅为有效的 502 条 raw 轨迹：200 成功、302 失败，三路原始相机、实测状态、命令、终止信息和 prompt。
发布状态和固定版本以 `data-release.json` 为准；revision 为空时尚未完成上传。
raw 不是训练 bundle；用户在新集群上运行处理脚本。不要把原始动作替换成未来实测状态，也不要训练终止观察行的占位动作。

本项目的 Bimanual 训练使用 `cam_head` 单视角，不拼接腕部相机；首帧和动作是条件，后续 RGB 是监督目标。
沿用官方 `sim_bimanual_stage0_bidir.toml`，但 ARX embodiment、25 FPS、593帧上限和任务文本仍是本项目选择，不声称论文逐项复现。
训练脚本默认 dry-run；实际训练需显式 `train`，从原始 Cosmos3-Nano DCP 初始化，不自动续训历史权重。

各环境分离。安装、数据版本锁定、单视角检查和训练前检查见工作流步骤。

## 已验证的范围

- 全量 raw：502条、2510个原始文件的校验和，以及逐条状态/时序/视频检查通过。
- 两条样本（训练集成功、验证集超时各一条）通过完整8步处理与最终 materialized audit；实际单主相机 RGB 为544×736。
- ARX EEF/窗口单元测试23项、workspace入口测试5项、skill格式验证通过。
- 独立目录中的官方Stage0配置 dry-run通过；没有启动模型训练，也不代表新集群已通过模型加载/反向传播。
