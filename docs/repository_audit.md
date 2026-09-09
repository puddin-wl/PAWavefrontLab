# 第一轮仓库审计

审计基线：`agent/pytest-test-log`，提交 `c04aa60`。整理前测试结果为
`55 passed, 10 skipped`。

## 整理前问题

- README 同时推荐五步静态 workflow 和旧 `run_dataset.py`，新旧入口竞争。
- README、PROJECT_ENTRYPOINTS 同时承担教程、机器环境、架构和入口说明，职责重复。
- `run_single_image.py`、`run_dataset.py`、单 BIN 去噪 direct-run 设置含本机绝对路径。
- `log/` 实为实验研究记录；`noise_cause_review/` 实为历史探索，但目录身份不够醒目。
- 正式 packed-12 位解码在 TIFF 与去噪代码中重复，存在行为漂移风险。
- `pytest` 与 runtime 依赖混在一起；仓库没有最小 CPU CI。
- 9 张已跟踪 PPT PNG 约 22.5 MiB，需限制为有来源说明的 canonical figures。
- Windows 下载附属文件 `*:Zone.Identifier` 没有被忽略。

## 第一轮分类

| 类别 | 文件/目录 |
| --- | --- |
| NeuWS 核心 | `networks.py`, `dataset.py`, `utils.py`, `recon_exp_data.py` |
| PAWavefrontLab 光学/评价核心 | `optics.py`, `evaluation.py` |
| 正式预处理 | `preprocessing/` |
| 正式完整 workflow | `workflows/static_simulation/` 与 PROJECT_ENTRYPOINTS 中的真实主链 |
| 单功能 command | `tools/`, `savetif/bin_to_tiff.py` |
| direct-run wrapper | `run_single_image.py`, `motor_crosstalk_denoise/run_denoise.py` 的配置区 |
| deprecated compatibility | `run_dataset.py`, `tools/prepare_denoised_real_dataset.py` 的旧方法用途 |
| 实验记录 | `log/`（由 `docs/experiments/` 导航） |
| 历史探索 | `noise_cause_review/` |
| canonical 展示图 | `motor_crosstalk_denoise/ppt_assets/` 已有 9 图 |
| 本地生成物 | `data/`, `outputs/`, `vis/`, BIN/MAT/TIFF/NPY 与 results/runs |

## 故意延后的工作

- 不把根模块整体迁到 `src/` package，不大改 NeuWS import。
- 不在第一轮移动 `noise_cause_review/`；未来整体迁移到 archive 时再修复历史脚本。
- 不在第一轮批量移动 `log/`；先提供稳定文档入口，未来用独立 rename 阶段处理。
- 不删除既有 PPT 图片；先固定来源和新增资产规则，避免失去已使用的汇报材料。
- 不调整已验证的网络、光学、去噪、评价数学与实验参数。

## CI 评估

没有在第一轮加入 GitHub Actions。原始 `LICENSE.txt` 限制复制和向第三方提供上游
NeuWS 软件；在没有确认代码托管与 runner 是否满足授权条件前，让第三方托管 runner
自动取得并执行仓库内容存在合规风险。当前采用本地 CPU `python -m pytest -q` 作为
统一验证入口。若后续确认仓库/runner 为获准的私有基础设施，可再加入只安装
`requirements-dev.txt`、不要求 CUDA 的最小 CI。
