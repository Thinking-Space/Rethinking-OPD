# OPD 在 NSCC ASPIRE 2A 上的完整复现报告

> 完成日期：2026-09-17
>
> 上游代码基线：daydreamer17/OPD，commit ac26e38d6f1572eb027597b48a9f4e01f6915ef8
>
> 训练平台：NSCC ASPIRE 2A，单节点 4 × NVIDIA A100 40GB
>
> Student：deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
>
> Teacher：hbx/JustRL-DeepSeek-1.5B

## 1. 项目结论

本项目在不重写 OPD 核心算法的前提下，将上游仓库适配到 NSCC ASPIRE 2A 的 PBS 环境，完成了环境固化、模型与数据准备、单卡推理验证、四卡训练冒烟、连续 30–50 步验收、完整一轮正式训练、独立评测，以及 sampled-token 对照实验。核心训练仍沿用上游的 verl OPD 实现：Student 在当前策略上生成回答，冻结的 Teacher 在相同 token 轨迹上提供逐 token 分布信号，token_reward_direct 将该信号作为逐 token advantage 送入 actor 的策略梯度更新。

主线 top-k16 OPD 和 sampled-token OPD 均相对初始 Student 在 MATH-500 与 AIME24 上取得了配对 bootstrap 95% 置信区间下界大于 0 的可靠增益。主线模型达到 MATH-500 0.8160、AIME24 0.3271；sampled-token 模型达到 0.8160、0.3125。两种 OPD 变体之间的差异在两个任务上都未达到统计显著，因此本次单 seed 结果不能证明 top-k16 的最终准确率优于 sampled-token。sampled-token 完整训练耗时更短，GPU·h 比主线低 6.47%，但这只是一次同平台实测，不能推广为稳定的算法级加速结论。

最终结论为：

- 复现目标达成：两种 OPD 训练均相对初始 Student 获得可靠增益。
- 主线与 Teacher 的最终差异不显著：MATH-500 略低，AIME24 略高，但置信区间均跨 0。
- top-k16 与 sampled-token 的准确率差异不显著。
- AIME24 仍有很高的输出截断率，分数可信但方差和生成长度风险较高。
- 当前证据来自一个训练 seed；要比较两种变体的细微差异，必须补做多 seed 实验。

## 2. 目标、范围与实现边界

### 2.1 目标

项目需要回答四个问题：

1. 指定的 1.5B Student/Teacher 模型对能否在 ASPIRE 2A 上按上游方法完成 OPD 训练？
2. 训练后的 Student 是否在独立、同协议评测中稳定优于初始 Student？
3. 上游 top-k16 方案与 sampled-token 方案在效果和训练成本上有何差异？
4. 如何将全过程整理为别人可以重新执行、检查和扩展的工程交付？

### 2.2 严格边界

本项目优先复用上游代码和参数，不另写一套训练框架。新增代码集中在以下必要范围：

- NSCC PBS 作业包装、运行目录、断点恢复和资源监控；
- 数据冻结、评测、结果分析和验收门；
- Teacher 冻结、rollout/actor 对齐和响应 mask 等诊断；
- SwanLab 到 W&B 的最小替换；
- 对新增行为的测试与可复现配置。

本项目没有执行新的 SFT、GRPO、Teacher 训练、第二训练 seed、超参数搜索或论文全量实验矩阵。结论只适用于本报告记录的模型、数据、commit、seed 和硬件配置。

## 3. 系统与算法流程

每个训练 step 的实际流程如下：

1. 从冻结后的 DAPO-Math-17k 训练集读取 32 个 prompt。
2. Student 用 vLLM 按当前策略为每个 prompt 生成 4 个回答，最大 prompt 长度 1024，最大 response 长度 7168。
3. Actor 重新计算已采样 token 的 log-prob；冻结的 Teacher 对同一条 Student 生成轨迹计算目标分布。
4. top-k16 主线在 Student 每个状态的 top-16 token 集合上查询 Teacher，以 Student 概率加权分布差异；sampled-token 对照令 LOG_PROB_TOP_K=0，只使用实际采样 token 的逐 token 信号。
5. token_reward_direct 直接构造逐 token advantage，actor 使用 PPO 风格的 clipped policy-gradient 路径更新；这里没有训练 value critic，也不是把一个序列级 reverse-KL 标量当成 advantage。
6. 每个 step 写入训练指标、诊断和 W&B；按固定频率保存 checkpoint，训练结束后合并为 Hugging Face checkpoint。

需要特别区分两个常见指标：

- critic/true_reward/mean：规则评分器输出的原始序列奖励均值，本项目中主要反映答案正确性；它被单独保存在 true_reward_score 中，不是 OPD 的逐 token advantage。
- critic/score/mean：训练实际使用的 token_level_scores 按序列求和后的均值。OPD 路径存在 rm_scores 时，该字段记录 Teacher 分布产生的蒸馏分数；critic/rewards/mean 才是在可选 KL penalty 之后的序列 reward。当前 token_reward_direct 再把逐 token rewards 直接作为 advantage，因此 score 也不能等同于正确率。

## 4. 数据、模型与可复现配置

### 4.1 模型

模型权重下载到项目私有模型目录，未纳入 Git：

- Student：model/DeepSeek-R1-Distill-Qwen-1.5B
- Teacher：model/JustRL-DeepSeek-1.5B

D2 单卡验收确认两者 tokenizer 精确兼容、均可正常生成、Teacher 处于 eval 模式且参数不产生梯度，推理前后 Teacher 参数校验不变。

### 4.2 数据

数据准备从 DAPO-Math-17k 出发，执行格式校验、过长样本过滤、测试集重叠过滤、去重和固定 seed 的 dev 切分：

| 项目 | 数量 |
|---|---:|
| 原始样本 | 17,917 |
| 过长过滤 | 8 |
| 测试集重叠过滤 | 4 |
| 重复样本过滤 | 729 |
| dev_256 | 256 |
| 最终训练集 | 16,920 |

正式训练启用 drop_last，32 的 batch 下实际消费 528 × 32 = 16,896 个样本，剩余 24 个样本不进入本轮更新。数据文件、模型 revision、源代码 commit 和 SHA-256 均记录在 repro/manifest.json；Python 依赖分别记录在顶层 requirements.lock.txt 与 repro/requirements.lock.txt。

### 4.3 正式训练参数

主线和 sampled-token 除 distillation token 集合外保持一致：

| 参数 | 主线 top-k16 | sampled-token |
|---|---:|---:|
| adv_estimator | token_reward_direct | token_reward_direct |
| LOG_PROB_TOP_K | 16 | 0 |
| top_k_strategy | only_stu | only_stu |
| reward_weight_mode | student_p | student_p |
| train_batch_size | 32 | 32 |
| rollout n | 4 | 4 |
| max prompt / response | 1024 / 7168 | 1024 / 7168 |
| actor learning rate | 1e-6 | 1e-6 |
| PPO mini / micro batch | 32 / 1 per GPU | 32 / 1 per GPU |
| rollout temperature / top-p | 1.0 / 1.0 | 1.0 / 1.0 |
| total epochs / steps | 1 / 528 | 1 / 528 |
| rollout backend | vLLM | vLLM |
| enforce_eager | false | false |
| GPUs | 4 × A100 40GB | 4 × A100 40GB |
| seed | 2026 | 2026 |

解析后的完整配置位于 configs/nscc/resolved_main_7168.yaml 与 configs/nscc/resolved_sampled_token_7168.yaml。它们是最终生效参数的证据，比只查看启动脚本更可靠。

## 5. 实现内容

### 5.1 上游核心代码的最小改动

核心训练路径没有被重写，只加入了可观测性和运行安全措施：

- verl/verl/trainer/ppo/ray_trainer.py
  - rollout JSONL 支持 Tensor、NumPy 等对象的稳定序列化；
  - 可选输出 response-mask 一致性诊断；
  - 将 SwanLab 图像记录改为 W&B，并复用 verl 的统一 logger。
- verl/verl/workers/fsdp_workers.py
  - Teacher 初始化后强制 eval；
  - 在诊断模式下冻结 Teacher，并在训练中复核参数身份、版本和 checksum；
  - 记录 actor 到 rollout engine 的权重同步标记。
- verl/verl/utils/debug/metrics.py
  - 在 response mask 内新增 rollout 与 actor 原始 log-prob 绝对差的 max、mean、std；
  - 这些指标用于观察一致性，不设置上游不存在的人为通过阈值。
- verl/verl/utils/opd_d4_diagnostics.py
  - 提供 rollout JSON 序列化和 response-mask 结构验收。
- verl/verl/utils/opd_teacher_guard.py
  - 提供 Teacher 冻结、无梯度和训练前后不变性检查。
- verl/verl/utils/tracking.py
  - 延迟导入 wandb 并创建原生 W&B Image，避免模块加载时产生不必要依赖副作用。

### 5.2 W&B 迁移

项目原始实验脚本默认使用 SwanLab。为与现有实验管理体系对齐，采用最小替换：

- on_policy_distillation.sh、grpo.sh、verl_example/opd.sh 和 verl/examples/ppo_trainer/on_policy_distillation.sh 改为 console + wandb；
- sft_trainer.yaml 与 LlamaFactory 示例配置改为 W&B；
- WANDB_DIR、WANDB_MODE、WANDB_RUN_ID 和 WANDB_RESUME 由 PBS 脚本统一管理；
- primary 与 resume 段复用同一个 run id，避免断点恢复产生多个孤立 run。

正式训练的 W&B 页面：

- top-k16：https://wandb.ai/2523767186-queen-mary-university-of-london/OPD-NSCC/runs/opd-d7-9-seed2026-21110343
- sampled-token：https://wandb.ai/2523767186-queen-mary-university-of-london/OPD-NSCC/runs/opd-sampled-token-seed2026-21908511

### 5.3 NSCC 作业与阶段门

scripts/nscc 下的脚本形成可复用的阶段化流水线：

| 阶段 | 主要文件 | 作用 |
|---|---|---|
| D2–3 | d2_single_gpu_inference.py/.pbs | 单卡模型加载、生成、tokenizer 与 Teacher 冻结验收 |
| D3–4 | d4_prepare_data.py/.pbs、d4_baseline_eval.py/.pbs | 数据冻结与初始 Student/Teacher 基线 |
| D4 G1 | d4_smoke_config.py、d4_smoke.pbs、d4_analyze_smoke.py | 四卡冒烟、checkpoint、rollout 和诊断验收 |
| G1 收尾 | d4_checkpoint_verify.py、d4_export_inference.py、d4_publish_readiness.py | checkpoint 结构检查、导出推理与原子发布 readiness |
| D5–6 G2 | d5_pilot_config.py/.pbs、d6_7168_config.py/.pbs 及分析脚本 | 4096 长度 pilot、7168 长度验证、30–50 步连续训练门 |
| D7–9 | d7_9_formal_config.py、d7_9_formal_run.pbs、d7_9_analyze_formal.py | 528 步正式训练、断点恢复、资源监控、最终验收 |
| 独立评测 | d12_final_eval.py/.pbs、d12_analyze_final.py、d12_closeout.pbs | 合并 checkpoint、冻结协议评测、配对 bootstrap 和结论 |
| sampled-token | d15_sampled_token_config.py、run/closeout/eval/verify PBS、d15_analyze_sampled.py | LOG_PROB_TOP_K=0 对照训练、同协议评测和对比 |

正式作业还实现了：

- 只接受 PBS 分配的 4 张 GPU，并在进程内映射为 0,1,2,3；
- 运行前执行 pip check、模型/数据/readiness/W&B 认证预检；
- 每分钟采集显存、利用率、温度和主进程 RSS；
- primary/resume 路径校验、并发 segment lock、最多三个 24 小时段；
- checkpoint 结构验收和固定步数恢复；
- 结果先写临时目录，再通过 hash 和结构检查后发布，避免半成品覆盖可信结果。

### 5.4 独立评测与统计

最终评测与训练内 validation 分离，所有模型使用相同 prompt、输入哈希、采样参数和评分实现：

| 数据集 | 问题数 | 每题采样数 | max tokens | temperature | top-p |
|---|---:|---:|---:|---:|---:|
| MATH-500 | 500 | 4 | 7168 | 0.7 | 0.95 |
| AIME24 | 30 | 16 | 7168 | 0.7 | 0.95 |
| dev_256（仅诊断） | 256 | 1 | 2048 | 0.7 | 0.95 |

比较采用以 problem 为重采样单位的 10,000 次配对 bootstrap，seed=2026。预先声明的可靠增益标准是两个必测任务的差值 95% CI 下界都大于 0。dev_256 不参与 checkpoint 选择，也不参与最终收益判断。

## 6. 推进过程与阶段结果

| 阶段 | Job ID | 结果 | GPU·h |
|---|---|---|---:|
| D2–3 单卡推理 | 20334672.pbs101 | 模型、tokenizer、生成和 Teacher 冻结通过 | 0.148 |
| Student/Teacher baseline | 20395781.pbs101 | 完成冻结协议基线 | 1.290 |
| D4 G1 主训练/恢复 | 20420432.pbs101 | 训练与 checkpoint 成功；旧后处理导致 PBS 非零退出 | 0.482 |
| G1 导出收尾 | 20658477.pbs101 | checkpoint 导出与 readiness 通过 | 0.123 |
| D5 4096 pilot | 21027386.pbs101 | 连续训练与诊断通过 | 2.399 |
| D6 7168 验证 | 21065750.pbs101 | 正式长度下显存/吞吐与恢复门通过 | 1.633 |
| D7–9 top-k16 正式训练 | 21110343.pbs101 | 528/528、1 epoch、最终 checkpoint 完成 | 60.296 |
| D12 主线独立评测 | 21857038.pbs101 | 技术验收与可靠增益均通过 | 0.621 |
| sampled-token 正式训练 | 21908511.pbs101 | 528/528、1 epoch 完成 | 56.392 |
| sampled-token 独立评测 | 22729409.pbs101 | 对初始模型可靠增益；对主线差异不显著 | 0.782 |
| 最终回归 | 22768431.pbs101 | 58 passed，py_compile、bash -n、diff check 通过 | CPU |

主线 accepted path（不含失败诊断、取消任务、CPU 作业和 sampled-token 可选实验）共 66.992 GPU·h，按 64 SU / A100 GPU·h 估算为 4,287.48 SU。

## 7. 最终对比结果

### 7.1 主线 top-k16

| 数据集 | 初始 Student | Teacher | top-k16 OPD | 相对初始差值 | 配对 95% CI |
|---|---:|---:|---:|---:|---:|
| MATH-500 | 0.7455 | 0.8250 | 0.8160 | +0.0705 | [0.0465, 0.0945] |
| AIME24 | 0.1833 | 0.3146 | 0.3271 | +0.1438 | [0.0729, 0.2208] |

主线相对初始 Student 的两个置信区间下界均大于 0，因此 verdict 为 REPRODUCED_RELIABLE_GAIN。相对 Teacher，MATH-500 差值为 -0.0090、AIME24 为 +0.0125，两个置信区间均跨 0，不能声称超过或落后 Teacher。

### 7.2 sampled-token 对照

| 数据集 | 初始 Student | sampled-token | 相对初始差值 | 配对 95% CI |
|---|---:|---:|---:|---:|
| MATH-500 | 0.7455 | 0.8160 | +0.0705 | [0.0490, 0.0920] |
| AIME24 | 0.1833 | 0.3125 | +0.1292 | [0.0688, 0.1979] |

sampled-token 对初始 Student 同样满足可靠增益标准。

### 7.3 sampled-token 与 top-k16

| 数据集 | top-k16 | sampled-token | sampled − top-k16 | 配对 95% CI | 结论 |
|---|---:|---:|---:|---:|---|
| MATH-500 | 0.8160 | 0.8160 | 0.0000 | [-0.0135, 0.0140] | 不显著 |
| AIME24 | 0.3271 | 0.3125 | -0.0146 | [-0.0542, 0.0229] | 不显著 |

这组结果说明：对于当前模型对和单 seed，dense top-k16 信号没有转化为可统计区分的最终准确率优势，但也没有证据证明 sampled-token 更好。实验的统计功效不足以判断 1–2 个百分点级别的差异。

### 7.4 长度、格式与资源

| 指标 | top-k16 | sampled-token |
|---|---:|---:|
| 训练 walltime | 15:04:26 | 14:05:53 |
| 训练 GPU·h | 60.296 | 56.392 |
| 估算 SU | 3,858.92 | 3,609.10 |
| 生成 response tokens | 369,239,639 | 369,377,684 |
| 训练命中长度上限比例 | 0.3108 | 0.3060 |
| 有效答案存在比例 | 0.7064 | 0.7112 |

评测截断率：

| 数据集 | 初始 Student | Teacher | top-k16 | sampled-token |
|---|---:|---:|---:|---:|
| MATH-500 | 0.2035 | 0.1210 | 0.1250 | 0.1260 |
| AIME24 | 0.7958 | 0.6542 | 0.6229 | 0.6271 |

AIME24 的截断率仍超过 62%。由于所有对比模型使用同一协议，配对比较仍然有效，但绝对 pass@1 受 7168 token 上限影响，不能视为模型无长度限制时的能力上限。

## 8. 主要问题、根因与解决方法

### 8.1 环境安装耗时与 wheel 问题

问题：初次环境构建时间远长于轻量项目，主要耗时来自 PyTorch/vLLM/FlashAttention 等与 CUDA、Python ABI 强相关的大包；部分依赖没有与当前模块组合完全匹配的现成 wheel，会触发编译或大体积解包。

解决：锁定 Python 3.12、CUDA 与包版本；优先复用兼容 wheel 和本地缓存；将最终依赖写入 lock 文件，并用 PBS 节点做 import/pip check 验收。后续不应在登录节点反复新建环境。

### 8.2 G1 失败但训练实际完成

问题：最初 G1 作业显示失败，但训练、checkpoint 和日志已经生成。

根因：旧后处理脚本把自定义 rollout/actor 差异阈值当作硬验收条件，并且导出/分析链路对日志格式过于脆弱。这个阈值不是上游仓库的验收标准。

解决：删除人为数值阈值，只保留上游式方向性诊断与结构检查；复用上游 checkpoint/export 路径；将训练完成与后处理完成分离；新增独立 closeout 作业。最终 G1 readiness 通过。

### 8.3 7168 长序列下的显存与执行模式

问题：直接套用上游 8 × A800 80GB 配置到 4 × A100 40GB 会产生显存和吞吐风险。

解决：先做 4096 pilot，再做 7168 短验收；保留上游动态 batch 和 token budget 设计，将 per-GPU micro-batch 设为 1、vLLM 显存利用率设为 0.6、每卡 max batched tokens 设为 8192。enforce_eager 最终保持 false，使用 CUDA graph 路径；只有诊断时才比较 eager 行为。

### 8.4 SwanLab 依赖与实验追踪

问题：项目原始脚本依赖 SwanLab，而目标工作流要求 W&B；只改 launcher 会导致训练内部图像仍直接 import SwanLab。

解决：替换脚本 logger，并将 trainer 内的 SwanLab Image/log 调用改为 W&B 原生对象和统一 logger；W&B import 延迟到实际创建图像时。正式训练和断点恢复共用稳定 run id。

### 8.5 rollout JSON 和日志解析

问题：rollout 元数据包含 Tensor/NumPy 对象，直接 json.dumps 会失败；进度条刷新、Unicode U+2028 和长行也会破坏基于行的分析。

解决：加入统一 json_default；分析器不再依赖终端渲染行，而是解析结构化 JSON marker，并兼容控制字符、被拆分的进度输出和多 segment 日志。

### 8.6 Teacher 是否真的冻结

问题：仅设置 requires_grad=False 不能完整证明 Teacher 没有被意外切回 train 或发生参数改变。

解决：初始化时强制 eval，冻结参数并记录参数身份、版本和 checksum；训练中周期性复核；测试覆盖 eval、无梯度和 checksum 不变。

### 8.7 NSCC 记账异常

问题：部分四卡 PBS 任务的 resources_used.ngpus 被记为 0，与实际分配和 nvidia-smi 证据不一致。

解决：资源账单使用“请求且实际观察到的 GPU 数 × PBS walltime”计算 GPU·h，并明确记载这是估算口径；SU 按 64 SU / A100 GPU·h 换算，不伪装成账户最终账单。

### 8.8 Scratch 环境文件被部分清理

问题：项目后期发现 Conda Python 包目录中 1,930 个受管理文件有 1,439 个缺失，导致看似随机的包缺失。

根因：Scratch 清理或不完整解包破坏了现有 Python 安装，而不是项目 requirements 变化。

解决：在 PBS 作业中从本地缓存精确重装 python-3.12.0-hab00c5b_0_cpython，修复后缺失文件为 0，并重新验证 torch 2.8.0+cu128 import。临时修复缓存随后删除。

### 8.9 正式训练完成后 PBS 仍非零退出

问题：主线训练已完成 528 步并保存 checkpoint，但旧 analyzer 的后处理错误使 PBS Exit_status=1。

解决：以 checkpoint、训练步数、数据消费和结构化 marker 判断训练完成；新增幂等 closeout 重新分析已有产物，不重跑昂贵训练；结果发布前做 hash 和结构校验。

### 8.10 指标在约 200 步附近震荡

现象：critic/true_reward/mean 在约 200 步附近先上升、后明显回落，再恢复。

解释：该指标是当步采样 batch 的序列正确率均值，batch 仅含 32 个 prompt 且每题 4 个 rollout，受题目难度、采样随机性和策略更新共同影响；数据按单 epoch 顺序推进时，不同步段的题目难度也会改变均值。同期训练没有出现 loss 非有限、Teacher 漂移、rollout 同步失败或 checkpoint 损坏，因此该波动不能单独判定为训练崩溃。最终应由冻结协议的独立评测判断，而不是从一段在线 reward 曲线反推结论。

## 9. 已实施改进与后续改进建议

### 9.1 已实施

- 用阶段门替代一次性长任务：D2 → G1 → G2 → 正式训练 → 独立评测。
- 用 resolved YAML、manifest、hash 和 lock 文件固化真实运行状态。
- 用 Teacher guard、response mask、rollout sync 和 log-prob 指标提升可观测性。
- 用 checkpoint-aware primary/resume、segment lock 和幂等 closeout 降低长任务失败成本。
- 用独立评测、问题级配对 bootstrap 和预声明判据替代只看训练 reward。
- 用同 seed、同数据、同协议、仅 LOG_PROB_TOP_K 不同的方式构造公平对照。
- 将 W&B 完整接入训练内部，而不只修改 shell 层。

### 9.2 建议但尚未实施

1. 补做至少 3 个训练 seed。当前 top-k16 与 sampled-token 的差异都落在置信区间内，多 seed 是判断稳定差异的首要改进。
2. 针对 AIME24 做长度敏感性评测，例如 8192、12288 或 16384 token；必须对所有比较模型使用同一新协议，且与本报告的 7168 结果分表呈现。
3. 报告 pass@k 或按 problem 聚合的成功概率，同时保留逐 rollout pass@1，减少小数据集上单一指标的偶然性。
4. 分桶分析难度、响应长度、是否 boxed 与正确率，定位约 200 步附近 reward 波动是否来自数据构成。
5. 若要研究 top-k 的收益，应增加 k={0,4,8,16,32} 的受控实验，并同时记录 walltime、通信、Teacher forward 成本和显存峰值。
6. 将硬编码的最终 checkpoint/job root 改为显式 CLI 参数或 manifest 引用，使独立评测脚本可直接复用于新 run。
7. 在 Scratch 上定期执行只读环境完整性校验，并保留可恢复的包缓存或打包环境，降低被清理后的恢复时间。

## 10. 测试与验收

最终回归覆盖以下内容：

- 51 个 pytest：D2 推理辅助逻辑、D4 pipeline/acceptance、Teacher guard、rollout JSON、W&B helper 和 debug metrics；
- 所有新增 Python 脚本通过 py_compile；
- 所有 PBS/shell 脚本通过 bash -n；
- git diff --check 无空白错误；
- 正式与对照训练 acceptance、独立评测 acceptance、输入 hash 和 checkpoint 结构均通过。

最新完整回归 Job 22768431.pbs101 的结果为 58 passed、1 个来自 pynvml 弃用提示的 warning；py_compile、bash -n 和全仓库 git diff --check 均通过。该作业在 README、报告、忽略规则及全部实现代码就位后执行，Exit_status=0。

## 11. 复现入口

登录节点只负责编辑、提交和 qsub；GPU 计算必须通过 PBS：

~~~bash
ssh xiongjie@aspire2a.nus.edu.sg
cd /home/users/nus/xiongjie/scratch/OPD

# 查看冻结配置与可复现清单
less repro/manifest.json
less configs/nscc/resolved_main_7168.yaml

# 提交主线正式训练
qsub scripts/nscc/d7_9_formal_run.pbs

# 提交 sampled-token 对照
qsub scripts/nscc/d15_sampled_token_run.pbs

# 提交回归验证
qsub scripts/nscc/d15_sampled_token_verify.pbs

# 查看队列
qstat -u xiongjie
~~~

独立评测脚本保留了本次冻结 run 的路径，重跑新实验时应先复制并更新 checkpoint、baseline 和输出根目录，不能覆盖已有结果。原始模型、数据、checkpoint、逐题输出和日志都属于大体积运行产物，不提交 Git。

## 12. 交付物与目录说明

- README.md：仓库入口、复现摘要、快速命令和结果表；
- NSCC_OPD_Reproduction_Timeline.md：阶段计划、边界、Job 执行记录和验收定义；
- docs/NSCC_OPD_PROJECT_REPORT.md：本完整报告；
- configs/ 与 configs/nscc/：各阶段解析后的真实配置；
- scripts/nscc/：PBS 作业、数据、训练、分析、评测和收尾脚本；
- tests/nscc/、verl/tests/utils/：回归测试；
- repro/manifest.json：commit、模型、数据、hash 和统计；
- requirements.lock.txt、repro/requirements.lock.txt：环境依赖快照；
- logs/、eval/、model/、datasets/nscc_d4/、reports/：本机运行产物，按交付约定不提交远程仓库。

## 13. 总结

该项目不仅完成了一次“能跑”的 OPD 训练，还建立了从上游代码基线、PBS 资源约束、数据冻结、短程门控、长任务恢复、在线追踪到独立统计评测的完整证据链。对于指定 Student/Teacher，OPD 的收益得到了可靠复现；top-k16 与 sampled-token 都有效，但单 seed 尚不足以区分两者最终准确率。最重要的后续工作不是继续凭在线曲线调参，而是补充多 seed 和更长输出上限的受控评测，以确定差异是否稳定、AIME 截断是否掩盖了真实能力，以及 top-k dense signal 的额外计算是否值得。
