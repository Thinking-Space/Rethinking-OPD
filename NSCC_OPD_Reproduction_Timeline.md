REPRODUCTION ROADMAP   /   v1.0

# NSCC OPD 复现项目 Timeline 与执行计划

NSCC ASPIRE 2A  ·  4×A100 40GB  ·  1.5B student

> **主线目标** 用一组固定的 student / teacher，完成“基线 → OPD 训练 → 独立评测 → 结果分析”的闭环；先验证实现和收益，不追求一次复完论文全部实验。

**计划窗口：2026 年 9 月 14–27 日；可选缓冲：9 月 28 日–10 月 4 日。** 日期按下一工作日启动拟定，不代表已预约 GPU。GPU 耗时、排队时间和当前 SU 余额尚未实测。

| 日期 / 阶段 | 主要任务 | 阶段产物与放行条件 |
| --- | --- | --- |
| 9/14 · D1<br>立项与资源核对 | 冻结代码、模型对、数据和评测口径；查询账户、队列与存储。 | manifest + 预算上限；确认不会改坏 Search-R1 环境。 |
| 9/15–16 · D2–3<br>环境与单卡验证 | 建立独立环境；验证 student / teacher 推理、tokenizer、数学评分器。 | 两模型均可生成并评分；确认实际导入本项目 verl。 |
| 9/17 · D4<br>基线与四卡冒烟 | 记录初始 student / teacher 基线；完成 5–10 步训练及短续训测试。 | G1：训练更新、teacher 信号、保存与恢复全部可验证。 |
| 9/18–19 · D5–6<br>Pilot 与冻结参数 | 连续 30–50 步；测峰值显存、每步时间、生成长度和截断率。 | G2：闭环稳定；重估费用后决定正式步数与序列长度。 |
| 9/20–22 · D7–9<br>正式 OPD | 按 ≤24 小时的 PBS 作业分段训练；目标 1 个数据遍历周期。 | checkpoint + 完整日志；受实测预算限制，可降为固定步数实验。 |
| 9/23–24 · D10–11<br>轻量检查窗口 | 仅检查已稳定作业、保存结果、做曲线初读；不升级环境或换模型。 | 保留给 Hackathon 收尾；出现问题先停，不临时开大规模新实验。 |
| 9/25–27 · D12–14<br>终评与归档 | 按冻结协议评测最终模型；整理增益、失败案例、算力账单和 README。 | G3：别人可复跑；结论区分技术跑通、收益证据与实验限制。 |
| 9/28–10/4 · 可选 | 补第二个 seed，或做一次 sampled-token 对照；二选一。 | 不计入两周必交项；排队延误优先顺延，而非压缩验证。 |

**投入安排：** 人工工作先按约 20–30 小时分摊到两周，GPU 运行与排队另计。9/23–24 保持低强度；9/28 仅按你此前拟定的入职窗口预留，不作为已确认入职日期。


---

01   /   EXPERIMENT CONTRACT

## 先锁定实验，再讨论扩展

### 1.1 主线模型与范围

**Student：** deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
**Teacher：** hbx/JustRL-DeepSeek-1.5B（冻结，仅计算监督信号）。

这是本次计划的建议默认组合：OPD 启动脚本使用这两个模型，JustRL 项目也提供该 teacher 的权重入口。[2][7] 先不换成 Qwen3-1.7B / 4B，避免同时改变模型家族、模板和训练实现；不是说更大 teacher 不可用。

**训练：** DAPO-Math-17k，另划固定 dev 子集并从训练中排除。**必做终评：** MATH-500 + AIME24；AIME25 视预算追加。使用同一套独立生成和评分流程，不混用不同评测口径。[1]

### 1.2 原版主线与新版备选不能混配

| 实现路线 | 四卡分配 | 本期定位 |
| --- | --- | --- |
| 原版：仓库根目录<br>on_policy_distillation.sh<br>+ 仓库内定制 verl | trainer.n_gpus_per_node=4<br>reward_model.enable_resource_pool=False<br>actor / rollout / teacher 共用四卡。[2][3] | **默认主线。** 保留作者的 token_reward_direct、top-k 选择与奖励加权实现。 |
| 新版：verl_example/opd.sh<br>+ 独立安装的兼容 verl | student pool 2 卡 + teacher pool 2 卡；<br>不能保留默认 4+4。[4][5] | 仅作环境路线备选。切换需另建实验，重新验证 loss 与梯度路径，不能宣称与原版自动等价。 |

### 1.3 原版路线的候选配置（尚未在 NSCC 验证）

| 参数 / 控制项 | 冒烟 | Pilot | 正式实验 |
| --- | --- | --- | --- |
| prompt / response 上限 | 1024 / 2048 | 1024 / 4096 | 1024 / 7168；不通过则固定 4096 |
| train batch / 每题 rollout 数 n | 8 / 1 | 16 / 4 | 32 / 4 |
| actor 动态 token budget / GPU | 3072 | 5120 | 8192；短上下文版本为 5120 |
| actor micro-batch / teacher micro-batch | 1 / 1 | 1 / 1 | 从 1 / 1 起，不直接用 teacher 默认 24 |

固定算法：token_reward_direct；top-k=16；only_stu；student_p；lr=1e-6。开启 gradient checkpointing，优先验证 actor 参数 / optimizer offload；记录真实 dtype 和显存，不把修改精度视为零影响。

> **配置检查重点** 原版有硬编码赋值，外部 export 不一定生效；其动态 token budget 还设了至少 32768 的下限。必须在新建的 NSCC launcher 中显式改写，并保存最终解析配置。micro-batch=1 不等于动态 batch 只装一条序列。[2]

### 1.4 对齐诊断与放行口径（按冻结上游实现）

- **rollout ↔ actor 实现一致性：** 保持原版
  `actor_rollout_ref.rollout.calculate_log_probs=True`。在 response mask 上记录上游
  `calculate_debug_metrics` 原生输出的概率差 max / mean / std 与 Pearson 相关系数。
  上游将这些量定义为 debug metrics，没有给出数值阈值或 pass / fail 断言；因此不再把
  额外的 raw log-prob max 阈值作为 G1 放行条件。[2][9]
- **student ↔ teacher 学习对齐：** 训练中记录 top-k overlap ratio、共享 top-k token
  advantage 与 student / teacher entropy gap；判断其跨 step / 窗口趋势，而不是用单步
  固定阈值。期望 overlap 上升，overlap-token advantage 与 entropy gap 向 0 收敛。[10]

正式 run 开始后，不在同一曲线中途更换长度、模型或 loss。需要改变时新建 run；4096-token 版本必须标记为短上下文复现。


---

02   /   DAYS 1–4

## 环境、基线与正确性检查

### D1｜资源确认与可复现快照

查询当前可用 SU、已占用额度、项目权限和存储；不沿用 8 月的余额。建立独立 OPD 目录和环境，不在 Search-R1 环境中升级依赖。代码、配置和小型记录放可长期保存的位置，大模型与中间产物按存储策略管理。

建议冻结本次核对的仓库 commit：**ac26e38d6f1572eb027597b48a9f4e01f6915ef8**。新增分支保存四卡补丁；记录模型 revision、数据哈希和启动命令。所有后续“与原版的差异”都写入 manifest。[8]

### D2–3｜独立环境 + 两模型单卡推理

按仓库原版说明从 Python 3.12 独立环境开始，先审阅安装脚本，再锁定 torch / CUDA / vLLM / transformers / flash-attn / Ray / verl 组合；不能只因 NSCC 已有 PyTorch 模块就假定兼容。[1] 编译、批量处理和 GPU 检查放到获批的计算作业中。

```bash
python -c "import verl; print(verl.__file__)"
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -m pip check
python -m pip freeze > requirements.lock.txt
# GPU 作业内再记录：nvidia-smi、module list、完整启动日志
```

抽 16 条 prompt 验证两模型生成、EOS 和答案抽取。token 级蒸馏要核对 vocab、token-id 映射、特殊 token 和 chat template；不能仅凭“同一家族”跳过检查。加载 teacher 后保持 eval / no-grad，确认模型路径不是 student 的重复副本。

### D3–4｜冻结数据与评测协议

固定 seed=2026，从训练来源划出 dev_256 并排除出训练；记录过滤长 prompt、去重及已发现的测试重叠。保存最终训练条数 N，后面按 N 计算 epoch，而不是直接把“17k”当实际样本数。

初始 student 与 teacher 都先跑同一 dev 小评测；正式基线使用与最终模型完全一致的 MATH-500 / AIME24 协议，原始输出逐题保存。dev 用于调参与选 checkpoint，最终测试集不用于反复挑选最优配置。

### D4｜5–10 步四卡冒烟：G1 验收

| 检查 | 必须看到的证据 |
| --- | --- |
| rollout / actor 路径可比 | student rollout 的 token / prefix / tensor shape 对齐；prompt、padding 不计入 response loss；上游概率差 max / mean / std 与 Pearson 指标使用 response mask 且为有限值。上述数值保存为调试证据，但上游没有规定硬阈值；额外 raw log-prob 差只能作为补充诊断。[9] |
| student / teacher 信号可观测 | top-k overlap ratio、共享 token advantage、student / teacher entropy 及其 gap 能按 step 记录；G1 只验证采集链路与数值有限，不要求 5–10 步内达到论文中的长期趋势。[10] |
| 更新正确 | loss 与梯度为有限值；student 参数发生更新，teacher 不更新；下一轮 rollout 使用更新后的 student。 |
| 可保存、可继续、可评测 | 保存一次训练 checkpoint 并退出；从 checkpoint 恢复 step、optimizer / scheduler 等状态，再跑至少 2–5 步；导出模型能独立推理。 |

> **G1 放行规则** 四项全部通过才进入 pilot。概率差和 Pearson 按上游用途用于发现
> 异常，但单个非零差值不自动判失败；硬失败包括 mask / shape / token 对齐错误、NaN /
> Inf、监督信号缺失、teacher 更新、student 未更新、rollout 未同步，或保存 / 完整恢复 /
> 独立推理失败。失败时只修最短闭环，不启动整晚训练，也不同时更换模型、框架和数据。

`enforce_eager` 保持上游默认关闭；只有定位 CUDA Graph 相关差异时才作为单变量诊断，
不为满足本项目自定义阈值而并入正式配置。

本阶段产物：repro/manifest.json、requirements.lock.txt、configs/resolved_smoke.yaml、eval/baseline/、logs/smoke/、reports/readiness.md。


---

03   /   DAYS 5–11

## Pilot 校准，再按预算正式训练

### D5–6｜连续 30–50 步：G2 验收

先以 4096 response 长度稳定运行，再单独验证 7168 配置；舍去启动 / 编译的前几步后，统计整体及 rollout、teacher scoring、actor update 的耗时。**Pilot 不要求几十步就显著涨分，也不要求 loss 单调下降。**

| 要测什么 | 怎样决定下一步 |
| --- | --- |
| step 时间：中位数与 P90 | 按最终候选长度、batch、n 实测；用 P90 做保守时长预算。短长度 pilot 的吞吐不能直接当长长度正式吞吐。 |
| 显存、主机 RAM、并发 | 分阶段记峰值；覆盖较长样本后无 OOM，才增加并发。top-k=16 不代表所有中间 logits 张量都只占 16 个 token 的空间。 |
| 长度 / 截断 / 有效答案率 | 记录 response 均值、P95、hit-length-limit 比例。截断明显时先分析长度分布，不能仅靠 loss 下降判定训练健康。 |
| teacher 信号、学习对齐与短续训 | 检查信号是否恒零、NaN、异常饱和；观察 top-k overlap ratio、overlap-token advantage 和 entropy gap 的窗口趋势，不设上游未声明的单点阈值；从已有 checkpoint 续跑后 step 与数据进度连续，下一段未偷偷从初始权重开始。[10] |

### 把“要跑多久”改成可计算的问题

```text
训练 global steps ≈ ceil(N_train / B_prompt) × epochs
每个 global step 的 rollout 条数 = B_prompt × n
训练小时 ≈ global_steps × P90_step_seconds / 3600
总耗时还要加：初始化、保存 / 恢复、独立评测
注意：global step 不等于内部 optimizer update 数；drop_last 以实际 loader 为准。
```

**仅为算例，不是实测：** 假设过滤后 N=16,000、B=32、1 epoch，则约 500 个 global steps；P90 为 240 秒时纯训练约 33.3 小时，为 360 秒时约 50 小时。n=4 意味着每步 128 条生成，不能把 batch=32 当成只生成 32 条。

### D7–9｜正式 run：只执行冻结配置

默认目标为过滤后训练集的 1 epoch；若 G2 估算超出本项目预算，改为预先声明的固定步数实验，并报告实际 prompt / rollout / token 数。**只完成 100–200 步时，不写成 full-data 完整训练。**

正式训练按 ≤24 小时 PBS 作业分段。建议每 10–20 步保存一次，并根据实测调整到大约每小时内至少一次；退出前预留保存余量。下一作业显式恢复同一 run 的训练状态，不只加载模型权重。

### D10–11｜留给 Hackathon 的轻量窗口

9/23–24 只进行简短日志检查、checkpoint 归档与既有作业续跑。未通过 G2 的实验不进入无人照看的长作业；出现 NaN、重复 OOM 或吞吐突降，暂停后续提交。禁止在此窗口顺手升级 verl 或增加一条实验线。

> **阶段放行** G2 = 数值与闭环正确 + 长度 / 显存可控 + 续训通过 + 全程费用可承受。成功证据来自最终评测，不能用 pilot 的 loss 或单批 reward 代替。

#### 2026-09-15 D5–6 执行记录

- 4096 Pilot Job `21027386.pbs101` 完成 primary step 1–30，并由独立进程从 `global_step_30` 完整恢复到 step 31–32；PBS `Exit_status=0`、实际 walltime `00:35:59`。step 30/32 的 model、optimizer、extra-state 四卡 shard 与 dataloader 状态均通过结构检查。
- 4096 稳定段（排除 1、2、3、31）整体 step 时间中位数/P90 为 `45.557/46.160 s`；response 均值 `3931.1`，hit-length-limit 均值 `0.8726`，有效答案存在率 `0.1538`。峰值监控 GPU 显存 `31075 MiB`，最小余量 `9885 MiB`；峰值主机 RSS `144.647 GiB`。
- 7168 独立验证 Job `21065750.pbs101` 从初始 Student 运行 step 1–8，保持正式候选 `prompt=1024`、`response=7168`、`B_prompt=32`、`n=4`、actor token budget/GPU `8192`；PBS `Exit_status=0`、实际 walltime `00:24:30`，`global_step_8` 完整 checkpoint 通过结构检查。验收代码的 CPU 门禁 Job `21063530.pbs101` 为 `31 passed`。
- 7168 稳定段（排除 1–3）整体、rollout、teacher scoring、actor update 的中位数/P90 分别为 `110.255/111.097`、`64.643/65.341`、`8.864/8.939`、`26.145/26.475 s`。response 均值 `6301.3`，P95 每步均为 `7168`，hit-length-limit 均值 `0.6836`，有效答案存在率 `0.3535`，aborted ratio 为 `0`。相较 4096，截断和无有效答案现象明显缓解但没有消失；正式结果必须继续报告这一限制。
- 7168 峰值监控 GPU 显存 `37841 MiB`，最小余量 `3119 MiB`；四卡被同时观测为活跃，峰值主机 RSS `146.726 GiB`。8 步 Teacher 信号非零且有限，冻结与 rollout 权重同步检查逐步通过；三步窗口对齐指标仅作趋势记录，没有添加上游未声明的数值阈值。
- 7168 实际 dataloader 为 `528` steps/epoch。按稳定段 P90 估算，1 epoch 约 `65.18 GPU·h / 4171.34 SU`，低于本阶段用于正式训练决策的 `18432 SU` 预算，因此冻结建议为 `7168`、`528` steps（1 epoch）。4096 实际用量与 7168 三小时预留的组合上界为 `14.3989 GPU·h`，低于 32 GPU·h Pilot 上限；两次成功作业的实际合计约 `4.032 GPU·h / 258.1 SU`。
- 7168 训练结束并保存 checkpoint 后，Ray 回收 DataLoader worker 时打印了 `Killed` 与 W&B transport 析构警告；主训练已完成 8/8 steps，PBS、checkpoint 检查、G2 分析和全部 artifact SHA256 均通过。W&B 保持 offline，本记录不宣称已同步到云端。
- **最终门禁状态：G2 PASS。** 正式候选冻结为 7168 配置和 1 epoch/528 steps；这只证明技术闭环、资源与预算可行，不代表 Pilot 已有分数收益。该条记录形成时 D7 尚未开始；现状见下方 D7–9 执行记录。

#### 2026-09-16 D7–9 执行记录

- 正式训练 Job `21110343.pbs101` 以冻结的 7168/B32/n4、seed 2026、`token_reward_direct`、top-k16、`only_stu`、`student_p`、`enforce_eager=False` 配置运行；单个 primary segment 连续完成 step `1–528`，没有发生恢复，严格覆盖过滤后训练集的 1 epoch。训练 walltime 为 `15:04:26`，最终 `global_step_528` 含 model、optimizer、extra-state 各 4 shard 和 dataloader `data.pt`，结构检查 `passed=true`。
- PBS 对训练 Job 记录 `Exit_status=1`，但失败点发生在 528/528 与最终 checkpoint 保存之后的自动验收阶段，而不是训练阶段。根因依次为：在线 W&B 日志中的 `wandb offline` 提示被旧正则误判、上游 rollout 同步标记使用 `sync_count` 而非 `step`、以及 Python `splitlines()` 把 JSON 字符串内合法的 Unicode U+2028 当作记录边界。修复均只作用于 D7–9 验收/JSONL 读取，不改变训练算法、配置或既有 12 GB rollout。
- 三轮回归均按 RED→GREEN 验证：W&B 正则 `21612280.pbs101` → `21613974.pbs101`，同步标记 schema `21619609.pbs101` → `21621545.pbs101`，物理 JSONL 行读取 `21671302.pbs101` → `21673477.pbs101`；最终门禁为 `35 passed, 1 warning`，warning 仅为环境既存的 `pynvml` 弃用提示。
- CPU-only 收尾 Job `21816806.pbs101` 实际 walltime `00:27:47`、`Exit_status=0`，日志含 `WANDB_FINAL_OK` 与 `D7_9_FORMAL_CLOSEOUT_OK`。`main_acceptance.json` 为 `passed=true`、`D7–9 status: PASS_TRAINING_COMPLETE`，`G3 status: NOT_STARTED`；发布配置与 readiness 和 run-root 原件逐字节一致，`artifact_sha256.txt` 的 11 项全部复核为 `OK`。
- 去除每段前 3 个 warmup steps 后，整体/rollout/Teacher scoring/actor update 的中位数-P90 分别为 `101.196–108.167`、`60.845–63.742`、`7.713–8.491`、`22.790–25.364 s`。共保存 `67,584` 条 rollout，覆盖 `16,896` 个唯一 prompt；response 均值 `5,463.4` tokens，逐步 P95 的均值/最大值为 `7,166.3/7,168`，hit-length-limit 比例均值/P90 为 `0.31075/0.53125`，有效答案存在率为 `47,740/67,584=0.70638`，aborted ratio 为 `0`。截断仍是正式评测需要报告的限制。
- 峰值监控 GPU 显存为 `40,145 MiB`，最小余量 `815 MiB`，四卡同时活跃；峰值主机 RSS `148.615 GiB / 220 GiB`。PBS accounting 仍错误显示 `resources_used.ngpus=0`，因此按实际请求并使用的 4×A100 与 walltime 估算约 `60.296 GPU·h / 3,858.92 SU`，低于单段预留 `96 GPU·h / 6,144 SU` 和正式上限 `288 GPU·h / 18,432 SU`。
- 20-step 窗口趋势仅作描述：top-k overlap 从 `0.72790` 升至 `0.95226`，shared-token advantage 从 `-0.01837` 接近 `-0.000305`，Student-minus-Teacher entropy 从 `-0.07722` 到 `0.00710`；没有添加上游不存在的数值阈值，也不把这些训练信号解释为独立任务收益。
- W&B 使用单一 online run `opd-d7-9-seed2026-21110343`，API 核验状态为 `finished` 且最终 `training/global_step=528`：https://wandb.ai/2523767186-queen-mary-university-of-london/OPD-NSCC/runs/opd-d7-9-seed2026-21110343 。本阶段没有启动独立终评、D10 或 G3，也不据训练 reward 宣称模型能力提升。


---

04   /   DAYS 12–14

## 独立评测、结论与止损条件

### 终评协议：先冻结，再执行

使用仓库 scripts/val/ 的独立评测路径。仓库明确提醒原版 verl v0.7.0 的内置验证可能低估结果，并建议关闭训练内验证、单独评测；本计划沿用该方式。[1]

| 项目 | 冻结口径 |
| --- | --- |
| 模型与 checkpoint | 初始 student、冻结 teacher、按 dev 规则选出的一个最终 student；三者使用完全相同协议。保留最终训练步数与模型 revision。 |
| 必做数据与采样 | MATH-500 全集，每题 n=4；AIME24 每题 n=16。temperature=0.7、top_p=0.95；评测 max response 先固定 7168，所有模型一致。预算不足需在正式基线前统一降采样数。 |
| 评分与指标 | 默认同一确定性规则评分器并保存逐题结果。主指标是多次采样平均正确率（pass@1 的估计），不是“至少答对一次”的 pass@k；记录输出截断率与无答案率。 |
| 误差与公平性 | 按题目做 paired bootstrap / 置信区间；同一题的多次生成不是独立题目。与论文的长度、采样数、评分器不同，就注明不能直接对表。 |

### 报告必须回答的三个问题

**是否真的学到了？** 比较初始与最终 student，报告绝对分数差和不确定性；训练 loss 变好不是任务能力提高的充分证据。
**过程是否可信？** 给出训练状态、response 长度 / 截断、teacher 信号，以及支持已有实现的 rollout / actor 概率差诊断和 student / teacher 的 overlap、advantage、entropy-gap 趋势。
**投入是否值得？** 汇总 GPU 卡时、SU、每阶段耗时、关键补丁和最主要失败原因。

### 止损表：发现问题先停扩量

| 触发条件 | 立即处理 |
| --- | --- |
| 同一阶段重复 OOM | 先缩动态 token packing、teacher scoring micro-batch、rollout 并发，检查 offload；再考虑短 response。一次只改一个变量，并重跑最短复现。 |
| loss / gradient 非有限值，或 teacher 信号异常恒零 | 暂停长作业；核对 token 对齐、mask、dtype、clamp、teacher 路径与冻结状态。不用加训练时长掩盖错误。 |
| 预算预测超上限 / 排队拖过里程碑 | 优先取消可选对照，或明确降为固定步数；保留 baseline、独立评测和可恢复 checkpoint。日期顺延，不跳过验收。 |
| 完整训练后没有可靠增益 | 项目仍可技术验收，但不能称收益复现成功；如实报告无显著增益，检查评测、长度与分布问题，只做有针对性的补充实验。 |

**可选扩展（二选一）：** 第二个训练 seed，或固定模型 / 数据 / 长度后用 LOG_PROB_TOP_K=0 做 sampled-token 对照。[1] 同时报样本、生成 token 和 GPU 卡时；不要只按“相同步数”宣称算力公平。本期不做 1-query OPD、SFT、重训 teacher 或 GRPO 大对比。

#### 2026-09-16 D12–14 独立终评与 G3 记录

- 测试集开始前将正式训练唯一的 full-epoch checkpoint `global_step_528` 冻结为最终 Student；没有查看 MATH-500/AIME24 后挑 checkpoint。原计划写有“按 dev 规则选择”，但本次单 epoch 正式作业没有预声明 checkpoint sweep/选择规则，因此用预先冻结的最终步避免事后挑选。`dev_256` 只作诊断，不参与选择。
- 独立评测 Job `21857038.pbs101` 使用 1×A100，PBS `Exit_status=0`、实际 walltime `00:37:15`。它用上游 `verl/scripts/legacy_model_merger.py` 合并 step 528，并严格复用 D4 的 prompt、vLLM worker 和 `scripts/val/eval/utils.py::grade_answer_verl`：MATH-500 全 500 题、n=4；AIME24 全 30 题、n=16；temperature `0.7`、top_p `0.95`、max response `7168`、seed `2026`。初始 Student、Teacher 与最终 Student 的数据 SHA256 和协议逐字段一致。
- MATH-500 的 pass@1 估计为初始 Student `0.7455`、最终 Student `0.8160`、Teacher `0.8250`。最终相对初始提升 `+0.0705`，按题目 paired bootstrap 10,000 次的 95% CI 为 `[+0.0465,+0.0945]`，判为可靠增益；最终相对 Teacher 为 `-0.0090`，CI `[-0.0235,+0.0055]`，证据不足以区分。
- AIME24 的 pass@1 估计为初始 Student `0.18333`、最终 Student `0.32708`、Teacher `0.31458`。最终相对初始提升 `+0.14375`，95% CI `[+0.07292,+0.22083]`，判为可靠增益；最终相对 Teacher 为 `+0.0125`，CI `[-0.03333,+0.05833]`，证据不足以区分。
- 截断与无 `boxed` 答案均改善但未消失：MATH-500 截断 `0.2035→0.1250`、missing-box `0.1995→0.1290`；AIME24 截断 `0.79583→0.62292`、missing-box `0.78542→0.56667`。`dev_256` 最终分数 `0.01953` 且截断率 `0.98828`，因冻结 max response 2048 且仅作诊断，不能用来推翻或选择 held-out 结果。
- 逐题差异中，MATH-500 为 130 题改善、43 题回退、327 题不变；AIME24 为 17 题改善、2 题回退、11 题不变。完整失败案例、截断和 missing-box 统计保存在 `eval/final/21857038_pbs101/reports/failure_cases.json`。
- 最终判定为 `REPRODUCED_RELIABLE_GAIN`：两个必做 held-out 任务的 paired 95% CI 下界均大于 0，收益复现门禁通过，完整训练后“无可靠增益”的止损条件未触发。没有自动启动可选第二 seed、sampled-token 对照或任何本期边界外实验。
- 正式训练约 `60.296 GPU·h / 3,858.92 SU`；本次终评 `0.6208 GPU·h / 39.73 SU`，远低于终评 `48 GPU·h / 3,072 SU` 上限。具有可恢复 walltime 证据的 accepted-path 合计约 `66.992 GPU·h / 4,287.48 SU`；不含 CPU-only、取消和失败诊断，不能当账户完整账单。
- **最终门禁：G3 PASS（服务器内可复跑）。** 逐样本输出、summary、acceptance、10,000 次题目级 bootstrap、失败案例、结论报告和 SHA256 清单全部完成。权威结果为 `eval/final/21857038_pbs101/`、`reports/NSCC_OPD_FINAL_REPORT.md`、`reports/NSCC_OPD_FINAL_EVALUATION.json` 与 `reports/compute_ledger.json`；训练曲线仍为同一 finished W&B run。当前工作树含项目补丁但未提交 Git，跨服务器分发时必须连同这些未提交脚本/配置或补丁一起保存。

### Sampled-token 对照预注册（执行前冻结）

- 本期可选扩展选择 `LOG_PROB_TOP_K=0` sampled-token 对照；不再执行第二训练 seed，也不扩展到 1-query OPD、SFT、Teacher 重训或 GRPO。
- 对照从同一个初始 Student 开始，固定 Teacher、训练/Dev 数据及其顺序、seed `2026`、1024/7168 长度、B32、n4、actor token budget/GPU 8192、micro-batch 1/1、`enforce_eager=False`、`token_reward_direct`、学习率 `1e-6`、1 epoch / 528 steps、checkpoint 频率和其余正式参数；相对主实验唯一算法变量是 `actor_rollout_ref.rollout.log_prob_top_k: 16→0`。`top_k_strategy=only_stu` 与 `reward_weight_mode=student_p` 保留在配置中以维持逐字段一致，但在上游 `top_k=0` 路径不参与 top-k 奖励计算。
- 上游 `top_k=0` 路径直接在实际采样 token 上计算 `teacher_logp - student_logp`，继续通过同一个 `token_reward_direct` advantage 和 actor policy-gradient 更新；不修改 vendored veRL 核心算法。
- 测试集开始前预承诺唯一 full-epoch `global_step_528` 为 sampled-token 最终 checkpoint，不根据 dev、MATH-500 或 AIME24 事后挑选。独立评测严格复用已冻结协议：MATH-500 n=4、AIME24 n=16、temperature 0.7、top_p 0.95、max response 7168、seed 2026、同一规则评分器和题目级 paired bootstrap。
- 最终同时比较 sampled-token vs 初始 Student、sampled-token vs top-k16 OPD，并报告每组 rollout 样本数、生成 token 数、实际 GPU·h/SU、截断率与无答案率；“相同步数”不作为算力公平的充分证据。
- 训练验收继续要求 528 步连续闭环、Teacher 冻结、response-only mask、rollout 权重同步、有限 advantage/loss/gradient、完整 checkpoint、在线 W&B 与资源记录。sampled-token 模式不产生 top-k overlap/shared-token 指标，因此只报告 entropy gap 趋势，不把不适用指标设为门禁。

#### 2026-09-17 Sampled-token 对照执行与终评记录

- 正式 sampled-token Job `21908511.pbs101` 保持预注册配置，仅将 `log_prob_top_k` 从 16 改为 0；单个 primary segment 连续完成 step `1–528`、覆盖同一训练集 1 epoch，最终 `global_step_528` 的 model/optimizer/extra-state 各 4 shard 与 dataloader `data.pt` 结构检查通过。训练生成 `67,584` 条 rollout、覆盖 `16,896` 个唯一 prompt；Teacher 冻结、response-only mask、rollout 权重同步及有限 advantage/loss/gradient 均通过验收。
- 训练 walltime 为 `14:05:53`，按实际 4×A100 计约 `56.392 GPU·h / 3,609.10 SU`。response 均值 `5,465.46` tokens，估算生成 response token `369,377,684`，hit-length-limit 比例 `0.30597`，有效答案存在率 `0.71122`，aborted ratio `0`。相同 `67,584` 个 rollout 下，top-k16 为 `60.296 GPU·h / 3,858.92 SU`、`369,239,639` tokens；sampled-token 使用约 `93.53%` 的 GPU·h/SU（减少约 `6.47%`），生成 token 数为 `100.037%`，不能把节省归因于少生成样本或明显少生成 token。
- PBS 对原训练 Job 记录 `Exit_status=1`，失败发生在 528/528 与 checkpoint 保存后的自动收尾，原因不是训练或模型，而是 Scratch 环境中 `python-3.12.0-hab00c5b_0_cpython` 的元数据仍在但 1,930 个受管文件中缺失 1,439 个。离线用完全相同的锁定 Conda 包原位修复后，缺失数为 0；环境修复 Job `22657007.pbs101` 与 sampled-token 收尾 Job `22701226.pbs101` 均为 `Exit_status=0`。最终训练 acceptance 为 `passed=true`、`PASS_TRAINING_COMPLETE`，W&B 单一 online run 状态为 `finished`、最终 step 528：https://wandb.ai/2523767186-queen-mary-university-of-london/OPD-NSCC/runs/opd-sampled-token-seed2026-21908511 。
- 独立评测 Job `22729409.pbs101` 使用 1×A100，PBS `Exit_status=0`、walltime `00:46:54`，约 `0.7817 GPU·h / 50.03 SU`；严格复用 D12 上游 FSDP merger、同一数据哈希、prompt、规则评分器及冻结的 MATH-500 n=4、AIME24 n=16、temperature `0.7`、top_p `0.95`、max response `7168`、seed `2026` 协议。此前三个 wrapper 诊断 Job `22705247`、`22717342`、`22723399` 均在模型合并/生成前退出，分别暴露相对路径 manifest、维护后 PBS 历史消失、以及主线未保存独立 qstat 文件；最小修复只改变证据读取，不改变模型或评测协议，也未污染最终结果目录。
- MATH-500：初始 Student `0.7455`、sampled-token `0.8160`、top-k16 `0.8160`。sampled-token−初始为 `+0.0705`，10,000 次题目级 paired bootstrap 95% CI `[+0.0490,+0.0920]`，可靠增益；sampled-token−top-k16 为 `0.0000`，CI `[-0.0135,+0.0140]`，inconclusive。sampled-token 截断/missing-box 为 `0.1260/0.1260`，top-k16 为 `0.1250/0.1290`。
- AIME24：初始 Student `0.18333`、sampled-token `0.31250`、top-k16 `0.32708`。sampled-token−初始为 `+0.12917`，95% CI `[+0.06875,+0.19792]`，可靠增益；sampled-token−top-k16 为 `-0.01458`，CI `[-0.05417,+0.02292]`，inconclusive。sampled-token 截断/missing-box 为 `0.62708/0.57917`，top-k16 为 `0.62292/0.56667`；长回答截断仍是两种方法共同的主要限制。
- 结论为 sampled-token vs 初始 Student `REPRODUCED_RELIABLE_GAIN`，sampled-token vs top-k16 `INCONCLUSIVE_NO_RELIABLE_DIFFERENCE`。这表示本 seed/协议下 sampled-token 保留了相对初始模型的可靠收益，并以约 6.47% 较低训练 GPU·h 达到与 top-k16 统计上不可区分的结果；不能据此宣称 sampled-token 精度更优或普遍更高效，第二 seed 未执行。
- 权威评测根为 `eval/sampled_token/22729409_pbs101/`；发布结果为 `reports/NSCC_OPD_SAMPLED_TOKEN_REPORT.md` 与 `reports/NSCC_OPD_SAMPLED_TOKEN_EVALUATION.json`。冻结计划、逐样本输出、acceptance、summary、failure cases、分析报告、导出权重及项目级报告均进入 `artifact_sha256.txt`，作业内和事后复核全部为 `OK`。最终回归 Job `22725534.pbs101` 为 `51 passed, 1 warning`，并通过 py_compile、PBS `bash -n` 与定向 `git diff --check`；warning 仅为既存 `pynvml` 弃用提示。


---

05   /   RESOURCE CONTROL

## 算力预算与 NSCC 执行规范

NSCC ASPIRE 2A 官方 GPU 计费为 **64 SU / GPU·小时**；因此四卡运行 1 小时为 256 SU，24 小时为 6,144 SU。[6] 下表是**可分配预算，不是运行时间预测或已发生消费**；CPU-only 准备作业另记。

| 阶段 | 预留 GPU·小时 | 折算 SU | 预算含义 |
| --- | --- | --- | --- |
| 环境 / 单模型推理 | 8 | 512 | 仅必要步骤用 GPU；不占四卡下载数据。 |
| 四卡 smoke + 短恢复 | 16 | 1,024 | 累计约 4 个四卡小时，可拆多个短作业。 |
| 初始 student / teacher 基线 | 32 | 2,048 | 用 1–2 卡独立评测，不要求四卡常驻。 |
| Pilot 与性能校准 | 32 | 2,048 | 累计约 8 个四卡小时。 |
| 正式 OPD 主线 | 288 | 18,432 | 最多约 72 个四卡小时，分段使用。 |
| 最终评测与验证 | 48 | 3,072 | 按评测吞吐决定并行度。 |
| 故障 / 重跑缓冲 | 64 | 4,096 | 仅用于定位明确的问题，不自动消耗。 |
| **合计** | **488** | **31,232** | 主线预算约 32k SU 封顶。 |

> **实际预算上限** D1 重新查询可用额度 A，建议本期总上限取 min(32,000 SU, 0.5×A)，为其他项目保留余量。不要把旧余额当 A；Balance / In Doubt 的可用口径以账户当前输出为准，避免重复扣减。[6]

### 队列与提交：沿用 PBS，不照搬 SBATCH

当前官方队列表中 gdev 为 ≤2 小时，四卡常规 g3 为 >2–24 小时；超过 24 小时另有长队列。优先通过 normal 路由队列提交，并以当日实际权限 / 路由为准；ai 队列权限不能假定已经具备。[6] 原仓库头部的 SBATCH 是另一套调度器指令，不能当作 NSCC 资源申请。[2]

```bash
# PBS 资源骨架；不是完整训练启动脚本
#PBS -q normal
#PBS -l select=1:ngpus=4
#PBS -l walltime=24:00:00
#PBS -P <D1 核实的项目 ID>
# smoke 将 walltime 改为 02:00:00，并绑定已验证的 launcher。
```

### 存储、断连与退出

Home 配额与 scratch 使用情况先核对；官方说明 scratch 中超过 30 天未访问的文件可能被清理，不能把它当唯一备份。[6] 建议只长期保留代码、配置、日志摘要、评分输出及关键 checkpoint；模型缓存和中间 checkpoint 做容量管理。

采用 PBS batch 承载训练，不靠登录终端维持主进程。每段按实测 step 与保存耗时预留退出余量；不能把 PBS 强杀当保存机制。作业结束确认资源释放；Ray 进程管理限定在自己的作业范围，避免误停其他任务。


---

06   /   HANDOFF & SOURCES

## 交付清单与执行记录

### 最终目录建议

```text
OPD/
  configs/nscc/         # smoke / pilot / main 的已解析配置
  scripts/nscc/         # PBS 包装、四卡 launcher、续训入口
  repro/               # commit、依赖、模型 revision、数据哈希
  eval/                # baseline / teacher / final 的逐题输出
  reports/             # 曲线、对比表、资源账单、结论文档
  docs/nscc_opd_timeline.md
# checkpoint 存放位置写入 manifest，不将大权重提交 Git。
```

### 验收定义

**技术完成：** 新环境可重建；训练和恢复通过；配置与原版差异可追踪；初始 / 最终模型按同一协议独立评分。
**收益复现：** 只有在可信评测中观察到相应提升，才声称该模型对的收益被复现；否则报告负结果或证据不足。
**本期结束：** 交付 README、结果表、训练曲线、逐题输出、算力记录和关键 checkpoint；不以“必须涨多少分”倒推结论。

### 每天只填这一行

```text
日期 | 阶段/Job ID | commit/run/step | GPU·h/SU
本次结果 | 当前阻塞 | 下一次最小验证 | 可用额度 / 预算余量
```

**第一天的完成标准：** 查清当前额度；固定模型对与实现路线；冻结 commit；建立独立目录；写好 manifest 草稿。第一天不要求把完整 OPD 训练跑起来。

### 来源与边界（核对日期：2026-09-13）

[1] [OPD README](https://github.com/daydreamer17/OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/README.md) — 环境、默认方法、数据与独立评测建议；作者实验硬件为 8×A800 80GB。

[2] [原版 on_policy_distillation.sh](https://github.com/daydreamer17/OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/on_policy_distillation.sh) — 原版参数、硬编码、动态 token budget、teacher micro-batch 与调度器头部。

[3] [定制 verl 的 main_ppo.py](https://github.com/daydreamer17/OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/verl/verl/trainer/main_ppo.py) — reward_model.enable_resource_pool=False 时，teacher 与 actor 共用 global_pool。

[4] [新版 verl_example/opd.sh](https://github.com/daydreamer17/OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/verl_example/opd.sh) — 新版参数命名、分离的 student / teacher 配置及 loss 选项。

[5] [verl 官方 OPD 文档](https://verl.readthedocs.io/en/latest/algo/opd.html) — 新版独立 teacher resource pool；该页面会随版本更新，实际运行要锁版本。

[6] [NSCC ASPIRE 2A FAQ](https://help.nscc.sg/aspire2a/faqs/) — GPU SU 费率、PBS 路由 / 时限、配额查询、预扣额度与 scratch 清理政策。

[7] [JustRL README](https://github.com/thunlp/JustRL/blob/main/README.md) — JustRL-DeepSeek-1.5B 权重入口及其基础模型。

[8] [本次核对的 OPD commit](https://github.com/daydreamer17/OPD/commit/ac26e38d6f1572eb027597b48a9f4e01f6915ef8) — 本文件的代码基线；后续修订需记录新的 commit 和补丁。

[9] [冻结 commit 的 rollout / actor debug metrics](https://github.com/daydreamer17/OPD/blob/ac26e38d6f1572eb027597b48a9f4e01f6915ef8/verl/verl/utils/debug/metrics.py)
— 上游在 response mask 上比较概率差 max / mean / std 与 Pearson，并明确用于 debugging；
代码没有数值阈值或 pass / fail 断言。

[10] [OPD 论文的动态对齐分析](https://arxiv.org/html/2604.13016)
— student / teacher 的 top-k overlap ratio、overlap-token advantage 与 entropy gap
用于描述训练中的策略对齐；成功表现看跨训练过程的方向性趋势，不是统一单步硬阈值。

**尚未验证：** 本阶段未重新查询终态 NSCC 账户余额，第二训练 seed 仍未执行。`LOG_PROB_TOP_K=0` sampled-token 对照已经完成 528 步训练、同协议独立评测和与 top-k16 的配对比较；两种方法相对初始 Student 均取得可靠增益，但二者差异在 MATH-500 与 AIME24 上都未达到统计显著。依赖、四卡峰值/吞吐、正式训练和冻结协议下的独立增益已经实测，详见各阶段执行记录。日期、门槛和预算仍是本项目的规划选择，不是上游仓库默认承诺。
