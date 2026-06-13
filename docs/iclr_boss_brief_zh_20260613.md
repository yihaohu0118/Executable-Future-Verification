# EFV ICLR Boss Brief

Date: 2026-06-13

## 一句话

我们现在的方向不是“再做一个 world model”，而是：

> future 生成越来越便宜，但 future 选择仍然很脆弱；给定多个候选未来，学习/验证哪个未来真的可执行、能完成任务。

当前最合适的名字是：

**Executable-Future Verification for Robot World-Action Candidates**

## 当前判断

值得继续，但不能无边界继续。

当前证据等级是：

```text
single_benchmark_mechanism
```

也就是说，RoboCasa365 已经有强机制证据，但还不能说 multi-benchmark ICLR-ready。现在应该继续投入一个明确窗口，把 RoboTwin2 和一个 world-model/trust diagnostic 层补起来；如果 RoboTwin2 做不出稳定 headroom 和 anti-template 结果，就应降级成 RoboCasa365 diagnostic / workshop paper。

## 核心反直觉点

1. 更多 action 细节不一定更好。
   在 energy-matched hard negatives 下，action-only 和 action magnitude/smoothness heuristic 仍然失败，而 EEF/gripper/robot execution envelope 可以恢复成功候选。

2. object state 不是 RoboCasa 最强结果的主要信号。
   object-only 只有 31.0/64，robot-only 达到 64.0/64，EEF+gripper distribution 达到 63.6/64。

3. 这不是 zero-shot universal verifier。
   leave-one-task-out/source-only 只有 25.6/64；四个 target-task calibration cases 恢复到 59.2/64，八个恢复到 62.2/64。所以故事应该是 few-shot task/contact-conditioned verifier。

4. RoboTwin2 暴露了一个更好的机制故事。
   `stack_blocks_two` 上 gripper/DTW-gripper 会失败，这反而说明 verifier 不能只靠 gripper timing；多阶段空间任务需要 object/contact relation envelope。

## 已有最强证据

### RoboCasa365

RoboCasa365 是当前唯一可以算作 passed 的主证据层。

| Selector / Bound | Success |
| --- | ---: |
| Rank0 conservative prior | 0/64 |
| Oracle-best | 64/64 |
| Action-only endpoint-free | 28.4/64 |
| Object-only trace | 31.0/64 |
| Proprio-only | 63.0/64 |
| Robot-only | 64.0/64 |
| EEF+gripper distribution | 63.6/64 |
| Same-task nearest-positive EEF+gripper | 59.0/64 |
| Source-only no-task-ID transfer | 25.6/64 |
| Four-shot target calibration | 59.2/64 |
| Eight-shot target calibration | 62.2/64 |

这支持的 claim 是：

> 在 shortcut-controlled hard negatives 下，compact robot execution envelope 能恢复 rank0、action-only、object-only 都错过的 executable futures。

### RoboTwin2

RoboTwin2 现在还不能当主结果。

当前状态：

| Task | Cases | Rank0 | Oracle | Useful signal |
| --- | ---: | ---: | ---: | --- |
| `stack_blocks_two` | 2 | 0/2 | 2/2 | 好的机制任务；gripper/DTW-gripper 失败，但 relation trace coverage 还没有。 |
| `open_laptop` | 2 | 0/2 | 2/2 | permissive counterexample；smoothness/gripper/DTW-gripper 都能成功。 |
| `stamp_seal` | 5 | 0/5 | 5/5 | contact positive control；gripper envelope 强，但任务太单一。 |

RoboTwin2 paper-readiness gate 当前失败：

| Check | Current |
| --- | --- |
| base-ready tasks | 2 / min 4 |
| relation-ready tasks | 0 / min 1 |
| non-template success tasks | pass |
| matched-negative tasks | 2 / min 3 |
| strong-envelope tasks | 0 / min 3 |
| relation-rescue tasks | 0 / min 1 |

## 现在不能 claim 什么

不能 claim：

- 已经在多个主流 benchmark 上验证；
- RoboTwin2 已经是第二主 benchmark；
- world-model/trust diagnostic 已经通过；
- 方法理解了通用物理可执行性；
- 方法能真实机器人部署；
- 我们提出了新 world model 或新 robot policy。

当前能 claim：

- RoboCasa365 上有强机制证据；
- future selection 是真实 bottleneck；
- rank0、action-only、object-only、action magnitude 等 shortcut 都不够；
- few-shot task/contact-conditioned execution envelope 是一个很强的 verifier 信号；
- RoboTwin2 和 diagnostic layer 是正在补齐的主风险。

## ICLR 通过线

继续冲完整 ICLR 的最低条件：

1. RoboCasa365 证据保持当前水平，并有 valid evidence card。
2. RoboTwin2 至少 4 个 base-ready tasks。
3. RoboTwin2 至少 1 个 relation-ready / relation-rescue task。
4. RoboTwin2 有 anti-template evidence：
   - diverse non-full-expert successes；
   - matched low-DTW failed negatives。
5. Envelope selector 稳定优于 rank0、random、energy/magnitude、action-only、candidate-ID/rank remap。
6. 至少一个 world-model/trust diagnostic 层通过：
   - multi-candidate judgments；
   - visual/model-score proxy 会失败；
   - EFV score 打败 visual/model-score proxy。

## 杀停线

如果出现以下情况，应降级为 RoboCasa365 diagnostic / workshop：

- RoboTwin2 做不到 4 个稳定 base-ready tasks；
- 成功候选主要还是 full expert / nearest expert template；
- DTW nearest-positive 基本等于最强 verifier；
- relation features 不能在任何空间/多阶段任务上 rescue gripper failure；
- world-model diagnostic 数据长期不可用，或者只能做 prompt/request pipeline，不能形成 multi-candidate judgment manifest。

## 下一步最有效动作

等 GPU 真正空闲后，只跑 RoboTwin2 primary window，不再扩散：

```bash
cd /home/yihao_hyh/Executable-Future-Verification
EXECUTE=1 GPU_ID=auto SEEDS=0-7 scripts/robotwin2_iclr_window_launcher.sh \
  /home/yihao_hyh/efv_runs/robotwin2_iclr_window_YYYYMMDD
```

注意：低 utilization 不等于空闲。必须满足 `free_gpus != none`，即无 compute PID 且显存低于阈值。

跑完后必须自动生成：

- RoboTwin2 readiness report；
- selector table；
- anti-template diagnostics；
- paper-readiness gate；
- registry-entry proposal；
- evidence card；
- updated ICLR status report。

## 当前一句话结论

这个 idea 现在是一个有潜力的 7/10 方向：切口清楚、RoboCasa 机制证据强、反直觉点真实，但还缺第二 executable benchmark 和 world-model diagnostic 的主证据闭环。下一阶段不该继续扩 benchmark，而是用 RoboTwin2 证明它不是 expert-template matching，再用一个 2025-2026 diagnostic benchmark 证明 visual/model-score proxy 也不够。
