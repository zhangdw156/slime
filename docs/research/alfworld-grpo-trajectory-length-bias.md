# ALFWorld 多步 GRPO 中的轨迹长度偏置

**slime 与 verl-agent 的机制差异、数值例子与论文研究假设**

> 状态：内部研究备忘录
> 日期：2026-07-11
> 对比提交：
> - slime 开发分支：[feat/astra-v0.3.0](https://github.com/zhangdw156/slime/tree/feat/astra-v0.3.0)
> - slime 固定快照：[21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b](https://github.com/zhangdw156/slime/tree/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b)
> - verl-agent 开发分支：[exp/h20](https://github.com/zhangdw156/verl-agent/tree/exp/h20)
> - verl-agent 固定快照：[287d52e088675d7d5adb0bec621e1fb53b40d28b](https://github.com/zhangdw156/verl-agent/tree/287d52e088675d7d5adb0bec621e1fb53b40d28b)
>
> 下文所有代码证据均使用固定 commit 的 GitHub `blob/<commit>/...#Lx-Ly` permalink；开发分支 URL 仅用于查看后续演进。
>
> 本文只讨论上述提交和当前 ALFWorld 普通 GRPO **训练路径**。它记录了代码机制、可复现的数学例子和待验证的论文假设，不把当前相关性观察表述为已经完成因果证明。

## 摘要

slime 和 verl-agent 都会将一条 ALFWorld 多步轨迹展开为多个 step-level 训练样本，但两者对这些样本的后续处理并不相同。

- slime 在计算 GRPO advantage 前，先按 trajectory 去重，使每条 trajectory 在同一 prompt group 中只贡献一个 episode reward；训练时再使用 `group_id`、`group_mask_sums` 和 group-aware scheduler，将同一 trajectory 的所有 step token 聚合为一个 trajectory-level token mean。
- verl-agent 默认让展平后的每个 step row 都参与 `uid` group 的 reward mean/std，并使用 step-row/token 级 mini-batch 更新；一条更长的 trajectory 因而会产生更多统计样本、更多训练 token，以及更多实际 Adam updates。
- 两边虽然都采用满分 1% 的 invalid-action penalty，但 slime 将 invalid 总数写入 trajectory final reward，并让动态过滤看到 penalty 后的 reward；verl-agent 则只修改当前 invalid step row 的 raw score，随后该变化通过 `uid` mean/std 耦合到组内所有 rows。两者分别形成 trajectory-level collective shaping 与 raw-score-local、advantage-group-coupled shaping。

这一区别提供了一个与现有实验曲线一致的机制解释：

1. verl-agent 在训练前期可能凭借稀有成功 step 的高正向 advantage，以及每个外层训练 step 内更多的 Adam updates，实现更快的初始提升。
2. 长失败轨迹在 verl-agent 中会带来更密集的负向状态覆盖和更严重的 outcome-level 错误 credit assignment，可能使合理的中间动作也被反复压低，最终更早达到平台。
3. slime 的更新更保守，但 trajectory-level 等权使训练目标更接近 episode success rate，并限制轨迹长度导致的重复权重，可能带来更高的后期上限。
4. slime 当前启用的动态后过滤会拒绝 reward std 不超过阈值的 prompt groups 并继续补采样；当这类 groups 与训练阶段或任务难度相关时，它可能影响前期 wall-clock 效率和后期有效训练信号。
5. trajectory-level penalty 与 post-penalty dynamic filtering 会保留 task outcome 相同但 invalid count 不同的 groups，可能在全失败阶段和普遍成功阶段继续提供行为质量信号。

这些机制共同构成一个值得系统验证的研究方向：**多步 agentic GRPO 的统计单位和优化单位应当是 step、token，还是 trajectory？**

## 五机制统一框架

当前 slime ALFWorld GRPO 可以被定义为一个由五项机制组成的
trajectory-aware GRPO stack：

```text
trajectory penalty
+ trajectory advantage
+ trajectory reducer
+ group scheduler
+ penalty-aware dynamic filter
```

与当前 verl-agent native GRPO 的对应关系为：

| 机制 | slime | verl-agent |
|---|---|---|
| M1 Penalty | trajectory invalid count 进入 final reward | invalid step 修改当前 raw score，advantage 组耦合 |
| M2 Advantage | 按唯一 trajectories 计算 mean/std | 按展开后的 step rows 计算 mean/std |
| M3 Reducer | 每条 trajectory 先形成一个 token mean | mini-batch 内直接 token mean |
| M4 Scheduler | 按 trajectory groups 决定 optimizer updates | 按 flattened rows 切 optimizer mini-batches |
| M5 Dynamic filter | 按 post-penalty trajectory reward 方差过滤 | 当前 native launcher 默认关闭 |

这五项机制又可以分成三个层次：

```text
学习信号构造：
  M1 trajectory penalty
  M2 trajectory advantage

梯度优化：
  M3 trajectory reducer
  M4 group scheduler

数据选择：
  M5 penalty-aware dynamic filter
```

三种角色、四个统计层和五项机制的关系如下：

| 机制 | 机制角色 | 对应统计层 | 说明 |
|---|---|---|---|
| M1 Penalty | 学习信号构造 | Episode reward | 决定 invalid 信号如何进入 processed reward |
| M2 Advantage | 学习信号构造 | Scalar advantage | 决定 mean/std 的统计单位 |
| M3 Reducer | 梯度优化 | Token-level loss reduction | 决定长短 trajectory 在单次 update 中的相对权重 |
| M4 Scheduler | 梯度优化 | Optimizer update schedule | 决定一次 rollout 产生多少次参数更新 |
| M5 Dynamic filter | 数据选择 | Accepted-data selection | 在 M1 reward shaping 后、M2 advantage 前决定哪些 groups 进入后续计算 |

其中“四个统计层”描述从 reward 到 optimizer 的计算链路。M5 在 M1 已经
产生 post-penalty final reward 后执行，决定哪些 groups 继续进入 M2
advantage、M3 reducer 和 M4 scheduler。因此，“四个统计层”和“五项机制”
并不矛盾。

### M1：Trajectory penalty

slime 使用：

$$
\tilde R_\tau=Y_\tau-0.01K_\tau
$$

其中 \(K_\tau\) 为整条 trajectory 的 invalid action 总数。一个 invalid
会降低整条 trajectory 的 final reward，并让该 trajectory 的所有 steps
共享相应 advantage。

verl-agent 仅在 invalid step raw score 上减去 `0.1`：

$$
\tilde r_{\tau,s}=10Y_\tau-0.1I_{\tau,s}
$$

但该 raw-score 修改会改变 `uid` group 的 mean/std，因此最终 advantage
仍然是组耦合的。

直观例子：8 条 trajectories 全部失败且
`K=[0,1,2,3,4,5,6,7]` 时，slime 会按整条 trajectory invalid count
排序；verl-agent 则会强烈惩罚具体 invalid rows，并轻微奖励其余 valid
rows。详细计算见第 10.4 节。

### M2：Trajectory advantage

在核心 8-rollout 例子中：

```text
成功 trajectories：长度 10、11、14
失败 trajectories：5 条，每条长度 50
```

slime 按 8 条 trajectories 计算：

```text
成功 advantage = +1.207612
失败 advantage = -0.724567
```

verl-agent 按 285 个 step rows 计算：

```text
成功 step advantage = +2.667919
失败 step advantage = -0.373509
```

因此，verl-agent 中稀有的成功 steps 获得更高的正向 advantage，而大量
失败 steps 各自获得较小的负向 advantage。详细推导见第 5 节。

### M3：Trajectory reducer

Reducer 决定**一次 optimizer update 内每条 trajectory 占多大权重**。

假设：

```text
Trajectory A：两个 token losses = [2, 4]
Trajectory B：一个 token loss = [10]
```

verl-agent token mean：

$$
L_{\text{verl}}=\frac{2+4+10}{3}=5.33
$$

此时 A 因 token 数更多，占有 `2/3` 的 token positions。

slime trajectory reducer：

$$
L_A=\frac{2+4}{2}=3,\qquad L_B=10
$$

$$
L_{\text{slime}}=\frac{L_A+L_B}{2}=6.5
$$

此时 A 和 B 在外层各占一条 trajectory 的权重。

Reducer 的长度加权机制是确定存在的，但其最终性能影响和方向并不确定。
GRPO advantage magnitude 会部分反向补偿 step 数量，实际梯度还受到状态、
动作、token 数、clipping 和 mini-batch 顺序影响，因此必须消融验证。

### M4：Group scheduler

Scheduler 决定**一次 rollout 最终执行多少次 `optimizer.step()`**。

当前 slime launcher：

```text
16 prompts × 8 trajectories = 128 groups
global_batch_size = 128 groups
=> 1 optimizer.step
```

同样的轨迹分布在 verl-agent 中，如果完整 batch 展开为 4560 rows：

```text
4560 rows
-> padding 到 4608
-> 4608 / 256
-> 18 optimizer steps
```

如果训练早期 128 条 trajectories 都运行到 50 步：

```text
128 × 50 / 256 = 25 optimizer steps
```

因此，scheduler 是解释 verl-agent 按 tracker/global step 观察时前期
收敛更快的最强候选机制。slime 虽然也运行很多 microbatches，但会先累积
所有 group gradients，再统一执行一次 optimizer update。

### M5：Penalty-aware dynamic filter

slime 当前使用 post-penalty trajectory final reward 做 nonzero-std
filter：

```text
trajectory penalty
-> final reward variance
-> dynamic filter
-> accepted prompt groups
```

它会：

- 确定性地拒绝 final reward `std <= 1e-6` 的 groups；
- 持续补采样，直到收集到目标数量的 `std > 1e-6` groups；
- 保留 task outcome 相同但 invalid count 不同、因而 final reward 有差异
  的 groups。

如果零方差 groups 在训练前期大量出现，该机制可能增加 rollout/环境交互
成本；如果零方差与后期已掌握任务相关，它可能表现为困难样本课程。但
“零方差”等同于“简单样本”并不是代码事实，这些学习阶段解释仍属于待验证
假设。该机制也可能产生选择偏差，因此需要与 trajectory penalty 做 2×2
消融。

### 当前机制假设

当前最值得优先验证的假设是：

| 现象 | 首要候选机制 | 其他候选机制 |
|---|---|---|
| verl-agent 前期提升更快 | M4 group scheduler | M2 step-row advantage |
| slime 后期仍持续提升 | M5 dynamic filter | M1 trajectory penalty |
| slime 最终上限更高 | M1 + M2 | M3 reducer + M5 filter |
| 长失败轨迹的重复影响 | M2 + M3 + M4 | M1 penalty |

这张表只是机制探索优先级，不是已经建立的因果结论。

---

## 1. 研究问题

### 1.1 现象

当前实验观察为：

- verl-agent 前期提升更快；
- slime 前期提升较慢；
- 随训练继续，slime 后期超过 verl-agent，并达到更高上限。

同时，ALFWorld 轨迹长度具有稳定的结果相关性：

- 成功轨迹通常较短；
- 失败轨迹通常运行到较大的最大步数，例如 50 步。

### 1.2 核心问题

在成功与失败轨迹长度显著不同的情况下：

1. step-level 展平是否改变 GRPO 的 advantage 分布？
2. step-level 展平是否改变每条 trajectory 的最终梯度权重？
3. 轨迹长度是否改变每个外层训练 step 中的实际参数更新次数？
4. 这些差异能否解释“前期速度”和“后期上限”的权衡？
5. 动态后过滤是否进一步形成一种在线困难样本课程？
6. trajectory-count penalty、raw-score-local penalty 与 penalty-aware filtering 分别贡献了多少最终性能？

### 1.3 必须区分的四个统计层

讨论“轨迹长度影响”时，不能把以下概念合并：

1. **Episode reward**：一条完整轨迹最终获得的任务奖励。
2. **Scalar advantage**：GRPO group normalization 后得到的标量。
3. **Token-level loss reduction**：标量 advantage 广播到 token 后如何聚合。
4. **Optimizer update schedule**：多少 rows/groups 触发一次 `optimizer.step()`。

将 invalid-action penalty 纳入后，slime 与 verl-agent 在上述四层均存在差异。

---

## 2. 共同点：两边都将训练轨迹展开为 step rows

### 2.1 slime

slime 的 ALFWorld rollout 在每个产生 response token 的有效环境 step 上创建一个独立 `Sample`：

- 当前 step 的 prompt；
- 当前 step 的 assistant response；
- 当前 step 的 response token；
- 全 1 的 response `loss_mask`；
- 当前 trajectory 的共享 `group_id`。

代码证据：

- [`examples/alfworld/batched_rollout.py:305-378`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/batched_rollout.py#L305-L378)
- 关键字段：[`examples/alfworld/batched_rollout.py:338-369`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/batched_rollout.py#L338-L369)

一条长度为 \(L_\tau\) 的 trajectory 通常产生 \(L_\tau\) 个 step-level `Sample`。

### 2.2 verl-agent

verl-agent 同样在每个环境 step 后，将当前 batch row 追加到该 trajectory 的 `total_batch_list`：

- [`../verl-agent/agent_system/multi_turn_rollout/rollout_loop.py:332-395`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/agent_system/multi_turn_rollout/rollout_loop.py#L332-L395)

rollout 结束后，`gather_rollout_data()` 将所有 active steps 展平为 `effective_batch`：

- [`../verl-agent/agent_system/multi_turn_rollout/rollout_loop.py:233-283`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/agent_system/multi_turn_rollout/rollout_loop.py#L233-L283)

因此，双方的直接差异并不是“是否 step-level 展平”，而是：

> **step rows 在后续 advantage、loss 和 optimizer scheduler 中被当成什么单位。**

---

## 3. 标识符和统计单位

| 语义 | slime | verl-agent |
|---|---|---|
| 原始 prompt group | `group_index` | `uid` |
| 单条 trajectory | `group_id` | `traj_uid` |
| step-level row | 独立 `Sample.index` | 展平后的 batch row |
| invalid penalty 单位 | trajectory invalid count | step raw-score indicator；normalized advantage 组耦合 |
| advantage 统计单位 | 唯一 trajectory | 默认是每个 step row |
| loss 外层单位 | trajectory group | row/token mini-batch |

### 3.1 slime 标识符

数据源为同一 prompt 创建 `n_samples_per_prompt` 条 trajectory：

- 同一 prompt 共享 `group_index`；
- 每条 trajectory 有唯一 `index`。

代码：

- [`slime/rollout/data_source.py:107-117`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/slime/rollout/data_source.py#L107-L117)

ALFWorld step samples 将 trajectory 的唯一编号写入 `group_id`：

- [`examples/alfworld/batched_rollout.py:338-343`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/batched_rollout.py#L338-L343)

### 3.2 verl-agent 标识符

`env.rollout.n=8` 时：

- 每连续 8 条 trajectory 共享一个 `uid`；
- 每条 trajectory 拥有唯一 `traj_uid`。

代码：

- [`../verl-agent/agent_system/multi_turn_rollout/rollout_loop.py:314-325`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/agent_system/multi_turn_rollout/rollout_loop.py#L314-L325)
- prompt 复制：[`../verl-agent/agent_system/multi_turn_rollout/rollout_loop.py:503-505`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/agent_system/multi_turn_rollout/rollout_loop.py#L503-L505)

需要特别注意：

- `env.rollout.n=8` 是 agent 环境 rollout group size；
- `actor_rollout_ref.rollout.n` 仍为 1，并由入口代码强制断言。

代码：

- [`../verl-agent/verl/trainer/main_ppo.py:152-160`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/main_ppo.py#L152-L160)

---

## 4. Advantage 计算差异

### 4.1 slime：每条 trajectory 只参与一次 mean/std

trajectory 结束后，slime 将同一个最终 episode reward 写入该 trajectory 的所有 step samples：

- [`examples/alfworld/batched_rollout.py:381-435`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/batched_rollout.py#L381-L435)

但在 GRPO reward normalization 前，代码使用：

```text
prompt key     = group_index
trajectory key = group_id
```

构造每个 prompt 下唯一的 trajectory reward 表：

- [`examples/alfworld/generate_with_alfworld.py:440-480`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/generate_with_alfworld.py#L440-L480)

因此：

$$
A_\tau =
\frac{
R_\tau-\mu_{\text{trajectory}}
}{
\sigma_{\text{trajectory}}+\epsilon
}
$$

轨迹有 10 步还是 50 步，都只在 mean/std 中出现一次。

得到 scalar advantage 后，GRPO 将其广播到该 step sample 的全部 response tokens：

- [`slime/utils/ppo_utils.py:201-208`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/slime/utils/ppo_utils.py#L201-L208)

$$
A_{\tau,s,t}=A_\tau
$$

### 4.2 verl-agent：默认跨 step rows 计算 mean/std

verl-agent 将完整 episode reward 写入同一 trajectory 的每个 active step row：

- [`../verl-agent/agent_system/multi_turn_rollout/rollout_loop.py:261-277`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/agent_system/multi_turn_rollout/rollout_loop.py#L261-L277)

`EpisodeRewardManager` 再把这个 reward 写入每个 step response 的最后一个有效 token：

- [`../verl-agent/agent_system/reward_manager/episode.py:39-79`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/agent_system/reward_manager/episode.py#L39-L79)

普通 GRPO 调用：

- 使用 `uid` 作为 prompt group；
- 传入 `traj_uid`；
- 但没有关闭 `compute_mean_std_cross_steps`。

调用代码：

- [`../verl-agent/verl/trainer/ppo/ray_trainer.py:283-299`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/ppo/ray_trainer.py#L283-L299)

核心函数默认：

```python
compute_mean_std_cross_steps=True
```

此时 `seen_pairs` 不会记录 `(uid, traj_uid)`，每个 step row 都会被加入 `id2score[uid]`：

- [`../verl-agent/verl/trainer/ppo/core_algos.py:113-172`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/ppo/core_algos.py#L113-L172)

若 trajectory \(\tau\) 有 \(L_\tau\) 个 steps，且忽略 invalid penalty，则其 episode reward 会在 group statistics 中重复 \(L_\tau\) 次：

$$
\mu_{\text{step}}
=
\frac{
\sum_\tau L_\tau R_\tau
}{
\sum_\tau L_\tau
}
$$

因此，verl-agent 的 GRPO advantage 是长度加权的 outcome advantage。

---

## 5. 核心数值例子

该例子应保留为后续论文中的 motivated example 或 running example。

### 5.1 设置

同一个 ALFWorld task rollout 8 次：

| 轨迹 | 结果 | 长度 |
|---|---|---:|
| \(\tau_1\) | 成功 | 10 |
| \(\tau_2\) | 成功 | 11 |
| \(\tau_3\) | 成功 | 14 |
| \(\tau_4\) | 失败 | 50 |
| \(\tau_5\) | 失败 | 50 |
| \(\tau_6\) | 失败 | 50 |
| \(\tau_7\) | 失败 | 50 |
| \(\tau_8\) | 失败 | 50 |

总计：

```text
成功 trajectory 数 = 3
失败 trajectory 数 = 5
成功 step rows      = 10 + 11 + 14 = 35
失败 step rows      = 5 × 50 = 250
总 step rows        = 285
```

假设：

- 不考虑 invalid-action penalty；
- slime reward 为成功 `1`、失败 `0`；
- verl-agent reward 为成功 `10`、失败 `0`；
- 两边均开启默认 std normalization；
- 使用 `torch.std` 对应的样本标准差。

### 5.2 slime 的 advantage

参与 mean/std 的 8 个 trajectory rewards：

```text
[1, 1, 1, 0, 0, 0, 0, 0]
```

均值：

$$
\mu_{\text{slime}}=\frac{3}{8}=0.375
$$

样本标准差：

$$
\sigma_{\text{slime}}
=
\sqrt{
\frac{
3(1-0.375)^2+5(0-0.375)^2
}{
8-1
}
}
=0.517549
$$

成功 trajectory：

$$
A^+_{\text{slime}}
=
\frac{1-0.375}{0.517549+10^{-6}}
=1.207612
$$

失败 trajectory：

$$
A^-_{\text{slime}}
=
\frac{0-0.375}{0.517549+10^{-6}}
=-0.724567
$$

每个 step 得到的 advantage：

| 轨迹 | step 数 | 每一步 advantage |
|---|---:|---:|
| 成功 \(\tau_1\) | 10 | `+1.207612` |
| 成功 \(\tau_2\) | 11 | `+1.207612` |
| 成功 \(\tau_3\) | 14 | `+1.207612` |
| 失败 \(\tau_4\) | 50 | `-0.724567` |
| 失败 \(\tau_5\) | 50 | `-0.724567` |
| 失败 \(\tau_6\) | 50 | `-0.724567` |
| 失败 \(\tau_7\) | 50 | `-0.724567` |
| 失败 \(\tau_8\) | 50 | `-0.724567` |

trajectory 层面的正负 scalar mass 保持平衡：

```text
3 × 1.207612 ≈ 5 × 0.724567 ≈ 3.62284
```

### 5.3 verl-agent 的 advantage：忽略 row padding

参与 mean/std 的是 285 个 step rows：

```text
35 个 reward=10
250 个 reward=0
```

均值：

$$
\mu_{\text{verl}}
=
\frac{35\times10}{285}
=1.228070
$$

样本标准差：

$$
\sigma_{\text{verl}}=3.287929
$$

成功 step：

$$
A^+_{\text{verl}}
=
\frac{10-1.228070}{3.287929+10^{-6}}
=2.667919
$$

失败 step：

$$
A^-_{\text{verl}}
=
\frac{0-1.228070}{3.287929+10^{-6}}
=-0.373509
$$

每个 step 得到的 advantage：

| 轨迹 | step 数 | 每一步 advantage |
|---|---:|---:|
| 成功 \(\tau_1\) | 10 | `+2.667919` |
| 成功 \(\tau_2\) | 11 | `+2.667919` |
| 成功 \(\tau_3\) | 14 | `+2.667919` |
| 失败 \(\tau_4\ldots\tau_8\) | 各 50 | `-0.373509` |

step-row 层面的正负 scalar mass同样近似平衡：

```text
35 × 2.667919 ≈ 250 × 0.373509 ≈ 93.3772
```

该结果解释了一个容易误判的现象：

- 失败 rows 数量很多，但每个失败 row 的负 advantage 较小；
- 成功 rows 数量很少，但每个成功 row 的正 advantage 很大。

原因是大量失败 rows 将 mean 拉向失败 reward，导致失败 reward 距离 mean 很近，而成功 reward 距离 mean 很远。

### 5.4 verl-agent 当前实现中的随机 row padding

verl-agent 在 reward 和 advantage 计算前调用 `adjust_batch()`：

- [`../verl-agent/verl/trainer/ppo/ray_trainer.py:1118-1236`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/ppo/ray_trainer.py#L1118-L1236)

`adjust_batch()` 默认随机复制真实 step rows，使 batch size 满足 rollout/ref/actor 的整除条件：

- [`../verl-agent/agent_system/multi_turn_rollout/utils.py:86-127`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/agent_system/multi_turn_rollout/utils.py#L86-L127)

`adjust_batch()` 作用于整个 flattened training batch，而不是分别对每个
`uid` prompt group 做 padding。因此不能把一个 285-row group 独立补成
384 rows，再乘以 prompt 数。

若当前完整 rollout batch 的 16 个 prompt groups 都与本例相同：

```text
原始 rows = 16 × 285 = 4560
row divisor = 32 × 4 = 128
padding 后 rows = 4608
全 batch 随机复制 48 rows
```

这 48 个 rows 从所有 prompt groups 中随机抽取。对某个具体 group \(g\)，设
其中被复制的成功和失败 rows 分别为 \(x_g^+\) 与 \(x_g^-\)，则该 group
真正参与 mean/std 的数量为：

```text
成功 rows = 35 + x_g^+
失败 rows = 250 + x_g^-
总 rows   = 285 + x_g^+ + x_g^-
```

因此，不同 `uid` groups 的 advantage 会因各自获得的随机复制 rows 略有变化。
平均而言，每个 group 约获得 3 个复制 rows，而且由于原始 rows 中
`250/285` 为失败 rows，大部分复制项通常来自失败轨迹。

例如某个 group 获得 3 个额外失败 rows：

```text
成功 rows = 35
失败 rows = 253
成功 advantage ≈ +2.683928
失败 advantage ≈ -0.371294
```

若获得 1 个成功 row 和 2 个失败 rows：

```text
成功 rows = 36
失败 rows = 252
成功 advantage ≈ +2.641153
失败 advantage ≈ -0.377308
```

随机 padding 不改变主要趋势，但会给 step-weighted GRPO 增加额外的、
按 prompt group 分布不均的 row-level 扰动。

---

## 6. Loss reduction 差异

### 6.1 slime：trajectory-level token mean

slime 在训练数据转换阶段，先计算同一 `group_id` 下所有 step samples 的总 loss-mask token 数：

- [`slime/ray/rollout.py:807-823`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/slime/ray/rollout.py#L807-L823)

每个 sibling sample 都携带相同的 trajectory denominator：

$$
T_\tau=\sum_s T_{\tau,s}
$$

reducer 对每个 step 的 token loss 除以该 trajectory 的总 denominator：

- [`slime/backends/megatron_utils/cp_utils.py:53-89`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/slime/backends/megatron_utils/cp_utils.py#L53-L89)

同一 trajectory 的所有 step contribution 相加后：

$$
L_\tau
=
\frac{
\sum_{s,t}\ell_{\tau,s,t}
}{
\sum_s T_{\tau,s}
}
$$

最终训练目标近似：

$$
L_{\text{slime}}
=
\frac{1}{G}
\sum_{\tau=1}^{G}
L_\tau
$$

其中 \(G\) 为当前 optimizer step 中的 trajectory group 数。

仓库测试直接覆盖：

- 多个 sibling samples 合并为一个 rollout mean；
- siblings 跨 microbatch 后仍恢复同一个完整 rollout mean。

测试证据：

- [`tests/test_cp_utils.py:1-14`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/tests/test_cp_utils.py#L1-L14)
- [`tests/test_cp_utils.py:65-100`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/tests/test_cp_utils.py#L65-L100)
- [`tests/test_metric_report.py:55-84`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/tests/test_metric_report.py#L55-L84)

### 6.2 verl-agent：step-row/token mini-batch

verl-agent 默认：

```yaml
loss_agg_mode: token-mean
```

- [`../verl-agent/verl/trainer/config/ppo_trainer.yaml:43-68`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/config/ppo_trainer.yaml#L43-L68)

每个当前 mini/microbatch 的 loss 为：

$$
L_{\text{mb}}
=
\frac{
\sum_{i,t}m_{i,t}\ell_{i,t}
}{
\sum_{i,t}m_{i,t}
}
$$

代码：

- [`../verl-agent/verl/trainer/ppo/core_algos.py:395-417`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/ppo/core_algos.py#L395-L417)
- [`../verl-agent/verl/workers/actor/dp_actor.py:398-433`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/workers/actor/dp_actor.py#L398-L433)

actor 更新时不再使用 `traj_uid` 构造 trajectory denominator。因此：

- 长 trajectory 产生更多 rows；
- 长 trajectory 通常产生更多 response tokens；
- 同一 trajectory 的 steps 可能进入不同 mini-batches 和 optimizer updates；
- 不存在与 slime 等价的 trajectory-level reducer。

---

## 7. Optimizer update schedule 差异

### 7.1 slime：按 trajectory group 计数

slime scheduler 明确规定：

> `global_batch_size` 是每个训练 step 的 group 数，而不是 training sample rows 数。

代码：

- [`slime/utils/dp_schedule.py:67-95`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/slime/utils/dp_schedule.py#L67-L95)
- group 聚合与 step 划分：[`slime/utils/dp_schedule.py:112-135`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/slime/utils/dp_schedule.py#L112-L135)

当前 launcher：

```text
rollout_batch_size    = 16 prompts
n_samples_per_prompt = 8 trajectories
global_batch_size     = 128 groups
```

- [`examples/alfworld/run_qwen2.5_3B_instruct_grpo.sh:66-95`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/run_qwen2.5_3B_instruct_grpo.sh#L66-L95)

因此：

$$
N_{\text{updates}}
=
\frac{
16\times8
}{
128
}
=1
$$

所有 step rows 仍然参与 forward/backward，但会被动态打包成多个 microbatches，梯度全部累积完成后才调用一次 `optimizer.step()`：

- forward/backward：[`slime/backends/megatron_utils/model.py:549-560`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/slime/backends/megatron_utils/model.py#L549-L560)
- optimizer update：[`slime/backends/megatron_utils/model.py:582-589`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/slime/backends/megatron_utils/model.py#L582-L589)

### 7.2 verl-agent：按 flattened step rows 切 mini-batch

verl-agent actor 直接执行：

```python
dataloader = batch.split(ppo_mini_batch_size)
```

- [`../verl-agent/verl/workers/actor/dp_actor.py:332-340`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/workers/actor/dp_actor.py#L332-L340)

每个 mini-batch 都经历：

```text
zero_grad
-> microbatch forward/backward
-> optimizer.step
```

- [`../verl-agent/verl/workers/actor/dp_actor.py:343-443`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/workers/actor/dp_actor.py#L343-L443)
- 实际 `actor_optimizer.step()`：[`../verl-agent/verl/workers/actor/dp_actor.py:234-250`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/workers/actor/dp_actor.py#L234-L250)

当前全局 `ppo_mini_batch_size=256`：

- [`../verl-agent/examples/grpo_trainer/run_alfworld.sh:42-49`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/examples/grpo_trainer/run_alfworld.sh#L42-L49)

当前普通 GRPO 路径还满足：

- `actor_rollout_ref.rollout.n=1`；
- FSDP data parallel size 为 4，因此 worker 侧将全局 mini-batch 256 归一化为每卡 64 rows；
- `ppo_micro_batch_size_per_gpu=32`，每个完整 local mini-batch 使用 2 个 microbatches 做梯度累积；
- `ppo_epochs=1`。

代码：

- [`../verl-agent/verl/workers/fsdp_workers.py:148-160`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/workers/fsdp_workers.py#L148-L160)
- [`../verl-agent/verl/trainer/config/ppo_trainer.yaml:43-68`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/config/ppo_trainer.yaml#L43-L68)
- [`../verl-agent/examples/grpo_trainer/run_alfworld.sh:79-80`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/examples/grpo_trainer/run_alfworld.sh#L79-L80)

若完整 rollout batch 的 16 个 prompt groups 都与核心例子相同：

```text
原始 rows = 16 × 285 = 4560
padding 后 = 4608
Adam updates = 4608 / 256 = 18
```

若训练早期所有 128 条 trajectories 都运行到 50 步：

```text
rows = 128 × 50 = 6400
Adam updates = 6400 / 256 = 25
```

需要额外注意：

- 多次 Adam updates 发生在一个外层 trainer step 内；
- actor LR scheduler 在整个 `update_actor()` 返回后只推进一次。

代码：

- [`../verl-agent/verl/workers/fsdp_workers.py:615-626`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/workers/fsdp_workers.py#L615-L626)

这意味着每个 tracker/global step 内的实际优化预算会随 trajectory 长度改变。

---

## 8. 对学习曲线的机制解释

以下内容是与当前代码和实验现象一致的**研究假设**，尚需消融实验确认。

### 8.1 verl-agent 为什么可能前期更快

#### 作用路径 A（主要关联 M2）：稀有成功 step 获得更大的正向 advantage

在核心例子中：

```text
slime 成功 advantage      = +1.207612
verl-agent 成功 advantage = +2.667919
```

早期只要偶然产生一条成功轨迹，其有限数量的成功 steps 就会得到非常强的正向更新，快速强化最先发现的成功行为。

#### 作用路径 B（主要关联 M4）：每个外层 step 内执行更多 Adam updates

在失败轨迹较长的训练前期，verl-agent 可能在一个 tracker/global step 内执行十几到二十多次 Adam updates，而 slime 当前 launcher 只执行一次。

因此如果横轴使用外层 trainer/global step，二者并不是等优化预算比较。

#### 作用路径 C（关联 M3/M4）：更小的 mini-batch 和更频繁的参数变化

verl-agent 的模型参数会在不同 step-row mini-batches 之间发生变化，具有：

- 更高的更新频率；
- 更高的顺序敏感性；
- 更强的早期拟合能力；
- 也可能有更高的梯度方差。

slime 则在一个较大的 trajectory-group batch 上先累积梯度，再统一更新，通常更加平滑和保守。

### 8.2 verl-agent 为什么可能更早达到平台

#### 作用路径 D（关联 M2/M3/M4）：长失败轨迹带来更广泛的负向状态覆盖

一条 50-step 失败轨迹的每个 step 都获得负 advantage。由于使用 outcome reward，这些 steps 中可能同时包含：

- 真正导致失败的错误动作；
- 合理但未能最终完成任务的中间动作；
- 与成功轨迹共享的正确前缀；
- 仅仅因为后续探索失败而被回溯性惩罚的动作。

verl-agent 不使用 trajectory reducer，因此这些 steps 会被分散到更多 mini-batches 和参数更新中。

这可能导致：

- 合理中间动作被过度压低；
- 策略变得保守；
- 探索空间收缩；
- 模型过度拟合最先出现的少量成功模式；
- 最终性能上限受限。

#### 作用路径 E（关联 M2/M3）：优化目标是 step-weighted surrogate

最终任务指标通常是 episode success：

$$
\frac{\text{成功 trajectory 数}}{\text{总 trajectory 数}}
$$

而 verl-agent 的普通 GRPO advantage 和训练 batch 更接近按 step rows 加权：

$$
\frac{\text{成功 step rows}}{\text{总 step rows}}
$$

当成功和失败轨迹长度分布不同，这两个目标不等价。

slime 按 trajectory 计算 advantage，并按 trajectory group 聚合 loss，因此其训练目标与 episode-level success 更一致。

#### 作用路径 F（主要关联 M4）：每个 global step 的实际更新次数随性能提高而下降

训练前期失败轨迹长：

```text
平均 50 steps -> 约 25 Adam updates/global step
```

训练后期成功轨迹增多、平均长度下降：

```text
平均约 12 steps -> 约 6 Adam updates/global step
```

因此 verl-agent 的曲线可能自然呈现：

```text
前期更新密集 -> 提升快
后期更新变少 -> 增长放缓
```

slime 的 optimizer update 数由 trajectory group 数决定，不随 trajectory 长度直接变化。

### 8.3 slime 为什么可能前期慢但上限高

#### 作用路径 G（主要关联 M2）：缺少极端正向 advantage spike

trajectory-level normalization 不会因为成功 trajectory 较短就将其视为极少数 step rows，因此成功 advantage 通常没有 step-weighted normalization 那样极端。

更新更加保守，可能导致较慢的初始提升。

#### 作用路径 H（关联 M3/M4）：trajectory 总权重不随长度重复

一条 50-step 失败 trajectory 和一条 10-step 成功 trajectory 在外层都只贡献一个 trajectory mean。

这限制了：

- 长失败轨迹的重复负权重；
- outcome reward 对合理中间动作的整体误伤；
- 由轨迹长度产生的非平稳优化预算。

#### 作用路径 I（关联 M2/M3）：更接近最终 episode-level 指标

每条 trajectory 等权的训练目标，与 ALFWorld success rate 的统计单位一致，更可能在长期优化中减少 surrogate-objective mismatch。

---

## 9. 动态后过滤的作用

当前 slime ALFWorld GRPO launcher 启用：

```text
--dynamic-sampling-filter-path generate_with_alfworld.check_episode_reward_nonzero_std
```

- [`examples/alfworld/run_qwen2.5_3B_instruct_grpo.sh:82-95`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/run_qwen2.5_3B_instruct_grpo.sh#L82-L95)

过滤器按同一 prompt 下的唯一 trajectory rewards 计算 std：

- [`examples/alfworld/generate_with_alfworld.py:404-437`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/generate_with_alfworld.py#L404-L437)

reward 零方差 group 会被丢弃，例如：

- 8 条全部失败且最终 reward 相同；
- 8 条全部成功且最终 reward 相同；
- 所有 trajectory 的最终 reward 完全相同。

### 9.1 前期影响

早期大量 prompt groups 可能全部失败且 final reward 零方差，需要额外 rollout 才能收集到 reward 有差异的 group。它通常表现为成功/失败混合，但如果 invalid penalty 已经造成失败 trajectory 之间的 final reward 差异，全失败 group 也可能被保留。

因此：

- wall-clock 速度可能下降；
- 环境交互和推理成本增加；
- 但每个真正进入训练的 group 都具有非零相对优势信号。

动态过滤不一定会降低“每个 optimizer step”的学习效率，但可能降低“每单位 wall-clock”的训练速度。

### 9.2 后期影响

当模型在简单任务上达到稳定成功且 final reward 一致时，全成功 groups 被过滤，训练继续聚焦仍然存在 trajectory reward 差异的 prompt。

这类似：

- 在线 hard-example mining；
- 自适应课程学习；
- 决策边界采样；
- 非零优势信号保持。

它是 slime 后期上限更高的另一个重要候选原因。

---

## 10. 惩罚项：trajectory-level collective penalty 与 raw-score-local penalty

slime 和 verl-agent 都将 invalid-action penalty 设为满分的 1%，但二者的
惩罚对象、归一化单位、credit assignment 和过滤位置均不相同。因此，
`0.01` 与 `0.1` 不能只被理解为 reward scale 不同后的同一个超参数。

### 10.1 slime：轨迹级连带惩罚

设：

- \(Y_\tau\in\{0,1\}\)：trajectory 是否成功；
- \(K_\tau\)：trajectory 中 invalid actions 的总数；
- \(\lambda=0.01\)。

slime 的 final trajectory reward 为：

$$
\tilde R_\tau=Y_\tau-\lambda K_\tau
$$

代码路径：

1. 每个 invalid step 增加 `invalid_action_count`：
   - [`examples/alfworld/batched_rollout.py:269-293`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/batched_rollout.py#L269-L293)
2. trajectory 结束后统一计算 final reward：
   - [`examples/alfworld/batched_rollout.py:381-397`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/batched_rollout.py#L381-L397)
3. 同一个 final reward 广播到 trajectory 的所有 step samples：
   - [`examples/alfworld/batched_rollout.py:424-431`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/batched_rollout.py#L424-L431)
4. 按唯一 trajectory final rewards 做 mean/std：
   - [`examples/alfworld/generate_with_alfworld.py:440-480`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/generate_with_alfworld.py#L440-L480)
5. scalar advantage 广播到所有 response tokens：
   - [`slime/utils/ppo_utils.py:201-208`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/slime/utils/ppo_utils.py#L201-L208)
6. loss 以 trajectory 总 token 数为 denominator：
   - [`slime/ray/rollout.py:807-823`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/slime/ray/rollout.py#L807-L823)
   - [`slime/backends/megatron_utils/cp_utils.py:53-136`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/slime/backends/megatron_utils/cp_utils.py#L53-L136)

因此，一个 invalid action 会降低整条 trajectory 的相对评价。它不只惩罚
发生 invalid 的那个 step，而会通过共享的 trajectory advantage 影响该
trajectory 的所有 step samples。

该机制可直观理解为一种 **trajectory-level collective penalty**：

> 模型不仅需要某一个动作合法，还需要整条 trajectory 尽可能保持低 invalid count。

但论文中不应直接称其必然鼓励“全局最优探索”。它更准确地鼓励：

- 全轨迹行为一致性；
- 更低的 invalid 总数；
- 更少的无效或不可执行动作；
- 在相同 task outcome 下选择更干净的 trajectory。

它也可能抑制高风险探索，因为一次 invalid 会降低整条 trajectory，包括
invalid 之前的合理 steps。

### 10.2 verl-agent：raw-score-local、advantage-group-coupled 惩罚

verl-agent 的 ALFWorld task reward scale 为成功 `10`、失败 `0`。设
\(I_{\tau,s}\in\{0,1\}\) 表示第 \(s\) 个 step 是否被判定为 invalid，
惩罚系数为 \(0.1\)。每个 step row 的 score 为：

$$
\tilde r_{\tau,s}=10Y_\tau-0.1I_{\tau,s}
$$

代码路径：

1. 完整 episode reward 被复制到每个 active step row：
   - [`../verl-agent/agent_system/multi_turn_rollout/rollout_loop.py:261-277`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/agent_system/multi_turn_rollout/rollout_loop.py#L261-L277)
2. reward manager 将 episode reward 写入每个 step response 的最后一个有效 token：
   - [`../verl-agent/agent_system/reward_manager/episode.py:39-79`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/agent_system/reward_manager/episode.py#L39-L79)
3. invalid penalty 只从对应 invalid step row 的最后 token 减去：
   - [`../verl-agent/verl/trainer/ppo/ray_trainer.py:200-224`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/ppo/ray_trainer.py#L200-L224)
4. 所有 step rows 参与 `uid` group 的 mean/std：
   - [`../verl-agent/verl/trainer/ppo/core_algos.py:113-172`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/ppo/core_algos.py#L113-L172)
5. 得到的 step scalar advantage 广播到该 step 的全部 response tokens：
   - [`../verl-agent/verl/trainer/ppo/core_algos.py:167-174`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/ppo/core_algos.py#L167-L174)
6. actor 使用 token-mean：
   - [`../verl-agent/verl/trainer/ppo/core_algos.py:395-417`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/ppo/core_algos.py#L395-L417)

因此，verl-agent 的 penalty 注入位置是 **raw-score-local**：

- invalid step 的 raw score 得到局部负向修正；
- 同一 trajectory 的其他 valid steps 不直接扣除该 `0.1`；
- 同一 trajectory 内不同 steps 可能获得不同 scalar advantages。

但最终 advantage 并不是严格局部的。当前 invalid row 被减去 `0.1` 后，
`uid` group 的 mean/std 会重新计算，因此：

- invalid row 的 score 会改变 group mean/std；
- 同一 `uid` 下所有其他 rows 的 normalized advantages 都会随之改变；
- 最终语义应描述为 **raw-score-local、advantage-group-coupled**。

如果只为了理解一条 trajectory 的平均原始 score，则：

$$
\frac{1}{L_\tau}\sum_s\tilde r_{\tau,s}
=
10Y_\tau-0.1\frac{K_\tau}{L_\tau}
$$

该式不是 verl-agent 的实际 trajectory reducer，但揭示了 raw-score-local 惩罚的聚合
语义更接近 invalid rate \(K_\tau/L_\tau\)，而 slime 的 final reward
直接依赖 invalid count \(K_\tau\)。

### 10.3 相同的 1% 为何产生不同训练信号

将 slime 的 reward 同样放大到 `0/10`：

```text
slime trajectory score: 10Y - 0.1K
verl step score:        10Y - 0.1I_s
```

例如一条 50-step 失败 trajectory 只有一个 invalid：

```text
slime trajectory score：-0.1
verl 50 个 step scores：1 个 -0.1，49 个 0
verl trajectory 内平均：-0.002
```

但经过 GRPO mean/std 后，绝对值为 1% 的原始惩罚可能被标准化为
\(O(1)\) advantage。特别是在所有 trajectories 的 task outcome 相同
时，penalty variance 会成为整个 GRPO advantage 的来源。

### 10.4 Penalty-only worked example

考虑 8 条 trajectories 全部失败、长度均为 50，但 invalid counts 为：

```text
K = [0, 1, 2, 3, 4, 5, 6, 7]
```

总 invalid step rows 为：

```text
0 + 1 + 2 + 3 + 4 + 5 + 6 + 7 = 28
```

以下 advantage 数值用于隔离 penalty granularity，暂不计入 verl-agent
`adjust_batch()` 对完整 training batch 的随机 row padding；实际在线数值会
随该 group 获得的复制 rows 略有变化。

#### slime

trajectory final rewards：

```text
[0, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07]
```

均值与样本标准差：

```text
mean = -0.035
std  = 0.0244949
```

trajectory advantages：

| invalid count \(K\) | trajectory advantage | 该 trajectory 每个 step 的 advantage |
|---:|---:|---:|
| 0 | `+1.428811` | `+1.428811` |
| 1 | `+1.020579` | `+1.020579` |
| 2 | `+0.612347` | `+0.612347` |
| 3 | `+0.204116` | `+0.204116` |
| 4 | `-0.204116` | `-0.204116` |
| 5 | `-0.612347` | `-0.612347` |
| 6 | `-1.020579` | `-1.020579` |
| 7 | `-1.428811` | `-1.428811` |

虽然所有 trajectories 都失败，但 slime 会学习：

> invalid 总数更少的失败 trajectory 优于 invalid 总数更多的失败 trajectory。

这一排序作用于整条 trajectory 的所有 steps。

#### verl-agent

共有：

```text
400 step rows
28 invalid rows，score=-0.1
372 valid rows，score=0
```

step-row mean/std：

```text
mean = -0.007
std  = 0.0255467
```

advantages：

```text
valid step advantage   = +0.273998
invalid step advantage = -3.640256
```

正负 scalar mass 近似平衡：

```text
372 × 0.273998 ≈ 28 × 3.640256 ≈ 101.927
```

verl-agent 在该例子中学习的是：

> 强烈惩罚具体 invalid step，同时轻微奖励其余格式有效的 steps，即使整条 trajectory 最终失败。

这是一种 raw-score 层面的局部注入；经过 group normalization 后，最终
advantage 仍然是组耦合的。它可能强化失败 trajectory 中虽然格式有效、
却对任务无效的动作。

### 10.5 Invalid-action 判定边界

slime 的 `parse_action()` 同时检查：

- `<action>...</action>`；
- `<think>...</think>`；
- 是否包含中文；
- action 是否属于当前 `admissible_actions`。

代码：

- [`examples/alfworld/prompts.py:117-153`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/prompts.py#L117-L153)

verl-agent 的 `alfworld_projection()` 虽然接收 `action_pools`，但当前实现
没有执行 action-pool membership 检查，主要检查：

- `<action>` 标签；
- `<think>` 标签；
- 是否包含中文。

代码：

- [`../verl-agent/agent_system/environments/env_package/alfworld/projection.py:19-62`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/agent_system/environments/env_package/alfworld/projection.py#L19-L62)

此外，两个 parser 的格式边界也不完全一致：

- slime 要求提取出的 action 非空；
- verl-agent 当前在找到 `<action></action>` 标签后会先将 `valids[i]`
  设为 1，没有单独检查提取内容是否为空；
- slime 的 `<think>` 正则为大小写不敏感；
- verl-agent 在原始字符串中使用小写 `<think>` / `</think>` 做大小写敏感查找。

因此，两边的 penalty 不仅聚合层级不同，所惩罚的 invalid 语义也不同：

- slime 更接近环境语义有效性；
- verl-agent 更接近输出格式有效性。

这可能是最终 success 上限差异的重要候选原因。

### 10.6 Penalty-aware dynamic filtering

slime 的 dynamic filter 读取的是已包含 trajectory penalty 的 final reward：

$$
\tilde R_\tau=Y_\tau-0.01K_\tau
$$

因此：

- 全部失败但 \(K_\tau\) 不同：group reward 有方差，可能被保留；
- 全部成功但 \(K_\tau\) 不同：group reward 有方差，可能被保留；
- 只有 final rewards 完全相同时，group 才会因零方差被丢弃。

这形成了如下闭环：

```text
trajectory-level penalty
-> 为相同 task outcome 的 trajectories 制造质量差异
-> dynamic filter 保留 penalty-aware nonzero-variance groups
-> GRPO 将 0.01 级 raw reward 差异标准化为 O(1) advantages
-> trajectory reducer 优化整条轨迹的 invalid count
```

因此，当前过滤器不仅是 success/failure variance filter，也可视为：

> **penalty-aware trajectory-quality filter**

候选收益：

1. 在全部失败阶段也可从 invalid count 获得训练信号；
2. 在普遍成功后仍可优化动作合法性和轨迹整洁度；
3. 保留相同 outcome、不同 trajectory quality 的困难 groups；
4. 让动态过滤与 trajectory-level reward shaping 形成在线课程。

候选风险：

1. 训练可能过度优化“少 invalid”，而非真实 task progress；
2. 高风险但潜在有价值的探索可能被整条 trajectory 惩罚；
3. 一次 invalid 会降低该 trajectory 中所有合理 steps 的 advantage；
4. filter 会改变 accepted prompt/group 的分布。

### 10.7 对学习曲线和最终上限的惩罚项假设

以下仍是待消融验证的机制假设：

1. slime 的 admissible-aware penalty 比 verl-agent 的 format-oriented penalty
   更接近 ALFWorld 的真实动作质量，因此可能提高最终 success 上限。
2. trajectory-count penalty 不会在长轨迹中按 \(K/L\) 稀释，可能更有效地
   约束全轨迹行为一致性。
3. penalty-aware dynamic filtering 使 task outcome 相同但 invalid count
   不同的 groups 继续参与训练，可能维持后期有效信号。
4. verl-agent 的 raw-score-local penalty 能更直接地区分 invalid row，但
   group normalization 会同步改变其他 rows 的 advantage；在 step-row
   weighting 和多次 mini-batch updates 下，它也可能过度强化“格式有效但
   任务无效”的失败 steps。
5. slime 的连带惩罚可能提升全局一致性，也可能降低探索性；其净收益必须
   通过 penalty granularity 与 validity checker 的独立消融确认。

---

## 11. 其他必须控制的差异

在将曲线差异归因于 trajectory weighting 前，至少需要控制以下变量。

| 维度 | slime | verl-agent |
|---|---|---|
| Invalid penalty | trajectory-level `0.01 × invalid_count` | invalid step row 上减 `0.1` |
| Action validity | 检查 admissible membership | 不检查 action pool membership |
| Dynamic filter | 当前启用 | 默认关闭 |
| Clip range | `[0.8, 1.28]` | `[0.8, 1.2]` |
| Dual clip | 默认关闭 | `c=3.0` |
| Entropy coefficient | `0` | `0.001` |
| Optimizer | Megatron `adam`, beta2 `.98`, wd `.1` | `torch.optim.AdamW`, beta2 `.999`, wd `.01` |
| Prompt history | 4 | 2 |
| Validation | seen + unseen，evaluation 返回 trajectory summary sample | 默认 seen/in-distribution；`success_rate` 在展平前按 episode 计算，ALFWorld `test_score` 按 step rows 汇总 |
| Rollout backend | SGLang | 默认 vLLM，可切换 |

相关代码：

- slime launcher：[`examples/alfworld/run_qwen2.5_3B_instruct_grpo.sh:35-138`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/run_qwen2.5_3B_instruct_grpo.sh#L35-L138)
- verl-agent launcher：[`../verl-agent/examples/grpo_trainer/run_alfworld.sh:1-84`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/examples/grpo_trainer/run_alfworld.sh#L1-L84)
- verl defaults：[`../verl-agent/verl/trainer/config/ppo_trainer.yaml:43-68`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/config/ppo_trainer.yaml#L43-L68); [`../verl-agent/verl/trainer/config/ppo_trainer.yaml:234-257`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/config/ppo_trainer.yaml#L234-L257); [`../verl-agent/verl/trainer/config/ppo_trainer.yaml:292-304`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/config/ppo_trainer.yaml#L292-L304)
- verl validation aggregation：[`../verl-agent/verl/trainer/ppo/ray_trainer.py:787-816`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/ppo/ray_trainer.py#L787-L816)

---

## 12. 可证伪的消融实验

要把当前机制观察发展为论文结论，需要分离 advantage、loss reducer、optimizer schedule 和 dynamic filter。

### 12.1 当前机制探索：verl-agent V0→V5 增量实验链

当前阶段优先采用**在 verl-agent native baseline 上逐项增加机制**的探索
方式，而不是立即追求完整 factorial。

指定对照组（runtime manifest 待冻结）：

V0 source = [`../verl-agent/examples/grpo_trainer/run_alfworld.sh:1-84`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/examples/grpo_trainer/run_alfworld.sh#L1-L84)

V0 当前正在运行，不需要重新启动，但在 V1 开始前必须将本次 V0 的实际运行
清单冻结下来。脚本接受首个 `ENGINE` 参数和末尾 `$@` overrides，仅固定脚本
路径不足以完整复现运行状态。需要记录：

| V0 运行字段 | 要求 |
|---|---|
| verl-agent commit | `287d52e088675d7d5adb0bec621e1fb53b40d28b` |
| launcher | [`examples/grpo_trainer/run_alfworld.sh:1-84`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/examples/grpo_trainer/run_alfworld.sh#L1-L84) |
| ENGINE | 记录实际值，例如 `vllm` |
| `$@` overrides | 记录完整 Hydra overrides；若无则明确写 `none` |
| 关键环境变量 | `ALFWORLD_DATA`、`PATH`、`LD_PRELOAD` 等 |
| model path | 记录实际 checkpoint 路径 |
| SwanLab run ID | 记录当前 baseline run ID |
| 启动时间与服务器/GPU | 记录运行环境 |

V0 并不是“没有机制”，而是使用五个机制在 verl-agent 中的 native 版本：

```text
M1 raw-score-local penalty
M2 step-row advantage
M3 token-mean reducer
M4 row-based scheduler
M5 dynamic filter off
```

后续五个版本会逐项将 native 机制替换为 trajectory-aware 版本，或启用
slime 机制：

| ID | 相比上一个版本的替换/启用 | 当前机制状态 | 主要问题 |
|---|---|---|---|
| V0 | native baseline | M1/M2/M3/M4 均为 native；M5 off | 原始 verl-agent GRPO 曲线 |
| V1 | 将 M4 替换为 group scheduler | trajectory M4 | 多次 optimizer updates 是否解释前期快速收敛？ |
| V2 | 将 M3 替换为 trajectory reducer | trajectory M3/M4 | 长短 trajectory 外层等权是否改变性能？ |
| V3 | 将 M2 替换为 trajectory advantage | trajectory M2/M3/M4 | trajectory mean/std 是否改善学习信号？ |
| V4 | 将 M1 替换为 trajectory penalty | trajectory M1/M2/M3/M4 | 连带惩罚是否改善全局行为质量？ |
| V5 | 启用 M5 dynamic filter | trajectory M1～M5 | penalty-aware filtering 是否提高后期上限？ |

该顺序满足两个依赖关系：

1. trajectory reducer 放在 group scheduler 之后，避免同一 trajectory 的
   steps 已经被拆进不同 optimizer updates；
2. dynamic filter 放在 trajectory penalty 之后，使 V5 真正测试
   post-penalty reward filtering。

五个新增实验的直接比较为：

```text
V1 - V0 -> 将 M4 从 row-based 替换为 group scheduler 的增量作用
V2 - V1 -> 将 M3 从 token mean 替换为 trajectory reducer 的增量作用
V3 - V2 -> 将 M2 从 step-row 替换为 trajectory advantage 的增量作用
V4 - V3 -> 将 M1 从 raw-score-local 替换为 trajectory penalty 的增量作用
V5 - V4 -> 启用 M5 dynamic filter 的增量作用
```

这是一条顺序依赖的机制发现链，而不是最终的单变量因果证明。若某一步出现
显著跃迁或退化，再围绕该机制补反向删除、交互组合或独立分支实验。

当前 ALFWorld 阶段共形成：

```text
V0 已有 baseline
+ V1、V2、V3、V4、V5 五个新增实验
= 6 个版本
```

### 12.2 后续受控核心 2×2×2 实验

三个因素：

#### 因素 A：Advantage normalization unit

- Step-row normalization
- Trajectory-level normalization

#### 因素 B：Loss aggregation unit

- Step/token mean
- Trajectory-level token mean

#### 因素 C：Dynamic filtering

- Off
- Trajectory-reward nonzero-std filter

总计 8 个实验设置。所有其他配置应保持一致。

### 12.3 Penalty、validity 与 filter 的分阶段消融

惩罚机制不适合直接与前述 2×2×2 全排列，因为会迅速形成过大的实验矩阵。
建议先固定 advantage、reducer 和 optimizer schedule，再分阶段比较。

#### Penalty granularity

- No penalty
- Step-local indicator：\(-\lambda I_{\tau,s}\)
- Trajectory invalid count：\(-\lambda K_\tau\)
- Trajectory invalid rate：\(-\lambda K_\tau/L_\tau\)

#### Validity checker

为避免将 parser 细节与 admissible membership 混在一起，正式消融必须先
定义一个双方共用的 canonical parser：

- `<think>` / `<action>` 标签采用相同的大小写规则；
- action 内容必须非空；
- 中文字符规则保持一致；
- action 提取和 fallback 规则保持一致。

随后只切换一个变量：

- Canonical format-only：共用 parser，但不检查 action-pool membership；
- Canonical admissible-aware：使用同一 parser，并增加
  `action in admissible_actions`。

双方各自的 native parser 结果可以作为额外复现实验报告，但不能用于
`format-only vs admissible-aware` 的单变量因果比较。

#### Filter reward source

- Filter off
- Task-reward filter：只使用成功/失败 task reward
- Post-penalty filter：使用已经包含 invalid penalty 的 final reward

#### 建议的最小 penalty 实验梯度

| ID | Penalty | Validity checker | Filter |
|---|---|---|---|
| P0 | none | canonical admissible-aware | off |
| P1 | raw-score-local | canonical format-only | off |
| P2 | raw-score-local | canonical admissible-aware | off |
| P3 | trajectory count | canonical admissible-aware | off |
| P4 | trajectory rate | canonical admissible-aware | off |
| P5 | trajectory count | canonical admissible-aware | task-reward filter |
| P6 | trajectory count | canonical admissible-aware | post-penalty filter |

#### Penalty strength calibration

保持“单次 invalid 为满分的 1%”只能复现实验默认，不能保证不同 penalty
granularity 具有相同的有效训练强度。例如：

- trajectory count 的累计惩罚与 \(K_\tau\) 成正比；
- trajectory rate 与 \(K_\tau/L_\tau\) 成正比；
- raw-score-local penalty 还会经过 step-row mean/std 和 token/mini-batch
  weighting。

因此，正式因果消融应同时报告两组实验：

1. **Native-ratio setting**：保持双方原生的满分 1% 定义，用于复现实验；
2. **Calibrated-strength setting**：在同一批离线 trajectories 上分别调节
   \(\lambda\)，至少匹配以下一项，并报告其他项：
   - 每条 trajectory 的平均绝对 raw penalty；
   - 每个 prompt group 的 penalty-induced reward variance；
   - penalty-only normalized advantage 的正负 scalar/token mass；
   - 每个 optimizer step 的 penalty-gradient norm。

建议以匹配“每条 trajectory 平均绝对 penalty”为主校准目标，再围绕校准值
执行 \(\{0.25,0.5,1,2,4\}\times\lambda\) sweep。只有在 effective penalty
strength 对齐后，`P2 vs P3` 和 `P3 vs P4` 才能主要归因于 penalty
granularity，而不是惩罚总量。

该实验顺序可分别回答：

1. `P1 vs P2`：最终上限差异是否来自 admissible membership 判定？
2. `P2 vs P3`：raw-score-local 与 trajectory-level collective penalty 谁更有效？
3. `P3 vs P4`：有效信号来自 invalid 总数还是 invalid 比例？
4. `P3 vs P5`：普通 task-reward filtering 是否有独立贡献？
5. `P5 vs P6`：penalty-aware filtering 是否是后期上限提升的关键？
6. `P0 vs P3`：trajectory penalty 的净收益和探索代价分别是多少？

Native-ratio 与 calibrated-strength 两组结果都应保留：前者回答真实默认配置
的差异，后者回答 penalty granularity 的独立因果作用。

### 12.4 Optimizer-budget 对齐

至少提供三种横轴：

1. 外层 trainer/global step
2. 实际 `optimizer.step()` 次数
3. 生成的环境 step rows 或 response tokens

如果只按 trainer global step 比较，verl-agent 每个 step 内可能执行更多 Adam updates，结论不公平。

建议额外做：

- 固定每个外层 step 的实际 optimizer updates；
- 固定总 optimizer updates；
- 固定总环境交互 steps；
- 固定总生成 token；
- 固定 wall-clock。

### 12.5 建议记录的指标

#### 轨迹级指标

- success rate
- success trajectory length
- failure trajectory length
- success/failure length ratio
- invalid action count
- truncated trajectory ratio

#### Advantage 指标

- trajectory success fraction \(p_{\text{traj}}\)
- step-row success fraction \(p_{\text{step}}\)
- mean positive advantage
- mean negative advantage
- positive/negative advantage row count
- positive/negative advantage token mass
- advantage 与 trajectory length 的相关系数

#### 优化指标

- actual optimizer steps per trainer step
- rows per optimizer step
- tokens per optimizer step
- gradient norm
- positive-only gradient norm
- negative-only gradient norm
- success/failure gradient cosine similarity
- clip fraction
- KL loss
- entropy

#### Dynamic-filter 指标

- dropped zero-variance all-failure group 数
- dropped zero-variance all-success group 数
- 每个 accepted batch 的额外 rollout 次数
- accepted prompt 的难度分布
- filter 前后 task-type 分布

#### Penalty 与 validity 指标

- invalid-format rate
- non-admissible-action rate
- invalid count \(K_\tau\)
- invalid rate \(K_\tau/L_\tau\)
- penalty-only nonzero-variance group 数
- 全失败但被 post-penalty filter 保留的 group 数
- 全成功但被 post-penalty filter 保留的 group 数
- valid-step 与 invalid-step advantage 分布
- penalty 前后 trajectory 排序变化
- penalty signal 与 task success signal 的梯度余弦

### 12.6 离线反事实重放

对同一批已收集 trajectory，同时离线计算：

1. step-row GRPO advantages；
2. trajectory-level GRPO advantages；
3. token-mean loss；
4. trajectory-mean loss；
5. no-penalty reward；
6. raw-score-local penalty；
7. trajectory-count penalty；
8. trajectory-rate penalty。

随后比较：

- advantage 排序；
- 每条 trajectory 的总梯度贡献；
- gradient norm；
- gradient cosine similarity；
- 与在线更新方向的差异。

该实验可以在不重新 rollout 的情况下直接证明两种算法对同一数据产生了不同优化信号。

---

## 13. 潜在论文定位

### 13.1 核心论点

一个谨慎但有潜力的论文主张是：

> In multi-step agentic GRPO, flattening trajectories into step-level samples creates a hidden mismatch between the episode-level objective and the optimization unit. When trajectory length correlates with outcome, reward-shaping granularity, filter placement, step-level normalization, and token-level mini-batching jointly introduce trajectory-length-dependent advantages and update budgets. Trajectory-aware normalization, aggregation, and penalty-aware filtering reduce this specific step-duplication pathway and may improve long-horizon performance.

中文：

> 在多步 agentic GRPO 中，将 trajectory 展平为 step samples 会在 episode-level 任务目标与优化单位之间产生隐藏错配。当轨迹长度与任务结果相关时，reward shaping 粒度、filter 位置、step-level normalization 和 token-level mini-batching 会共同引入依赖轨迹长度的 advantage 与更新预算。trajectory-aware normalization、aggregation 和 penalty-aware filtering 可以减少由 step-row 重复计权形成的特定偏置路径，并可能提高长期性能上限。

### 13.2 可能的贡献点

#### Contribution 1：识别隐藏的 trajectory-length bias

揭示多步 GRPO 中，即使 reward 本身是 episode-level，step-row flattening 仍可能通过 mean/std、loss reduction 和 optimizer scheduling 引入长度偏置。

#### Contribution 2：统一的四层分析框架

将问题拆分为：

1. reward shaping 与 filter placement；
2. advantage statistics；
3. loss aggregation；
4. optimizer update schedule。

该框架可用于分析 ALFWorld、WebShop、ScienceWorld 和其他 agentic RL 环境。

#### Contribution 3：Trajectory-aware GRPO

提出或系统化：

- trajectory-level reward normalization；
- trajectory-level token reducer；
- group-count-based optimizer scheduling；
- trajectory-level collective penalty；
- admissible-aware validity shaping；
- penalty-aware trajectory filtering。

#### Contribution 4：系统实证

通过受控实验检验：

- step-row 方法可能前期提升更快；
- trajectory-aware 方法训练更稳定；
- trajectory-aware 方法是否在长时程任务上具有更高上限；
- 差异在按实际 optimizer steps 对齐后是否仍然存在；
- trajectory-level penalty 与 post-penalty filtering 是否具有独立收益。

### 13.3 论文成立所需的最低证据

当前发现非常有价值，但要形成可靠论文，至少需要：

1. 多随机种子；
2. optimizer-budget 对齐；
3. advantage-only 与 reducer-only 消融；
4. penalty granularity 与 validity checker 消融；
5. task-reward filter 与 post-penalty filter 消融；
6. 至少两个 agentic 环境；
7. trajectory length 与 outcome 的统计相关性；
8. gradient/advantage/penalty 机制指标；
9. 在相同 prompt、reward、optimizer 和 evaluation protocol 下复现。

---

## 14. 建议图表

### Figure 1：Motivated example

使用本文的 8-rollout 例子：

- 上层：3 条短成功、5 条长失败；
- 中层：slime 按 8 trajectories 计算 mean/std；
- 下层：verl-agent 按 285 step rows 计算 mean/std；
- 右侧展示 advantage：
  - slime：`+1.208 / -0.725`
  - verl-agent：`+2.668 / -0.374`

### Figure 2：Optimizer schedule

```text
slime:
128 trajectories -> many microbatches -> 1 optimizer step

verl-agent:
thousands of step rows -> 256-row mini-batches -> N optimizer steps
```

### Figure 3：Learning curves under different x-axes

同时绘制：

- success vs trainer step；
- success vs optimizer step；
- success vs environment steps；
- success vs generated tokens；
- success vs wall-clock。

### Figure 4：Length-conditioned advantage

展示：

- \(A^+\) 与成功轨迹长度；
- \(A^-\) 与失败轨迹长度；
- trajectory contribution 与 trajectory length；
- step-level 和 trajectory-level 方法对比。

### Figure 5：Penalty-filter coupling

展示两条不同链路：

```text
slime:
invalid count
-> trajectory final reward
-> post-penalty dynamic filter
-> trajectory advantage
-> trajectory reducer

verl-agent:
invalid step
-> local raw step score
-> step-row advantage
-> token-mean mini-batch
```

并使用 8 条全失败、`K=[0,1,2,3,4,5,6,7]` 的 worked example 展示：

- slime 按 trajectory invalid count 排序；
- verl-agent 强烈惩罚 invalid steps，并轻微奖励其余 valid steps。

### Table 1：Advantage/reducer/filter 机制消融

| Advantage unit | Loss unit | Filter | Early speed | Final success |
|---|---|---|---:|---:|
| step | step/token | off |  |  |
| trajectory | step/token | off |  |  |
| step | trajectory | off |  |  |
| trajectory | trajectory | off |  |  |
| step | step/token | on |  |  |
| trajectory | step/token | on |  |  |
| step | trajectory | on |  |  |
| trajectory | trajectory | on |  |  |

### Table 2：Penalty 机制消融

| Penalty | Validity | Filter source | Invalid rate | Final success |
|---|---|---|---:|---:|
| none | canonical admissible-aware | off |  |  |
| raw-score-local | canonical format-only | off |  |  |
| raw-score-local | canonical admissible-aware | off |  |  |
| trajectory count | canonical admissible-aware | off |  |  |
| trajectory rate | canonical admissible-aware | off |  |  |
| trajectory count | canonical admissible-aware | task reward |  |  |
| trajectory count | canonical admissible-aware | post-penalty reward |  |  |

---

## 15. 威胁与边界

1. **当前曲线差异不是单变量实验。** 两个项目仍存在 prompt、history、clip、entropy、optimizer、penalty 和 evaluation 差异。
2. **advantage scalar mass 平衡不代表梯度影响平衡。** 不同 states/actions 的梯度方向不同，token 数和 mini-batch 顺序也会改变实际更新。
3. **trajectory-aware 不等于长度完全无关。** 长度仍影响：
   - trajectory 内部 token 比例；
   - prompt/history；
   - truncation；
   - invalid count；
   - 计算量；
   - 梯度方向。
4. **verl-agent 的 row padding 是条件性的。** 只有 row 数不能被当前 divisor 整除时才会随机复制。
5. **评测路径不同。** 本文核心比较限定于训练路径；slime evaluation 返回 trajectory summary sample。verl-agent 的 `success_rate` 在展平前按 episode 计算后复制到 step rows，而 ALFWorld validation `test_score` 整体按展平后的 step rows 汇总，因此不能直接把所有验证标量视为同一统计单位。
6. **第 5 节核心长度数值例子假设没有 invalid actions。** 第 10.4 节另行给出了包含 invalid actions 的 penalty-only worked example；在线训练中两种机制会同时存在。
7. **学习上限需要多种子确认。** 单次训练曲线不足以证明稳定上限。
8. **少 invalid 不一定等价于更接近成功。** trajectory-level collective penalty 可能提高全局一致性，也可能抑制高风险探索或强化安全但低效的行为。
9. **Penalty-aware filtering 会改变数据分布。** 它可能形成有效的困难样本课程，也可能使训练过度关注 invalid count，而不是 task progress。

---

## 16. 代码证据索引

### slime

| 机制 | 路径 |
|---|---|
| ALFWorld launcher | [`examples/alfworld/run_qwen2.5_3B_instruct_grpo.sh:35-138`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/run_qwen2.5_3B_instruct_grpo.sh#L35-L138) |
| step sample 构造 | [`examples/alfworld/batched_rollout.py:305-378`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/batched_rollout.py#L305-L378) |
| trajectory final reward 广播 | [`examples/alfworld/batched_rollout.py:381-435`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/batched_rollout.py#L381-L435) |
| admissible-aware action validity | [`examples/alfworld/prompts.py:117-153`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/prompts.py#L117-L153) |
| trajectory-level reward normalization | [`examples/alfworld/generate_with_alfworld.py:440-480`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/generate_with_alfworld.py#L440-L480) |
| dynamic filter | [`examples/alfworld/generate_with_alfworld.py:404-437`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/examples/alfworld/generate_with_alfworld.py#L404-L437) |
| Sample/group_id contract | [`slime/utils/types.py:9-21`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/slime/utils/types.py#L9-L21) |
| group denominator | [`slime/ray/rollout.py:807-823`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/slime/ray/rollout.py#L807-L823) |
| group-aware DP scheduler | [`slime/utils/dp_schedule.py:67-192`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/slime/utils/dp_schedule.py#L67-L192) |
| GRPO scalar broadcast | [`slime/utils/ppo_utils.py:201-208`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/slime/utils/ppo_utils.py#L201-L208) |
| trajectory reducer | [`slime/backends/megatron_utils/cp_utils.py:53-136`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/slime/backends/megatron_utils/cp_utils.py#L53-L136) |
| policy loss | [`slime/backends/megatron_utils/loss.py:830-1028`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/slime/backends/megatron_utils/loss.py#L830-L1028) |
| optimizer step | [`slime/backends/megatron_utils/model.py:419-605`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/slime/backends/megatron_utils/model.py#L419-L605) |
| reducer tests | [`tests/test_cp_utils.py:1-115`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/tests/test_cp_utils.py#L1-L115) |
| report consistency tests | [`tests/test_metric_report.py:55-95`](https://github.com/zhangdw156/slime/blob/21edc006b06cd93b7f6d59bdcb3cd15da1d1fc1b/tests/test_metric_report.py#L55-L95) |

### verl-agent

| 机制 | 路径 |
|---|---|
| ALFWorld launcher | [`../verl-agent/examples/grpo_trainer/run_alfworld.sh:1-84`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/examples/grpo_trainer/run_alfworld.sh#L1-L84) |
| uid/traj_uid 与 rollout | [`../verl-agent/agent_system/multi_turn_rollout/rollout_loop.py:285-414`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/agent_system/multi_turn_rollout/rollout_loop.py#L285-L414) |
| step rows flatten | [`../verl-agent/agent_system/multi_turn_rollout/rollout_loop.py:233-283`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/agent_system/multi_turn_rollout/rollout_loop.py#L233-L283) |
| episode reward manager | [`../verl-agent/agent_system/reward_manager/episode.py:39-79`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/agent_system/reward_manager/episode.py#L39-L79) |
| format-oriented action validity | [`../verl-agent/agent_system/environments/env_package/alfworld/projection.py:19-62`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/agent_system/environments/env_package/alfworld/projection.py#L19-L62) |
| invalid-action penalty | [`../verl-agent/verl/trainer/ppo/ray_trainer.py:200-224`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/ppo/ray_trainer.py#L200-L224) |
| GRPO dispatch | [`../verl-agent/verl/trainer/ppo/ray_trainer.py:244-362`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/ppo/ray_trainer.py#L244-L362) |
| cross-step GRPO advantage | [`../verl-agent/verl/trainer/ppo/core_algos.py:113-174`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/ppo/core_algos.py#L113-L174) |
| token-mean reducer | [`../verl-agent/verl/trainer/ppo/core_algos.py:395-428`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/ppo/core_algos.py#L395-L428) |
| policy loss | [`../verl-agent/verl/trainer/ppo/core_algos.py:431-492`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/ppo/core_algos.py#L431-L492) |
| row padding | [`../verl-agent/agent_system/multi_turn_rollout/utils.py:86-130`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/agent_system/multi_turn_rollout/utils.py#L86-L130) |
| actor mini-batch updates | [`../verl-agent/verl/workers/actor/dp_actor.py:317-443`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/workers/actor/dp_actor.py#L317-L443) |
| FSDP optimizer/scheduler | [`../verl-agent/verl/workers/fsdp_workers.py:360-375`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/workers/fsdp_workers.py#L360-L375); [`../verl-agent/verl/workers/fsdp_workers.py:600-626`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/workers/fsdp_workers.py#L600-L626) |
| defaults | [`../verl-agent/verl/trainer/config/ppo_trainer.yaml:43-68`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/config/ppo_trainer.yaml#L43-L68); [`../verl-agent/verl/trainer/config/ppo_trainer.yaml:234-257`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/config/ppo_trainer.yaml#L234-L257); [`../verl-agent/verl/trainer/config/ppo_trainer.yaml:292-304`](https://github.com/zhangdw156/verl-agent/blob/287d52e088675d7d5adb0bec621e1fb53b40d28b/verl/trainer/config/ppo_trainer.yaml#L292-L304) |

---

## 17. 当前结论

### 17.1 当前代码已经证实的机制事实

1. 两个项目都将 ALFWorld 训练 trajectory 展开为 step-level samples。
2. slime 将 trajectory 作为 GRPO advantage、loss 外层和 optimizer scheduling 的统计单位。
3. verl-agent 默认将 step rows 作为 GRPO mean/std 和 actor mini-batch 的统计单位。
4. 当失败轨迹显著长于成功轨迹时，verl-agent 会产生：
   - 稀有成功 step 的高正向 advantage；
   - 大量较小的负向 advantage rows；
   - 更多失败状态和动作的训练暴露；
   - 随 trajectory 长度变化的实际 Adam update 数。
5. slime 的 invalid penalty 以 trajectory invalid count 进入 final reward，并与 trajectory-level normalization、reducer 和 post-penalty dynamic filtering 耦合。
6. verl-agent 的 invalid penalty 只在当前 invalid step row 注入 raw-score 修正，但该修正会通过 `uid` mean/std 改变组内其他 rows 的 normalized advantages；其当前 validity checker 还不检查 admissible-action membership。

### 17.2 与实验曲线一致、但仍待消融验证的假设

1. 上述机制可能解释 verl-agent 前期更快但后期更早进入平台。
2. slime 的 trajectory-aware 设计可能牺牲部分前期速度，但减少长度偏置和错误 credit 的重复权重，并可能因此获得更高后期上限。
3. 动态后过滤可能通过困难样本筛选提高 slime 的长期学习效率和最终性能。
4. trajectory-level collective penalty、admissible-aware validity 与 penalty-aware filtering 的组合可能是 slime 后期上限更高的另一项原因。

### 17.3 当前机制探索目标

下一阶段将在 verl-agent ALFWorld 3B baseline 上执行 V0→V5 累积增量链：

```text
V0 native M1/M2/M3/M4，M5 off
-> V1 replace M4 with group scheduler
-> V2 replace M3 with trajectory reducer
-> V3 replace M2 with trajectory advantage
-> V4 replace M1 with trajectory penalty
-> V5 enable M5 dynamic filter
```

V0 为当前正在运行的指定对照组，但必须在 V1 启动前补齐第 12.1 节列出的
runtime manifest，之后才能视为可复现的固定 baseline。后续新增 V1～V5
五个实验。该阶段目标不是立即完成论文最终实验，而是定位：

- 哪个机制主要改变前期收敛速度；
- 哪个机制主要改变后期上限；
- 哪些机制组合存在协同或冲突；
- 是否可以找到比当前 slime full stack 更好的混合配置。

下一步应通过 optimizer-budget 对齐、advantage/reducer、penalty granularity、validity checker、filter reward source 的独立消融和多随机种子实验，将上述机制解释提升为可发表的因果证据。
