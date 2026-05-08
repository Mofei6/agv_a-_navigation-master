# 动态 AGV 数量优化版项目代码

## 核心变化

本版本不再先固定 AGV 数量再规划路径，而是：

1. 在 `generate_data.py` 中设置最大候选 AGV 数量 `MAX_CANDIDATE_AGVS`；
2. `generate_data.py` 生成 `agv_position_candidates.csv`，里面的 AGV 只是候选池；
3. `navigation.py` 在优化过程中自动决定启用哪些 AGV；
4. 输出的 `agv_position.csv` 只包含最终实际启用的 AGV；
5. `display.py` 只显示实际启用的 AGV。

## 运行顺序

```bash
python generate_data.py
python navigation.py
python display.py
```

## 动态 AGV 数量优化逻辑

`navigation.py` 中维护两类 AGV：

- `active_agvs`：已经启用的 AGV；
- `inactive_agvs`：候选池中尚未启用的 AGV。

每次任务分配时，算法同时评估两类动作：

1. `insert`：把任务分配给已启用 AGV；
2. `activate`：启用一台新的候选 AGV 执行任务。

动作代价函数包含：

- AGV 固定启用成本；
- 总延迟；
- 延迟平方惩罚；
- 最大延迟惩罚；
- 行驶距离；
- 完成时间。

因此，只有当启用新 AGV 带来的延迟下降价值大于 AGV 使用成本时，算法才会启用新 AGV。

## 输出文件

- `agv_position.csv`：最终实际启用 AGV 的位置文件；
- `agv_trajectory.csv`：最终无冲突轨迹；
- `agv_task_result.csv`：每个任务的执行结果；
- `agv_summary.csv`：包含候选 AGV 数量、实际启用 AGV 数量、总延迟、最大延迟、总行驶距离、总成本等指标。

## 关键参数

在 `generate_data.py` 中：

```python
MAX_CANDIDATE_AGVS = 10
```

这表示最多允许启用 10 台候选 AGV，但不是最终启用 10 台。

在 `navigation.py` 中：

```python
AGV_FIXED_COST = 9000
TOTAL_DELAY_WEIGHT = 1200
DELAY_SQUARE_WEIGHT = 30
MAX_DELAY_WEIGHT = 2500
DISTANCE_WEIGHT = 12
```

如果想让算法更愿意启用更多 AGV，可以降低 `AGV_FIXED_COST` 或提高延迟相关权重。

如果想让算法更节约 AGV 数量，可以提高 `AGV_FIXED_COST`。
