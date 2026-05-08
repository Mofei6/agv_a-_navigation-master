# 固定 5 台 AGV 的 GA-A*-CP 任务分配与路径规划

本项目保留原有工厂布局、3x3 工位、取料点、合法卸货点、任务时间窗、轨迹 CSV 与 display.py 动画输出格式；仅将原来的逐任务贪心调度替换为固定 AGV 数量下的 hybrid metaheuristic algorithm：GA-A*-CP。

## 固定 AGV 数量说明

本版本不优化动态 AGV 数量。默认固定启用 5 台 AGV：

```bash
python navigation.py
```

默认等价于：

```bash
MAX_ACTIVE_AGVS=5 python navigation.py
```

也可以指定具体 5 台 AGV：

```bash
FIXED_AGV_IDS=1,3,5,7,9 python navigation.py
```

候选 AGV 仍从 `agv_position.csv` 读取，但 GA 只在固定启用的 AGV 集合内优化任务分配和排序。

## 算法流程

1. `generate_data.py` 生成或更新：
   - `agv_position.csv`
   - `agv_task.csv`

2. `navigation.py` 固定启用 5 台 AGV，并调用 GA-A*-CP：
   - GA 上层优化 `task_order` 和 `agv_assign`；
   - A*-CP 下层对每个染色体从零构建预约表并规划无冲突路径；
   - 目标函数综合时间窗迟到、最大迟到、完工时间、距离、等待和负载均衡。

3. `display.py` 读取输出 CSV 进行动画展示或导出视频。

## 运行步骤

```bash
python generate_data.py
python navigation.py
python display.py
```

快速测试：

```bash
GA_POPULATION=4 GA_GENERATIONS=2 python navigation.py
```

更充分搜索：

```bash
MAX_ACTIVE_AGVS=5 GA_POPULATION=80 GA_GENERATIONS=120 python navigation.py
```

指定固定 AGV：

```bash
FIXED_AGV_IDS=1,3,5,7,9 GA_POPULATION=80 GA_GENERATIONS=120 python navigation.py
```

导出 MP4：

```bash
python display.py --save-mp4 agv_simulation.mp4
```

## 主要输出文件

- `agv_trajectory.csv`：AGV 逐时刻轨迹，兼容原 `display.py`。
- `agv_task_result.csv`：任务执行结果。
- `agv_summary.csv`：固定启用 AGV 数量、任务延迟、距离、makespan 等汇总。
- `ga_history.csv`：GA 每代收敛记录。
- `best_chromosome.csv`：最终最优染色体编码。
- `best_agv_sequences.csv`：解码后的每台 AGV 任务序列。

## 文件说明

- `navigation.py`：保留地图、A*、预约表、输出 CSV，并调用 GA-A*-CP。
- `ga_optimizer.py`：GA 上层任务分配与排序优化。
- `cpp_evaluator.py`：A*-CP 下层完整方案评估。
- `generate_data.py`：原数据生成逻辑。
- `display.py`：原动画展示逻辑。
