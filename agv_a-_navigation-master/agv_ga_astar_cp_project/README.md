
# AGV GA-A*-CP 全局调度版本

本项目保留原项目的数据格式、工厂工位布局、3x3 工位、两个合法卸货点、取料点、任务时间窗、Pygame 动画播放和 MP4 导出功能；只把 `navigation.py` 中的逐任务贪心调度替换为 GA-A*-CP 全局优化。

## 文件说明

- `generate_data.py`：生成 `agv_position.csv` 与 `agv_task.csv`。
- `navigation.py`：主入口。读取输入，调用 GA-A*-CP，输出原有 CSV。
- `cpp_evaluator.py`：下层 A*-CP 评估器。对一个完整染色体重建预约表并规划所有 AGV 路径。
- `ga_optimizer.py`：上层 GA。优化任务排列与 AGV 分配。
- `display.py`：原动画显示程序，保持不变。
- `agv_position.csv`、`agv_task.csv`：示例输入。
- `agv_trajectory.csv`、`agv_task_result.csv`、`agv_summary.csv`：运行 `navigation.py` 后生成。
- `ga_history.csv`：GA 收敛过程。
- `best_chromosome.csv`：最终最优染色体编码。
- `best_agv_sequences.csv`：最终每台 AGV 的任务序列。

## 运行步骤

```bash
python generate_data.py
python navigation.py
python display.py
```

导出视频：

```bash
python display.py --save-mp4 agv_simulation.mp4
```

## 调整 GA 规模

默认参数为了方便快速运行，设置为：

- population = 8
- generations = 8

可以通过环境变量提高搜索强度：

```bash
GA_POPULATION=30 GA_GENERATIONS=50 python navigation.py
```

Windows PowerShell:

```powershell
$env:GA_POPULATION=30
$env:GA_GENERATIONS=50
python navigation.py
```

## 目标函数

下层 A*-CP 对每个染色体输出完整执行结果；上层 GA 使用加权词典序目标：

```text
score =
  1_000_000 * infeasible_count
+   100_000 * late_task_count
+    10_000 * total_tardiness
+     1_000 * max_tardiness
+        10 * makespan
+         1 * total_distance
+         2 * total_wait
+        50 * workload_imbalance
```

优先保证可行性和时间窗，再优化完工时间、距离、等待和负载均衡。
