from pathlib import Path
from rmfs.config import Params, AGVS, TASKS
from rmfs.layout import WarehouseLayout
from rmfs.collision import CPPlanner
from rmfs.ga import GeneticTAS
from rmfs.visualize import (
    plot_history,
    plot_tas,
    plot_one_path,
    write_results,
    plot_chromosome_encoding,
    plot_all_agv_paths,
    animate_agv_paths_gif,
)


def main():
    out_dir = Path(__file__).parent / 'output'
    out_dir.mkdir(exist_ok=True)

    params = Params()
    layout = WarehouseLayout(params)
    planner = CPPlanner(layout, params)
    ga = GeneticTAS(AGVS, TASKS, planner, params)

    # 1) 先跑 GA，得到最优解
    result = ga.run()

    # 2) 用“详细 CPP 评估器”重新评估一次最终最优解
    #    这样拿到更准确的逐段路径和时间信息，适合做动画
    final_eval = planner.evaluate_sequences(
        AGVS,
        TASKS,
        result.best_eval.agv_task_sequences
    )

    # 3) 原有图
    plot_history(result.history, out_dir / 'fig8_iteration_curve.png')
    plot_tas(layout, final_eval, out_dir / 'fig9_tas_grid.png')
    plot_one_path(layout, final_eval, out_dir / 'fig10_path_planning.png', preferred_agv=3, preferred_task=16)

    # 4) 新增：染色体编码可视化
    plot_chromosome_encoding(result.best_chromosome, final_eval, out_dir / 'fig_best_chromosome.png')

    # 5) 新增：所有 AGV 的静态路径图
    plot_all_agv_paths(layout, final_eval, out_dir / 'fig_all_agv_paths.png')

    # 6) 新增：所有 AGV 执行动图
    animate_agv_paths_gif(layout, final_eval, AGVS, out_dir / 'agv_paths_animation.gif', fps=4, tail=8)

    # 7) 写结果
    write_results(final_eval, result.best_chromosome, result.history, out_dir / 'results.json')

    print('Done. Key results:')
    print(f'  picker completion time: {final_eval.makespan_picker} s')
    print(f'  AGV completion time:    {final_eval.makespan_agv} s')
    print(f'  total AGV cost:         {final_eval.total_cost:.6f} CNY')
    print(f'  waiting time:           {final_eval.waiting_time} s')
    print(f'Outputs are in: {out_dir}')


if __name__ == '__main__':
    main()