# AGV Navigation and Scheduling System

## Overview

This repository contains multiple implementations and experiments for Automated Guided Vehicle (AGV) path planning and task scheduling in warehouse environments. The project includes various approaches ranging from greedy algorithms to advanced genetic algorithm optimizations, developed for research, competitions, and practical applications.

The system coordinates multiple AGVs to efficiently complete transportation tasks in a grid-based warehouse, ensuring collision-free operations, optimal task allocation, and adherence to time constraints.

## Project Components

### Root Directory - Competition Version
**Description**: Core implementation developed for the 2025 Siemens Xcelerator Open Competition - MioVerse Track. Uses greedy task allocation with A* path planning for collision avoidance.

**Key Features**:
- Multi-agent path planning with temporal constraints
- Greedy cost-matrix task allocation
- Real-time conflict resolution
- Pygame-based visualization with video export
- Handles 12 AGVs and 102 tasks in a 21×21 grid

**Main Files**:
- `navigation.py`: Main simulation algorithm
- `display.py`: Visualization module
- `project_report.ipynb`: Comprehensive analysis notebook
- `agv_position.csv`, `agv_task.csv`: Input data
- `agv_trajectory.csv`: Output trajectories

**Run Instructions**:
```bash
python navigation.py
python display.py  # For visualization
```

### GA-Astar-CP/
**Description**: Reproduction of the bi-level GA-A*-CP algorithm from research literature.

**Key Features**:
- Upper level: Genetic algorithm for task allocation and sequencing
- Lower level: Improved A* path planning with turn penalties
- Conflict resolution via queue/conflict waiting
- Visualizations for convergence, task allocation, and path planning

**Run Instructions**:
```bash
cd GA-Astar-CP
pip install -r requirements.txt
python main.py
```

### agv_ga_astar_cp_project/
**Description**: Global scheduling version replacing greedy allocation with GA-A*-CP optimization.

**Key Features**:
- Genetic algorithm optimization for task sequencing and AGV assignment
- A* path planning with conflict detection
- Maintains original data formats and visualization

**Run Instructions**:
```bash
cd agv_ga_astar_cp_project
python generate_data.py
python navigation.py
python display.py
```

### Dynamic AGV Quantity Optimization
**Folders**: `动态AGV数量——基于文章/`, `动态AGV数量——基于项目/`

**Description**: Implementations for optimizing the number of AGVs required based on task demands.

**Key Features**:
- Dynamic AGV fleet sizing
- Based on research articles and project requirements
- Genetic algorithm optimization

### New AGV Implementations
**Folders**: `new_agv/`, `new_agv_demo1/`, `new_agv_demo2/`, `new_agv_demo3/`, `new_agv_demo4/`

**Description**: Various demo versions and new implementations with different warehouse layouts, features, and optimizations.

**Key Features**:
- Workstation-based layouts
- Real-world data generation
- Enhanced visualization options

### WareRover-main/
**Description**: Additional AGV-related project or integrated components.

## Installation

### System Requirements
- Python 3.7+
- Windows/Linux/macOS

### Python Dependencies
```bash
pip install pandas numpy matplotlib seaborn pygame imageio
```

Individual subprojects may have additional requirements listed in their respective `requirements.txt` files.

## Usage

### General Workflow
1. **Data Generation**: Run data generation scripts if needed (e.g., `generate_data.py`)
2. **Simulation**: Execute the main navigation script
3. **Visualization**: Run display scripts to visualize results
4. **Analysis**: Review output CSV files and analysis notebooks

### Example Commands
```bash
# Root directory
python navigation.py
python display.py --save-mp4 simulation.mp4

# GA-Astar-CP
cd GA-Astar-CP
python main.py

# Global GA version
cd agv_ga_astar_cp_project
python navigation.py
python display.py
```

## Algorithm Overview

### Path Planning
- **A* Algorithm**: Manhattan distance heuristic with turn penalties
- **Constraints**: Collision avoidance, boundary limits, orientation changes
- **Features**: Temporal planning, conflict prediction

### Task Allocation
- **Greedy Approach**: Cost-matrix based assignment (root version)
- **Genetic Algorithm**: Evolutionary optimization for global scheduling
- **Priority Handling**: Weighting for urgent tasks

### Conflict Resolution
- **Detection**: Position conflicts and path swaps
- **Resolution**: Waiting strategies and replanning

## Data Formats

### Input Files
- `agv_position.csv`: Warehouse layout, AGV initial positions
- `agv_task.csv`: Task definitions with pickup/delivery points

### Output Files
- `agv_trajectory.csv`: Complete AGV movement logs
- `agv_summary.csv`: Performance summaries
- `ga_history.csv`: Optimization convergence data

## Performance Metrics

- **Scalability**: Tested with 12 AGVs, 102 tasks
- **Runtime**: Varies by algorithm complexity (seconds to minutes)
- **Success Rate**: Collision-free task completion
- **Optimization**: GA versions show improved scheduling efficiency

## Future Enhancements

- Advanced MAPF algorithms (CBS, Priority-based)
- Machine learning integration
- Real-time replanning
- Multi-objective optimization
- Cloud deployment capabilities

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## License

This project is for educational and research purposes. Please check individual subprojects for specific licensing.

## Contact

For questions or collaborations, please refer to the project documentation or create an issue in the repository.

---

For detailed competition rules, algorithm explanations, and performance analysis, see the original documentation in each subproject's README and analysis notebooks.

