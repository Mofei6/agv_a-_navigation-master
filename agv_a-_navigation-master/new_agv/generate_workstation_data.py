"""
Generate workstation-based input files for the AGV simulation.

Output:
- agv_position_workstation.csv
- agv_task_workstation.csv

Coordinate system is 1-based, same as the original project.
Types:
- material_zone: the single material area where every AGV picks up materials
- workstation: delivery/processing workstation, such as 130A/200B/209D/240A
- agv: initial AGV position and pitch
"""
import csv
import random
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
random.seed(7)

# Approximate layout extracted from the provided floor-plan image.
# You can add/replace IDs such as 240A here without changing navigation.py.
WORKSTATIONS = [
    # left lower block
    ("130A", 4, 5), ("130B", 6, 5), ("140A", 8, 5), ("140B", 10, 5), ("140C", 12, 5), ("150", 14, 5),
    ("130A-2", 4, 9), ("130B-2", 6, 9), ("140A-2", 8, 9), ("140B-2", 10, 9), ("140C-2", 12, 9), ("150-2", 14, 9),
    ("130A-3", 4, 13), ("130B-3", 6, 13), ("140A-3", 8, 13), ("140B-3", 10, 13), ("140C-3", 12, 13), ("150-3", 14, 13),
    # right lower block
    ("200A", 16, 5), ("200B", 18, 5), ("200C", 20, 5),
    ("200A-2", 16, 9), ("200B-2", 18, 9), ("200C-2", 20, 9),
    ("200A-3", 16, 13), ("200B-3", 18, 13), ("200C-3", 20, 13),
    # upper block
    ("209F", 14, 18), ("209E", 15, 18), ("209D", 16, 18), ("209C", 17, 18), ("209B", 18, 18), ("209A", 19, 18),
    # examples for new-style station IDs, disabled by default; uncomment when needed
    # ("240A", 5, 16), ("240B", 7, 16),
]

AGVS = [
    ("Optimus", 2, 1, 90), ("Bumblebee", 4, 1, 90), ("Jazz", 6, 1, 90),
    ("Sideswipe", 8, 1, 90), ("Wheeljack", 10, 1, 90), ("Ratchet", 12, 1, 90),
    ("Ironhide", 14, 1, 90), ("Hound", 16, 1, 90), ("Smokescreen", 18, 1, 90),
    ("Megatron", 20, 1, 90), ("Bluestreak", 2, 20, 270), ("RedAlert", 4, 20, 270),
]

MATERIAL_ZONE = ("Material", 1, 1)


def write_positions():
    with open(OUT_DIR / "agv_position_workstation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "name", "x", "y", "pitch"])
        writer.writerow(["material_zone", MATERIAL_ZONE[0], MATERIAL_ZONE[1], MATERIAL_ZONE[2], ""])
        for name, x, y in WORKSTATIONS:
            writer.writerow(["workstation", name, x, y, ""])
        for name, x, y, pitch in AGVS:
            writer.writerow(["agv", name, x, y, pitch])


def write_tasks(tasks_per_workstation=3):
    priorities = ["Normal", "Normal", "Normal", "Urgent"]
    with open(OUT_DIR / "agv_task_workstation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id", "workstation", "release_time", "due_time", "processing_time", "priority", "material_zone"
        ])
        for ws, _, _ in WORKSTATIONS:
            base = random.randint(0, 20)
            for i in range(1, tasks_per_workstation + 1):
                release = base + (i - 1) * random.randint(25, 45)
                processing = random.randint(8, 18)
                priority = random.choice(priorities)
                slack = 55 if priority == "Urgent" else 95
                due = release + slack + random.randint(0, 35)
                writer.writerow([f"{ws}-{i}", ws, release, due, processing, priority, MATERIAL_ZONE[0]])


if __name__ == "__main__":
    write_positions()
    write_tasks()
    print("Generated agv_position_workstation.csv and agv_task_workstation.csv")
