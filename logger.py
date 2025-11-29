import os
import csv
from datetime import datetime

LOG_DIR = "logs"
INTERACTIONS_PATH = os.path.join(LOG_DIR, "interactions.csv")
TASKS_PATH = os.path.join(LOG_DIR, "tasks.csv")

INTERACTION_COLUMNS = [
    "timestamp",
    "user_name",
    "role",
    "question",
    "answer",
    "category",
]

TASK_COLUMNS = [
    "task_id",
    "user_name",
    "role",
    "start_date",
    "title",
    "day",
    "due_date",
    "type",
    "status",
]

os.makedirs(LOG_DIR, exist_ok=True)


def _init_csv(path, columns):
    if not os.path.exists(path):
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)


def log_interaction(user_name, role, question, answer, category):
    _init_csv(INTERACTIONS_PATH, INTERACTION_COLUMNS)
    with open(INTERACTIONS_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            user_name,
            role,
            question,
            answer,
            category,
        ])


def log_tasks(user_name, role, start_date, tasks):
    _init_csv(TASKS_PATH, TASK_COLUMNS)
    with open(TASKS_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for t in tasks:
            writer.writerow([
                t["task_id"],
                user_name,
                role,
                start_date,
                t["title"],
                t["day"],
                t["due_date"],
                t["type"],
                t.get("status", "pending"),
            ])
