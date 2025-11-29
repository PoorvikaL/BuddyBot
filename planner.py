from datetime import datetime, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI


def generate_onboarding_plan(user_name: str, role: str, start_date_str: str, llm: ChatGoogleGenerativeAI):
    """
    Returns a list of tasks with: task_id, title, type, day, due_date, status.
    """
    start_date = datetime.fromisoformat(start_date_str)

    prompt = f"""
You are an onboarding coordinator.

Create a simple 10-day (2-week work days) onboarding plan for a new hire.

New hire name: {user_name}
Role: {role}
Start date: {start_date_str}

Return 10 tasks as bullet points in this exact format:
Day X - [type] - Title of task

Where:
- type is one of: "form", "training", "meeting", "tools", "other".
- Day 1 means first working day (start_date), Day 10 is day 10.

Do NOT add explanations, only the bullet list.
"""

    resp = llm.invoke(prompt)
    lines = resp.content.splitlines()
    tasks = []
    task_id = 1

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # remove bullet markers like "- " or "* "
        if line.startswith(("-", "*")):
            line = line[1:].strip()

        try:
            if line.lower().startswith("day"):
                # split into: Day X | [type] | title
                parts = [p.strip() for p in line.split("-", 2)]
                day_part = parts[0]                      # "Day 1"
                type_part = parts[1].strip("[]") if len(parts) > 1 else "other"
                title_part = parts[2] if len(parts) > 2 else "Task"

                day_num = int(day_part.lower().replace("day", "").strip())
                due_date = (start_date + timedelta(days=day_num - 1)).date().isoformat()

                tasks.append({
                    "task_id": f"T{task_id}",
                    "title": title_part,
                    "type": type_part.lower(),
                    "day": day_num,
                    "due_date": due_date,
                    "status": "pending",
                })
                task_id += 1
        except Exception:
            continue

    return tasks
