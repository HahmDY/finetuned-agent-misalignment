import os
import json


PING_HOME = os.getenv("PING_HOME")
BENIGN_TASKS = f"{PING_HOME}/WebDojo/tasks/tasks_benign_raw.json"
HARMFUL_TASKS = f"{PING_HOME}/WebDojo/tasks/tasks_harmful_raw.json"
BENIGN_TASKS_PROCESSED = f"{PING_HOME}/WebDojo/tasks/tasks_benign.json"
HARMFUL_TASKS_PROCESSED = f"{PING_HOME}/WebDojo/tasks/tasks_harmful.json"

def replace_git_home_with_localhost(port):
    """
    Replace all occurrences of __GIT_HOME__ in the benign and harmful tasks JSON files
    with localhost:{port} and save the results to the processed files.
    """
    file_pairs = [
        (BENIGN_TASKS, BENIGN_TASKS_PROCESSED),
        (HARMFUL_TASKS, HARMFUL_TASKS_PROCESSED)
    ]
    for src_path, dst_path in file_pairs:
        with open(src_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        def replace_in_criteria(criteria):
            for item in criteria:
                if isinstance(item, list) and len(item) == 2 and isinstance(item[1], str):
                    item[1] = item[1].replace("__GIT_HOME__", f"localhost:{port}")
            return criteria
        for entry in data:
            if 'criteria' in entry:
                entry['criteria'] = replace_in_criteria(entry['criteria'])
        with open(dst_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python deploy.py <port>")
        exit(1)
    port = sys.argv[1]
    replace_git_home_with_localhost(port)