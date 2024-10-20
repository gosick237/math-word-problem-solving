import os, json

def path_checker(pathes):
    for path in pathes:
        if not os.path.exists(path):
            print(f"Given path '{path}' is not exist. Check your path!")
            quit()

def load_jsonl(filename):
    data = []
    with open(filename, "r") as new_file:
        data = [json.loads(l) for l in new_file]
    return data

def save_jsonl(data, target_dir, filename):
    # File configuration
    new_filename = os.path.splitext(filename)[0] + '.jsonl'
    target_path = os.path.join(target_dir, new_filename)
    # Save
    with open(target_path, 'w', encoding="utf-8") as f:
        for chunk in data:
            json.dump(chunk, f)
            f.write("\n")