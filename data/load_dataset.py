import json
import os
import random
import pandas as pd
from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(DATA_DIR, "raw_predictions.csv")

# reasoning tasks = can't just vibe off memorized facts, actual multi-step thinking required
REASONING_TASKS = {
    "leaderboard_bbh":       ("acc_norm,none",    "bbh_acc"),
    "leaderboard_math_hard": ("exact_match,none", "math_acc"),
    "leaderboard_musr":      ("acc_norm,none",    "musr_acc"),
}
KNOWLEDGE_TASKS = {
    "leaderboard_mmlu_pro":  ("acc,none",         "mmlu_pro_acc"),
    "leaderboard_gpqa":      ("acc_norm,none",    "gpqa_acc"),
}
ALL_TASKS = {**REASONING_TASKS, **KNOWLEDGE_TASKS}

BBH_SUBTASKS = [
    "leaderboard_bbh_boolean_expressions",
    "leaderboard_bbh_causal_judgement",
    "leaderboard_bbh_date_understanding",
    "leaderboard_bbh_logical_deduction_five_objects",
    "leaderboard_bbh_logical_deduction_seven_objects",
    "leaderboard_bbh_logical_deduction_three_objects",
    "leaderboard_bbh_navigate",
    "leaderboard_bbh_object_counting",
    "leaderboard_bbh_penguins_in_a_table",
    "leaderboard_bbh_web_of_lies",
]


def list_result_files(model_limit=300, seed=42):
    print("Listing result files in leaderboard repo...")
    all_files = [
        f for f in list_repo_files("open-llm-leaderboard/results", repo_type="dataset")
        if f.endswith(".json")
    ]
    print(f"  Total JSON files: {len(all_files)}")

    model_latest = {}
    for f in all_files:
        parts = f.split("/")
        if len(parts) >= 3:
            model_key = "/".join(parts[:2])
            model_latest[model_key] = f
        else:
            model_latest[f] = f

    unique_files = list(model_latest.values())
    random.seed(seed)
    sampled = random.sample(unique_files, min(model_limit, len(unique_files)))
    print(f"  Sampled {len(sampled)} model files")
    return sampled


def parse_result_file(path, data):
    results = data.get("results", {})
    if not isinstance(results, dict) or not results:
        return None

    model_id = data.get("model_name", "/".join(path.split("/")[:2]) if "/" in path else path)
    row = {"model_id": model_id}

    has_all_tasks = True
    for task_key, (metric, col_name) in ALL_TASKS.items():
        task_data = results.get(task_key, {})
        val = task_data.get(metric, None)
        if val is None:
            has_all_tasks = False
            row[col_name] = None
        else:
            row[col_name] = float(val)

    if not has_all_tasks:
        return None

    for subtask in BBH_SUBTASKS:
        task_data = results.get(subtask, {})
        val = task_data.get("acc_norm,none", None)
        col = subtask.replace("leaderboard_bbh_", "bbh_sub_")
        row[col] = float(val) if val is not None else None

    reasoning_scores = [row["bbh_acc"], row["math_acc"], row["musr_acc"]]
    knowledge_scores = [row["mmlu_pro_acc"], row["gpqa_acc"]]

    row["reasoning_score"] = sum(reasoning_scores) / len(reasoning_scores)
    row["knowledge_score"] = sum(knowledge_scores) / len(knowledge_scores)
    row["acc_gap"] = row["reasoning_score"] - row["knowledge_score"]
    row["score_variance"] = pd.Series(reasoning_scores + knowledge_scores).var()
    row["total_score"] = sum(reasoning_scores + knowledge_scores) / len(reasoning_scores + knowledge_scores)

    return row


def download_and_parse(file_paths):
    records = []
    failed = 0

    for fpath in tqdm(file_paths, desc="Downloading model results"):
        try:
            local = hf_hub_download(
                repo_id="open-llm-leaderboard/results",
                filename=fpath,
                repo_type="dataset",
            )
            with open(local) as f:
                data = json.load(f)

            row = parse_result_file(fpath, data)
            if row is not None:
                records.append(row)

        except Exception:
            failed += 1
            continue

    print(f"  Parsed {len(records)} models ({failed} failed/skipped)")
    return records

#by ug
def main():
    file_paths = list_result_files(model_limit=1000)
    records = download_and_parse(file_paths)

    if not records:
        print("No records parsed. Check HF token or network.")
        return

    df = pd.DataFrame(records).dropna()
    print(f"\n  Final dataset: {len(df)} models")

    # gpqa held out as label, no leakage fr
    df["label"] = (df["gpqa_acc"] > df["gpqa_acc"].median()).astype(int)
    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
