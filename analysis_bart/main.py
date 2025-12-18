import os
from transformers import pipeline

PROJECT_PATH = "Jarvis"
OUTPUT_PATH = "outputs"

RELEASE_LABELS = [
    "Rapid Releases",
    "Release Train",
    "Long Term Support (LTS)",
    "Insufficient Information"
]

WORKFLOW_LABELS = [
    "Gitflow",
    "GitHub Flow",
    "Trunk-Based Development",
    "Insufficient Information"
]

MAX_CHARS = 4000


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()[:MAX_CHARS]


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("[INFO] Loading BART-large-MNLI...")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

    for file in sorted(os.listdir(PROJECT_PATH)):
        if not file.endswith(".md"):
            continue

        print(f"[INFO] Analyzing {file}...")

        text = read_file(os.path.join(PROJECT_PATH, file))

        release_result = classifier(
            text,
            candidate_labels=RELEASE_LABELS,
            hypothesis_template="The release strategy of this project is {}."
        )

        workflow_result = classifier(
            text,
            candidate_labels=WORKFLOW_LABELS,
            hypothesis_template="The branching workflow of this project follows {}."
        )

        output = f"""
File analyzed: {file}

Release Strategy:
- Prediction: {release_result['labels'][0]}
- Confidence: {release_result['scores'][0]:.2f}

Workflow Model:
- Prediction: {workflow_result['labels'][0]}
- Confidence: {workflow_result['scores'][0]:.2f}
"""

        out_file = os.path.join(
            OUTPUT_PATH, f"bart_{file.replace('.md','')}.txt"
        )

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(output)

        print(f"[OK] Saved {out_file}")


if __name__ == "__main__":
    main()