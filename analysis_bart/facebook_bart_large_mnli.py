import os
from transformers import pipeline


class FacebookBartLargeMNLI:
    def __init__(self, project_path: str, output_path: str = "outputs", max_chars: int = 4000):
        self.project_path = project_path
        self.output_path = output_path
        self.max_chars = max_chars

        self.release_labels = [
            "Rapid Releases",
            "Release Train",
            "Long Term Support (LTS)",
            "Insufficient Information"
        ]

        self.workflow_labels = [
            "Gitflow",
            "GitHub Flow",
            "Trunk-Based Development",
            "Insufficient Information"
        ]

        os.makedirs(self.output_path, exist_ok=True)

        print("[INFO] Loading facebook/bart-large-mnli...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

    def _read_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()[:self.max_chars]

    def analyze_file(self, file_name: str):
        print(f"[INFO] Analyzing {file_name}...")

        text = self._read_file(os.path.join(self.project_path, file_name))

        release_result = self.classifier(
            text,
            candidate_labels=self.release_labels,
            hypothesis_template="The release strategy of this project is {}."
        )

        workflow_result = self.classifier(
            text,
            candidate_labels=self.workflow_labels,
            hypothesis_template="The branching workflow of this project follows {}."
        )

        output = f"""
            File analyzed: {file_name}

            Release Strategy:
            - Prediction: {release_result['labels'][0]}
            - Confidence: {release_result['scores'][0]:.2f}

            Workflow Model:
            - Prediction: {workflow_result['labels'][0]}
            - Confidence: {workflow_result['scores'][0]:.2f}
            """

        out_file = os.path.join(
            self.output_path, f"bart_{file_name.replace('.md', '')}.txt"
        )

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(output)

        print(f"[OK] Saved {out_file}")

    def run(self):
        for file in sorted(os.listdir(self.project_path)):
            if file.endswith(".md"):
                self.analyze_file(file)
