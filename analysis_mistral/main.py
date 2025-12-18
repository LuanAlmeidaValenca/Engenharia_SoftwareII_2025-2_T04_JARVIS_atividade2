import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========= CONFIGURAÇÕES =========
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
PROJECT_PATH = "Jarvis"
OUTPUT_PATH = "outputs/mistral_analysis.txt"

MAX_NEW_TOKENS = 900

# ========= FUNÇÕES =========

def load_documents(folder):
    texts = []
    for file in sorted(os.listdir(folder)):
        if file.endswith(".md"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                content = f.read()
                texts.append(f"\n===== {file} =====\n{content}")
    return "\n".join(texts)


def build_prompt(documents):
    return f"""
You are a software engineering researcher specialized in open-source governance.

Based ONLY on the documentation below from the GitHub project microsoft/jarvis, perform the following tasks:

1. Identify the release strategy adopted by the project (Rapid Releases, Release Train, LTS + Current, or Insufficient Information).
2. Identify the branching/workflow model used (Gitflow, GitHub Flow, or Insufficient Information).
3. Justify each classification using explicit evidence from the documentation.
4. If the documentation is insufficient, explicitly state this limitation.

Documentation:
{documents}
"""


# ========= EXECUÇÃO =========

def main():
    print("[INFO] Carregando documentos...")
    documents = load_documents(PROJECT_PATH)

    print("[INFO] Construindo prompt...")
    prompt = build_prompt(documents)

    print("[INFO] Carregando modelo e tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("[INFO] Executando inferência...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    os.makedirs("outputs", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"[INFO] Análise concluída. Resultado salvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()