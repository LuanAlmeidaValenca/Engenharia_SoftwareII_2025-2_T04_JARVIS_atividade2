import torch
from transformers import pipeline, BitsAndBytesConfig


class Qwen25Coder:
    def __init__(self):
        self.model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"

        print("[INFO] Loading Qwen 2.5 Coder (4-bit)...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"quantization_config": bnb_config},
            device_map="auto",
        )

    def analyze_document(self, document_text: str):
        messages = [
            {
                "role": "system",
                "content": (
                    "Você é um Engenheiro de Software Sênior especialista em "
                    "DevOps, Governança de TI e Processos de Software Open Source."
                )
            },
            {
                "role": "user",
                "content": f"""
Com base EXCLUSIVAMENTE na documentação fornecida abaixo, analise a governança do projeto
e responda em Português do Brasil:

1. Estratégia de Releases
2. Modelo de Fluxo de Trabalho (Branching Model)

DOCUMENTAÇÃO:
{document_text}
"""
            }
        ]

        outputs = self.pipe(
            messages,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
        )

        return outputs[0]["generated_text"][-1]["content"]
