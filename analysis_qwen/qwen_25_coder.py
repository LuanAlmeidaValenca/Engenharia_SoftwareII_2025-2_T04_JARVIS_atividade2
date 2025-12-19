import torch
import json
import os
from datetime import datetime
from transformers import pipeline, BitsAndBytesConfig

class Qwen25Coder:
    def __init__(self, output_file="analise_resultados.json"):
        self.model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
        self.output_file = output_file  # Define o nome do arquivo de saída

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

    def _save_to_json(self, analysis_result, source_name):
        """Salva o resultado da análise em um arquivo JSON de forma incremental."""
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_document": source_name,
            "analysis": analysis_result
        }

        # Verifica se o arquivo já existe e carrega o conteúdo
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [] # Garante que seja uma lista
            except (json.JSONDecodeError, ValueError):
                data = []
        else:
            data = []

        # Adiciona a nova entrada
        data.append(entry)

        # Salva de volta no arquivo
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"[INFO] Resultado salvo em '{self.output_file}'")

    def analyze_document(self, document_text: str, source_identifier: str = "documento_generico"):
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

        result_text = outputs[0]["generated_text"][-1]["content"]
        
        # Chama a função para salvar antes de retornar
        self._save_to_json(result_text, source_identifier)

        return result_text
