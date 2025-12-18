import os
import sys

from facebook_bart_large_mnli import FacebookBartLargeMNLI
from qwen_25_coder import Qwen25Coder


def escolher_modelo():
    modelos = {
        1: "facebook/bart-large-mnli",
        2: "Qwen/Qwen2.5-Coder-7B-Instruct",
        3: "Modelo futuro (não implementado)",
    }

    print("\n=== Escolha o MODELO de análise ===")
    for k, v in modelos.items():
        print(f"  {k}. {v}")

    while True:
        try:
            escolha = int(input("Digite o número do modelo desejado: ").strip())
            if escolha in modelos:
                return escolha
            print("Opção inválida.")
        except ValueError:
            print("Entrada inválida.")


def executar_bart():
    caminho = os.path.join(os.getcwd(), "Jarvis")
    model = FacebookBartLargeMNLI(caminho)
    model.run()


def executar_qwen():
    print("\n⚠ Este modelo requer GPU (recomendado Google Colab).")
    texto = input("\nCole o conteúdo do arquivo que deseja analisar:\n\n")
    model = Qwen25Coder()
    resultado = model.analyze_document(texto)
    print("\n--- RESULTADO ---\n")
    print(resultado)


def main():
    escolha = escolher_modelo()

    if escolha == 1:
        executar_bart()
    elif escolha == 2:
        executar_qwen()
    elif escolha == 3:
        print("Modelo ainda não implementado.")
        sys.exit(0)
    else:
        raise ValueError("Escolha inválida.")


if __name__ == "__main__":
    main()
