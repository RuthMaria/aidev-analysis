"""
explore.py - Primeiro contato com o dataset AIDev.

Mostra, para cada tabela principal, o tamanho, as colunas e algumas linhas.
Rode primeiro este script para entender o que tem em cada arquivo.

Origem dos dados (variavel de ambiente AIDEV_DATA):
  - nao definida -> le direto do Hugging Face (precisa de `huggingface_hub`)
  - caminho local -> ex.:  set AIDEV_DATA=C:\\Users\\Ruth\\Downloads\\aidev

Uso:  python explore.py
"""

import os
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

DATA_DIR = os.environ.get("AIDEV_DATA", "hf://datasets/hao-li/AIDev")

TABELAS = [
    "all_pull_request.parquet",       # todos os PRs (rotulados por agente de IA)
    "all_repository.parquet",         # metadados dos repos (linguagem, estrelas...)
    "pull_request.parquet",           # subset AIDev-pop (repos > 100 estrelas)
    "pr_reviews.parquet",             # revisoes (APPROVED / CHANGES_REQUESTED...)
    "pr_review_comments_v2.parquet",  # comentarios inline de revisao
    "pr_task_type.parquet",           # tipo da tarefa (feat/fix/docs...) via LLM
    "human_pull_request.parquet",     # PRs de humanos (baseline)
]


def caminho(nome):
    return f"{DATA_DIR}/{nome}"


def main():
    print(f"[i] lendo de: {DATA_DIR}\n")
    for nome in TABELAS:
        try:
            df = pd.read_parquet(caminho(nome))
        except FileNotFoundError:
            print(f"[!] nao encontrado: {nome}\n")
            continue
        print("=" * 80)
        print(f"{nome}   ->   {len(df):,} linhas, {len(df.columns)} colunas")
        print("-" * 80)
        print("Colunas:", list(df.columns))
        print("-" * 80)
        print(df.head(3))
        print()


if __name__ == "__main__":
    main()
