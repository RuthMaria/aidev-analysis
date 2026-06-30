r"""
explore.py - Primeiro contato com o dataset AIDev.

Mostra, para cada tabela principal, o tamanho, as colunas e algumas linhas.
Rode primeiro este script para entender o que tem em cada arquivo.

Origem dos dados (variavel de ambiente AIDEV_DATA):
  - nao definida -> le direto do Hugging Face (precisa de `huggingface_hub`)
  - caminho local -> aponte AIDEV_DATA para a pasta com os .parquet (veja abaixo)

Uso (PowerShell):
  $env:AIDEV_DATA = "C:\Users\Ruth\Downloads\aidev"   # opcional (dados locais)
  python src/explore.py

Uso (Git Bash):
  AIDEV_DATA="C:\Users\Ruth\Downloads\aidev" python src/explore.py
"""

import os
import sys
import pandas as pd

# O console do Windows usa cp1252 e quebra ao imprimir emojis presentes nos
# titulos/corpos dos PRs. Forca a saida em UTF-8 (substitui o que nao der show).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

DATA_DIR = os.environ.get("AIDEV_DATA", "hf://datasets/hao-li/AIDev")

TABLES = [
    "all_pull_request.parquet",       # todos os PRs (rotulados por agente de IA)
    "pull_request.parquet",           # subset AIDev-pop (repos > 100 estrelas)
    "pr_reviews.parquet",             # revisoes (APPROVED / CHANGES_REQUESTED...)
    "pr_review_comments_v2.parquet",  # comentarios inline de revisao
    "pr_task_type.parquet",           # tipo da tarefa (feat/fix/docs...) via LLM
    "human_pull_request.parquet",     # PRs de humanos (baseline)
    "pr_commit_details.parquet",      # detalhes dos commits (RQ5)
    "pr_timeline.parquet"            # linha do tempo dos PRs
]


def path(name):
    return f"{DATA_DIR}/{name}"


def main():
    for index, name in enumerate(TABLES, start=1):
        try:
            df = pd.read_parquet(path(name))
        except FileNotFoundError:
            print(f"[!] não encontrado: {name}\n")
            continue
        print(f"\n ======================================= {index}º TABELA SELECIONADA =======================================\n")
        print(f"{name}   ->   {len(df):,} linhas, {len(df.columns)} colunas \n")
        print("COLUNAS:", list(df.columns), "\n")
        print(df.head(1))
        print()


if __name__ == "__main__":
    main()
