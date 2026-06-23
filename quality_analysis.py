"""
quality_analysis.py - Analise da qualidade de codigo gerado por agentes de IA
usando o dataset AIDev.

O dataset NAO tem uma coluna "qualidade". Aqui usamos PROXIES extraidos das
tabelas:

  1. Taxa de aceitacao (merge)  -> proxy mais direto: o codigo foi aceito?
  2. Taxa de rejeicao           -> fechado sem merge
  3. Esforco de revisao         -> % de revisoes pedindo mudancas + nº de
                                   comentarios inline por PR
  4. Comparacao com humanos     -> mesma metrica para human_pull_request
  5. Controle por tipo de tarefa-> junta com pr_task_type (feat/fix/docs...)

Origem dos dados (variavel de ambiente AIDEV_DATA):
  - nao definida -> le direto do Hugging Face (precisa de `huggingface_hub`)
  - caminho local -> ex.:  set AIDEV_DATA=C:\\Users\\Ruth\\Downloads\\aidev

Uso:  python quality_analysis.py
Gera tabelas no terminal e graficos PNG na pasta atual.
"""

import os
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")          # salva PNG sem precisar de tela
    import matplotlib.pyplot as plt
    TEM_PLOT = True
except ImportError:
    TEM_PLOT = False
    print("[i] matplotlib nao instalado -> graficos serao pulados "
          "(pip install matplotlib)")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

DATA_DIR = os.environ.get("AIDEV_DATA", "hf://datasets/hao-li/AIDev")


def caminho(nome):
    return f"{DATA_DIR}/{nome}"


def carregar_prs(nome, rotulo_agente=None):
    """Carrega uma tabela de PRs e garante as colunas que usamos."""
    df = pd.read_parquet(caminho(nome))
    if rotulo_agente is not None:          # PRs humanos nao tem coluna 'agent'
        df["agent"] = rotulo_agente
    # 'merged_at' preenchido => PR foi mesclado (aceito)
    df["merged"] = df["merged_at"].notna()
    # fechado sem merge => rejeitado
    df["rejected"] = df["closed_at"].notna() & df["merged_at"].isna()
    return df


# ---------------------------------------------------------------------------
# 1 + 2. Aceitacao / rejeicao por agente
# ---------------------------------------------------------------------------
def aceitacao_por_agente(df):
    g = df.groupby("agent").agg(
        n_prs=("id", "size"),
        taxa_merge=("merged", "mean"),
        taxa_rejeicao=("rejected", "mean"),
    )
    g["taxa_merge"] = (g["taxa_merge"] * 100).round(1)
    g["taxa_rejeicao"] = (g["taxa_rejeicao"] * 100).round(1)
    return g.sort_values("taxa_merge", ascending=False)


# ---------------------------------------------------------------------------
# 3. Esforco de revisao
# ---------------------------------------------------------------------------
def esforco_de_revisao(df_pr):
    """Junta PRs (subset AIDev-pop) com revisoes e comentarios inline.

    Usa 'pull_request.parquet' (nao 'all_') porque so o subset AIDev-pop
    tem as tabelas de revisao.
    """
    reviews = pd.read_parquet(caminho("pr_reviews.parquet"))
    # % de revisoes que pediram mudancas, por PR
    reviews["changes_requested"] = reviews["state"].eq("CHANGES_REQUESTED")
    rev_por_pr = reviews.groupby("pr_id").agg(
        n_reviews=("id", "size"),
        n_changes_requested=("changes_requested", "sum"),
    )

    comments = pd.read_parquet(caminho("pr_review_comments_v2.parquet"))
    # pr_review_comments_v2 liga pela review; trazemos o pr_id via reviews
    rev_map = reviews[["id", "pr_id"]].rename(columns={"id": "pull_request_review_id"})
    comments = comments.merge(rev_map, on="pull_request_review_id", how="left")
    com_por_pr = comments.groupby("pr_id").size().rename("n_inline_comments")

    base = df_pr.set_index("id")[["agent", "merged"]]
    base = base.join(rev_por_pr).join(com_por_pr)
    base[["n_reviews", "n_changes_requested", "n_inline_comments"]] = (
        base[["n_reviews", "n_changes_requested", "n_inline_comments"]].fillna(0)
    )

    return base.groupby("agent").agg(
        media_reviews=("n_reviews", "mean"),
        media_changes_requested=("n_changes_requested", "mean"),
        media_comentarios_inline=("n_inline_comments", "mean"),
    ).round(2)


# ---------------------------------------------------------------------------
# 5. Controle por tipo de tarefa
# ---------------------------------------------------------------------------
def aceitacao_por_tipo(df_pr):
    tipos = pd.read_parquet(caminho("pr_task_type.parquet"))[["id", "type"]]
    m = df_pr.merge(tipos, on="id", how="inner")
    tab = m.groupby(["agent", "type"])["merged"].mean().mul(100).round(1)
    return tab.unstack(fill_value=float("nan"))


def grafico_barras(serie, titulo, ylabel, arquivo):
    if not TEM_PLOT:
        return
    ax = serie.plot(kind="bar", figsize=(8, 5), color="#4C78A8")
    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(arquivo, dpi=120)
    plt.close()
    print(f"   -> grafico salvo em {arquivo}")


def main():
    print(f"[i] lendo de: {DATA_DIR}")
    print("\n### Carregando PRs de agentes (all_pull_request) ...")
    agentes = carregar_prs("all_pull_request.parquet")

    print("\n========== 1+2. ACEITACAO / REJEICAO POR AGENTE ==========")
    tab_ace = aceitacao_por_agente(agentes)
    print(tab_ace)
    grafico_barras(tab_ace["taxa_merge"],
                   "Taxa de merge por agente (%)", "% mesclado",
                   "q_taxa_merge.png")

    print("\n### Carregando baseline humano (human_pull_request) ...")
    humanos = carregar_prs("human_pull_request.parquet", rotulo_agente="Human")
    combinado = pd.concat([agentes, humanos], ignore_index=True)

    print("\n========== 4. AGENTES vs HUMANOS (taxa de merge) ==========")
    print(aceitacao_por_agente(combinado)[["n_prs", "taxa_merge"]])

    print("\n### Esforco de revisao (subset AIDev-pop) ...")
    try:
        pop = carregar_prs("pull_request.parquet")
        print("\n========== 3. ESFORCO DE REVISAO POR AGENTE ==========")
        tab_rev = esforco_de_revisao(pop)
        print(tab_rev)
        grafico_barras(tab_rev["media_changes_requested"],
                       "Media de revisoes 'CHANGES_REQUESTED' por PR",
                       "media por PR", "q_changes_requested.png")
    except FileNotFoundError as e:
        print(f"[!] pulei esforco de revisao: {e}")

    print("\n========== 5. TAXA DE MERGE POR TIPO DE TAREFA (%) ==========")
    try:
        print(aceitacao_por_tipo(agentes))
    except FileNotFoundError as e:
        print(f"[!] pulei controle por tipo: {e}")

    print("\nPronto. Interprete: maior taxa de merge e menor 'changes_requested' "
          "sugerem PRs de maior qualidade percebida pelos mantenedores.")


if __name__ == "__main__":
    main()
