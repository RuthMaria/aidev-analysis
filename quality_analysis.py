"""
quality_analysis.py - Análise da qualidade de código gerado por agentes de IA
usando o dataset AIDev.

O dataset NAO tem uma coluna "qualidade". Aqui usamos PROXIES extraídos das
tabelas:

  1. Taxa de aceitação (merge)  -> proxy mais direto: o código foi aceito?
  2. Taxa de rejeição           -> fechado sem merge
  3. Esforço de revisão         -> % de revisões pedindo mudanças + nº de
                                   comentários inline por PR
  4. Comparação com humanos     -> mesma métrica para human_pull_request
  5. Controle por tipo de tarefa-> junta com pr_task_type (feat/fix/docs...)

Origem dos dados (variável de ambiente AIDEV_DATA):
  - não definida -> lê direto do Hugging Face (precisa de `huggingface_hub`)
  - caminho local -> ex.:  set AIDEV_DATA=C:\\Users\\Ruth\\Downloads\\aidev

Uso:  python quality_analysis.py
Gera tabelas no terminal e gráficos PNG na pasta atual.
"""

import os                # acessa variáveis de ambiente (ex.: AIDEV_DATA)
import pandas as pd       # biblioteca central: lê os .parquet, faz joins e agrega

# matplotlib só é necessário para os gráficos PNG. Como é opcional, tentamos
# importar dentro de um try/except: se não estiver instalado, o script ainda
# roda e apenas pula a parte visual (as tabelas continuam saindo no terminal).
try:
    import matplotlib
    matplotlib.use("Agg")          # backend "Agg": salva PNG sem precisar de tela/janela
    import matplotlib.pyplot as plt
    TEM_PLOT = True                 # flag: temos como desenhar gráficos
except ImportError:
    TEM_PLOT = False               # sem matplotlib -> gráficos serão ignorados
    print("[i] matplotlib nao instalado -> graficos serao pulados "
          "(pip install matplotlib)")

# Configurações de exibição do pandas no terminal:
pd.set_option("display.max_columns", None)  # mostra TODAS as colunas (não corta)
pd.set_option("display.width", 160)         # largura maior antes de quebrar linha

# De onde ler os dados. Se a variável de ambiente AIDEV_DATA existir, usa o
# caminho local apontado por ela; senão, lê direto do Hugging Face via hf://.
DATA_DIR = os.environ.get("AIDEV_DATA", "hf://datasets/hao-li/AIDev")


def caminho(nome):
    # Monta o caminho completo de uma tabela juntando a pasta-base + nome do arquivo.
    return f"{DATA_DIR}/{nome}"


def carregar_prs(nome, rotulo_agente=None):
    """Carrega uma tabela de PRs e garante as colunas que usamos."""
    df = pd.read_parquet(caminho(nome))    # lê o .parquet para um DataFrame
    if rotulo_agente is not None:          # PRs humanos não têm coluna 'agent';
        df["agent"] = rotulo_agente        # então criamos uma com rótulo fixo (ex.: "Human")
    # Deriva o proxy de ACEITAÇÃO (RQ1): se 'merged_at' tem data, o PR foi mesclado.
    # .notna() devolve True/False linha a linha.
    df["merged"] = df["merged_at"].notna()
    # Deriva o proxy de REJEIÇÃO (RQ1): fechado (closed_at preenchido) E nunca
    # mesclado (merged_at vazio). O '&' combina as duas condições elemento a elemento.
    df["rejected"] = df["closed_at"].notna() & df["merged_at"].isna()
    return df                              # devolve o DataFrame já enriquecido


# ---------------------------------------------------------------------------
# 1 + 2. Aceitação / rejeição por agente
# ---------------------------------------------------------------------------
def aceitacao_por_agente(df):
    # Agrupa por agente e calcula três métricas por grupo (RQ1):
    g = df.groupby("agent").agg(
        n_prs=("id", "size"),              # quantos PRs cada agente tem (contagem)
        taxa_merge=("merged", "mean"),     # média de True/False = proporção mesclada
        taxa_rejeicao=("rejected", "mean"),# média de True/False = proporção rejeitada
    )
    # Converte as proporções (0..1) para percentual (0..100) e arredonda a 1 casa.
    g["taxa_merge"] = (g["taxa_merge"] * 100).round(1)
    g["taxa_rejeicao"] = (g["taxa_rejeicao"] * 100).round(1)
    # Ordena do agente mais aceito para o menos aceito.
    return g.sort_values("taxa_merge", ascending=False)


# ---------------------------------------------------------------------------
# 3. Esforço de revisão
# ---------------------------------------------------------------------------
def esforco_de_revisao(df_pr):
    """Junta PRs (subset AIDev-pop) com revisões e comentários inline.

    Usa 'pull_request.parquet' (não 'all_') porque só o subset AIDev-pop
    tem as tabelas de revisão.
    """
    # --- Parte A: revisões formais (tabela pr_reviews) ---
    reviews = pd.read_parquet(caminho("pr_reviews.parquet"))
    # Marca cada revisão que pediu mudanças (estado == "CHANGES_REQUESTED").
    reviews["changes_requested"] = reviews["state"].eq("CHANGES_REQUESTED")
    # Agrega as revisões por PR: quantas revisões teve e quantas pediram mudança.
    rev_por_pr = reviews.groupby("pr_id").agg(
        n_reviews=("id", "size"),                       # nº de revisões do PR
        n_changes_requested=("changes_requested", "sum"),# nº de "CHANGES_REQUESTED"
    )

    # --- Parte B: comentários inline (tabela pr_review_comments_v2) ---
    comments = pd.read_parquet(caminho("pr_review_comments_v2.parquet"))
    # Os comentários só referenciam a REVISÃO (pull_request_review_id), não o PR.
    # Para descobrir o pr_id de cada comentário, criamos um "mapa" id->pr_id a
    # partir de pr_reviews, renomeando a chave para casar com a coluna dos comentários.
    rev_map = reviews[["id", "pr_id"]].rename(columns={"id": "pull_request_review_id"})
    # Junta o pr_id em cada comentário (left join: mantém todos os comentários).
    comments = comments.merge(rev_map, on="pull_request_review_id", how="left")
    # Conta quantos comentários inline cada PR recebeu.
    com_por_pr = comments.groupby("pr_id").size().rename("n_inline_comments")

    # --- Parte C: junta tudo de volta nos PRs ---
    # Indexa os PRs por 'id' e mantém só agente + merged, para depois cruzar por pr_id.
    base = df_pr.set_index("id")[["agent", "merged"]]
    # join usa o índice (id do PR = pr_id) para anexar as contagens de A e B.
    base = base.join(rev_por_pr).join(com_por_pr)
    # PRs sem nenhuma revisão/comentário ficam como NaN após o join -> viram 0.
    base[["n_reviews", "n_changes_requested", "n_inline_comments"]] = (
        base[["n_reviews", "n_changes_requested", "n_inline_comments"]].fillna(0)
    )

    # Média de cada métrica por agente = esforço médio de revisão por PR (RQ2).
    return base.groupby("agent").agg(
        media_reviews=("n_reviews", "mean"),
        media_changes_requested=("n_changes_requested", "mean"),
        media_comentarios_inline=("n_inline_comments", "mean"),
    ).round(2)


# ---------------------------------------------------------------------------
# 5. Controle por tipo de tarefa
# ---------------------------------------------------------------------------
def aceitacao_por_tipo(df_pr):
    # Lê a classificação de tipo de tarefa (feat/fix/docs...) feita por LLM.
    tipos = pd.read_parquet(caminho("pr_task_type.parquet"))[["id", "type"]]
    # inner join: mantém apenas PRs que têm tipo classificado.
    m = df_pr.merge(tipos, on="id", how="inner")
    # Para cada combinação (agente, tipo), calcula a taxa de merge em % (RQ4).
    tab = m.groupby(["agent", "type"])["merged"].mean().mul(100).round(1)
    # unstack vira os 'type' em colunas -> tabela agente (linhas) x tipo (colunas).
    return tab.unstack(fill_value=float("nan"))


def grafico_barras(serie, titulo, ylabel, arquivo):
    if not TEM_PLOT:                 # sem matplotlib, não há o que desenhar
        return
    # Desenha um gráfico de barras a partir de uma Series (índice = eixo X).
    ax = serie.plot(kind="bar", figsize=(8, 5), color="#4C78A8")
    ax.set_title(titulo)            # título do gráfico
    ax.set_ylabel(ylabel)          # rótulo do eixo Y
    ax.set_xlabel("")              # sem rótulo no eixo X (os nomes já aparecem)
    plt.xticks(rotation=30, ha="right")  # inclina os rótulos para não sobrepor
    plt.tight_layout()             # ajusta margens para nada ficar cortado
    plt.savefig(arquivo, dpi=120)  # salva o PNG no disco
    plt.close()                    # libera a figura da memória
    print(f"   -> grafico salvo em {arquivo}")


def main():
    print(f"[i] lendo de: {DATA_DIR}")   # mostra a origem dos dados (local ou HF)

    # ---- RQ1: carrega TODOS os PRs de agentes e mede aceitação/rejeição ----
    print("\n### Carregando PRs de agentes (all_pull_request) ...")
    agentes = carregar_prs("all_pull_request.parquet")

    print("\n========== 1+2. ACEITACAO / REJEICAO POR AGENTE ==========")
    tab_ace = aceitacao_por_agente(agentes)
    print(tab_ace)                        # imprime a tabela no terminal
    grafico_barras(tab_ace["taxa_merge"], # e salva o gráfico de taxa de merge
                   "Taxa de merge por agente (%)", "% mesclado",
                   "q_taxa_merge.png")

    # ---- RQ3: adiciona o baseline humano e compara na mesma métrica ----
    print("\n### Carregando baseline humano (human_pull_request) ...")
    humanos = carregar_prs("human_pull_request.parquet", rotulo_agente="Human")
    # concat empilha os dois DataFrames; ignore_index renumera as linhas.
    combinado = pd.concat([agentes, humanos], ignore_index=True)

    print("\n========== 4. AGENTES vs HUMANOS (taxa de merge) ==========")
    # Reaproveita a mesma função de RQ1, agora com 'Human' incluso.
    print(aceitacao_por_agente(combinado)[["n_prs", "taxa_merge"]])

    # ---- RQ2: esforço de revisão (só existe no subset AIDev-pop) ----
    print("\n### Esforco de revisao (subset AIDev-pop) ...")
    try:
        pop = carregar_prs("pull_request.parquet")  # PRs de repos populares
        print("\n========== 3. ESFORCO DE REVISAO POR AGENTE ==========")
        tab_rev = esforco_de_revisao(pop)
        print(tab_rev)
        grafico_barras(tab_rev["media_changes_requested"],
                       "Media de revisoes 'CHANGES_REQUESTED' por PR",
                       "media por PR", "q_changes_requested.png")
    except FileNotFoundError as e:
        # Se as tabelas de revisão não existirem, apenas avisa e segue em frente.
        print(f"[!] pulei esforco de revisao: {e}")

    # ---- RQ4: taxa de merge controlada por tipo de tarefa ----
    print("\n========== 5. TAXA DE MERGE POR TIPO DE TAREFA (%) ==========")
    try:
        print(aceitacao_por_tipo(agentes))
    except FileNotFoundError as e:
        print(f"[!] pulei controle por tipo: {e}")

    # Lembrete de interpretação dos resultados.
    print("\nPronto. Interprete: maior taxa de merge e menor 'changes_requested' "
          "sugerem PRs de maior qualidade percebida pelos mantenedores.")


# Só executa main() quando o arquivo é rodado direto (não quando importado).
if __name__ == "__main__":
    main()
