r"""
RQ1_to_RQ4_perceived_quality.py - Análise da qualidade de código gerado por agentes
de IA usando o dataset AIDev (RQ1 a RQ4).

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
  - caminho local -> aponte AIDEV_DATA para a pasta com os .parquet (veja abaixo)

Uso (PowerShell):
  $env:AIDEV_DATA = "C:\Users\Ruth\Downloads\aidev"   # opcional (dados locais)
  python src/RQ1_to_RQ4_perceived_quality.py

Uso (Git Bash):
  AIDEV_DATA="C:\Users\Ruth\Downloads\aidev" python src/RQ1_to_RQ4_perceived_quality.py

Gera tabelas no terminal e gráficos PNG na pasta outputs/.
"""

import os                       # acessa variáveis de ambiente (ex.: AIDEV_DATA)
import sys                       # usado para forçar a saída do console em UTF-8
from pathlib import Path        # monta caminhos de forma portável (Windows/Linux)
import pandas as pd            # biblioteca central: lê os .parquet, faz joins e agrega

# O console do Windows usa cp1252 e quebra com caracteres fora desse conjunto.
# Forçar UTF-8 evita UnicodeEncodeError ao imprimir (substitui o que não der show).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# matplotlib só é necessário para os gráficos PNG. Como é opcional, tentamos
# importar dentro de um try/except: se não estiver instalado, o script ainda
# roda e apenas pula a parte visual (as tabelas continuam saindo no terminal).
try:
    import matplotlib
    matplotlib.use("Agg")          # backend "Agg": salva PNG sem precisar de tela/janela
    import matplotlib.pyplot as plt
    HAS_PLOT = True                 # flag: temos como desenhar gráficos
except ImportError:
    HAS_PLOT = False               # sem matplotlib -> gráficos serão ignorados
    print("[i] matplotlib não instalado -> gráficos serão pulados "
          "(pip install matplotlib)")

# Configurações de exibição do pandas no terminal:
pd.set_option("display.max_columns", None)  # mostra TODAS as colunas (não corta)
pd.set_option("display.width", 160)         # largura maior antes de quebrar linha

# De onde ler os dados. Se a variável de ambiente AIDEV_DATA existir, usa o
# caminho local apontado por ela; senão, lê direto do Hugging Face via hf://.
DATA_DIR = os.environ.get("AIDEV_DATA", "hf://datasets/hao-li/AIDev")

# Pasta onde os gráficos PNG são salvos: <raiz-do-projeto>/outputs.
# Path(__file__) é este arquivo; .parent.parent sobe de src/ para a raiz.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)             # cria a pasta se ainda não existir


def path(name):
    # Monta o caminho completo de uma tabela juntando a pasta-base + nome do arquivo.
    return f"{DATA_DIR}/{name}"


def load_prs(name, agent_label=None):
    """Carrega uma tabela de PRs e garante as colunas que usamos."""
    df = pd.read_parquet(path(name))       # lê o .parquet para um DataFrame
    if agent_label is not None:            # PRs humanos não têm coluna 'agent';
        df["agent"] = agent_label          # então criamos uma com rótulo fixo (ex.: "Human")
    # Deriva o proxy de ACEITAÇÃO (RQ1): se 'merged_at' tem data, o PR foi mesclado.
    # .notna() devolve True/False linha a linha.
    df["merged"] = df["merged_at"].notna()
    # Deriva o proxy de REJEIÇÃO (RQ1): fechado (closed_at preenchido) E nunca
    # mesclado (merged_at vazio). O '&' combina as duas condições elemento a elemento.
    df["rejected"] = df["closed_at"].notna() & df["merged_at"].isna()
    # EM ABERTO (RQ1): nem mesclado nem fechado -> ainda sem decisão. Torna
    # explícita a 3ª categoria, para que mesclado + rejeitado + aberto = 100%.
    df["open"] = df["closed_at"].isna() & df["merged_at"].isna()
    return df                              # devolve o DataFrame já enriquecido


# ---------------------------------------------------------------------------
# Intervalo de confiança de Wilson para uma proporção
# ---------------------------------------------------------------------------
def wilson_interval(successes, n, z=1.96):
    """Intervalo de confiança de Wilson (95% por padrão) para uma proporção.

    Por que Wilson e não a fórmula "normal" (p ± z·sqrt(p(1-p)/n))? Porque os
    volumes por agente são MUITO desiguais (ver §7): com n enorme (Codex, 814k) o
    intervalo é estreitíssimo; com n pequeno (Claude Code, ~5k) ele é largo. O
    intervalo de Wilson permanece válido mesmo com n pequeno ou p perto de 0/1,
    onde a aproximação normal falha. Recebe o nº de sucessos (PRs mesclados) e o
    total n; devolve (limite_inferior, limite_superior) como proporções (0..1).
    """
    if n == 0:                              # sem PRs não há intervalo a calcular
        return float("nan"), float("nan")
    p = successes / n                       # proporção observada (taxa de merge)
    denom = 1 + z**2 / n                    # denominador comum da fórmula de Wilson
    center = (p + z**2 / (2 * n)) / denom   # centro do intervalo (ajustado)
    # Margem (meia-largura) do intervalo; usa **0.5 para a raiz quadrada.
    margin = (z / denom) * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5
    return center - margin, center + margin  # (inferior, superior) em proporção


# ---------------------------------------------------------------------------
# 1 + 2. Aceitação / rejeição por agente
# ---------------------------------------------------------------------------
def acceptance_by_agent(df):
    # Agrupa por agente e calcula três métricas por grupo (RQ1):
    g = df.groupby("agent").agg(
        n_prs=("id", "size"),              # quantos PRs cada agente tem (contagem)
        n_merged=("merged", "sum"),        # nº de PRs mesclados (sucessos p/ o IC)
        merge_rate=("merged", "mean"),     # média de True/False = proporção mesclada
        rejection_rate=("rejected", "mean"),# média de True/False = proporção rejeitada
        open_rate=("open", "mean"),        # proporção ainda em aberto (sem decisão)
    )
    # Intervalo de confiança de Wilson (95%) da taxa de merge, por agente. Torna
    # explícita a INCERTEZA desigual entre agentes com volumes muito diferentes
    # (§7): a banda é estreita para o Codex (n enorme) e larga para o Claude Code.
    ci = g.apply(lambda r: wilson_interval(r["n_merged"], r["n_prs"]), axis=1)
    g["merge_ci_low"] = (ci.str[0] * 100).round(1)   # limite inferior do IC (%)
    g["merge_ci_high"] = (ci.str[1] * 100).round(1)  # limite superior do IC (%)
    # Converte as proporções (0..1) para percentual (0..100) e arredonda a 1 casa.
    # As três taxas somam ~100% (mesclado + rejeitado + em aberto).
    g["merge_rate"] = (g["merge_rate"] * 100).round(1)
    g["rejection_rate"] = (g["rejection_rate"] * 100).round(1)
    g["open_rate"] = (g["open_rate"] * 100).round(1)
    # Remove a coluna auxiliar (n_merged) e reordena para o IC ficar ao lado da
    # taxa de merge — assim cada % vem acompanhada da sua faixa de incerteza.
    g = g[["n_prs", "merge_rate", "merge_ci_low", "merge_ci_high",
           "rejection_rate", "open_rate"]]
    # Ordena do agente mais aceito para o menos aceito.
    return g.sort_values("merge_rate", ascending=False)


# ---------------------------------------------------------------------------
# 3. Esforço de revisão
# ---------------------------------------------------------------------------
def review_effort(df_pr):
    """Junta PRs (subset AIDev-pop) com revisões e comentários inline.

    Usa 'pull_request.parquet' (não 'all_') porque só o subset AIDev-pop
    tem as tabelas de revisão.
    """
    # --- Parte A: revisões formais (tabela pr_reviews) ---
    reviews = pd.read_parquet(path("pr_reviews.parquet"))
    # Marca cada revisão que pediu mudanças (estado == "CHANGES_REQUESTED").
    reviews["changes_requested"] = reviews["state"].eq("CHANGES_REQUESTED")
    # Agrega as revisões por PR: quantas revisões teve e quantas pediram mudança.
    reviews_per_pr = reviews.groupby("pr_id").agg(
        n_reviews=("id", "size"),                       # nº de revisões do PR
        n_changes_requested=("changes_requested", "sum"),# nº de "CHANGES_REQUESTED"
    )

    # --- Parte B: comentários inline (tabela pr_review_comments_v2) ---
    comments = pd.read_parquet(path("pr_review_comments_v2.parquet"))
    # Os comentários só referenciam a REVISÃO (pull_request_review_id), não o PR.
    # Para descobrir o pr_id de cada comentário, criamos um "mapa" id->pr_id a
    # partir de pr_reviews, renomeando a chave para casar com a coluna dos comentários.
    rev_map = reviews[["id", "pr_id"]].rename(columns={"id": "pull_request_review_id"})
    # Junta o pr_id em cada comentário (left join: mantém todos os comentários).
    comments = comments.merge(rev_map, on="pull_request_review_id", how="left")
    # Conta quantos comentários inline cada PR recebeu.
    comments_per_pr = comments.groupby("pr_id").size().rename("n_inline_comments")

    # --- Parte C: junta tudo de volta nos PRs ---
    # Indexa os PRs por 'id' e mantém só agente + merged, para depois cruzar por pr_id.
    base = df_pr.set_index("id")[["agent", "merged"]]
    # join usa o índice (id do PR = pr_id) para anexar as contagens de A e B.
    base = base.join(reviews_per_pr).join(comments_per_pr)
    # PRs sem nenhuma revisão/comentário ficam como NaN após o join -> viram 0.
    base[["n_reviews", "n_changes_requested", "n_inline_comments"]] = (
        base[["n_reviews", "n_changes_requested", "n_inline_comments"]].fillna(0)
    )

    # Média de cada métrica por agente = esforço médio de revisão por PR (RQ2).
    return base.groupby("agent").agg(
        mean_reviews=("n_reviews", "mean"),
        mean_changes_requested=("n_changes_requested", "mean"),
        mean_inline_comments=("n_inline_comments", "mean"),
    ).round(2)


# ---------------------------------------------------------------------------
# 5. Controle por tipo de tarefa
# ---------------------------------------------------------------------------
def acceptance_by_type(df_pr):
    # Lê a classificação de tipo de tarefa (feat/fix/docs...) feita por LLM.
    types = pd.read_parquet(path("pr_task_type.parquet"))[["id", "type"]]
    # inner join: mantém apenas PRs que têm tipo classificado.
    m = df_pr.merge(types, on="id", how="inner")
    # Para cada combinação (agente, tipo), calcula a taxa de merge em % (RQ3).
    tab = m.groupby(["agent", "type"])["merged"].mean().mul(100).round(1)
    # unstack vira os 'type' em colunas -> tabela agente (linhas) x tipo (colunas).
    return tab.unstack(fill_value=float("nan"))


def bar_chart(series, title, ylabel, file):
    if not HAS_PLOT:                 # sem matplotlib, não há o que desenhar
        return
    # Desenha um gráfico de barras a partir de uma Series (índice = eixo X).
    ax = series.plot(kind="bar", figsize=(8, 5), color="#4C78A8")
    ax.set_title(title)             # título do gráfico
    ax.set_ylabel(ylabel)          # rótulo do eixo Y
    ax.set_xlabel("")              # sem rótulo no eixo X (os nomes já aparecem)
    plt.xticks(rotation=30, ha="right")  # inclina os rótulos para não sobrepor
    plt.tight_layout()             # ajusta margens para nada ficar cortado
    out = OUTPUT_DIR / file        # caminho final dentro de outputs/
    plt.savefig(out, dpi=120)      # salva o PNG no disco
    plt.close()                    # libera a figura da memória
    print(f"\n-> gráfico salvo em {out}")


def grouped_bar_chart(table, title, ylabel, file, legend_title="Categoria"):
    """Gráfico de barras AGRUPADAS para uma tabela 2D (linhas x colunas).

    Genérico: cada linha do DataFrame vira um grupo no eixo X e cada coluna uma
    barra dentro do grupo. Usado na RQ3 (agente x tipo de tarefa) e reaproveitado
    pela RQ6 (status por agente; cobertura/mutação por PR). 'legend_title' nomeia
    a legenda conforme o que as colunas representam.
    """
    if not HAS_PLOT:               # sem matplotlib, não há o que desenhar
        return
    # DataFrame.plot(kind="bar") já agrupa por linha (índice) e cria uma barra
    # por coluna; figsize maior para caber a legenda.
    ax = table.plot(kind="bar", figsize=(10, 6), colormap="viridis")
    ax.set_title(title)            # título do gráfico
    ax.set_ylabel(ylabel)          # rótulo do eixo Y
    ax.set_xlabel("")              # sem rótulo no eixo X (os rótulos das linhas já aparecem)
    ax.legend(title=legend_title, bbox_to_anchor=(1.0, 1.0))  # legenda fora da área
    plt.xticks(rotation=30, ha="right")  # inclina os rótulos do eixo X
    plt.tight_layout()             # ajusta margens para nada ficar cortado
    out = OUTPUT_DIR / file        # caminho final dentro de outputs/
    plt.savefig(out, dpi=120)      # salva o PNG no disco
    plt.close()                    # libera a figura da memória
    print(f"\n-> gráfico salvo em {out}")


# Rótulos em português para EXIBIR as tabelas (iguais ao guia de pesquisa).
# As colunas internas continuam em inglês; só traduzimos na hora de imprimir.
PT_LABELS = {
    "n_prs": "#PRs",
    "merge_rate": "% mesclado",
    "merge_ci_low": "IC95% inf.",
    "merge_ci_high": "IC95% sup.",
    "rejection_rate": "% rejeitado",
    "open_rate": "% em aberto",
    "mean_reviews": "Média revisões/PR",
    "mean_changes_requested": "Média CHANGES_REQUESTED/PR",
    "mean_inline_comments": "Média comentários inline/PR",
}


def to_portuguese(table, columns_name=None):
    """Renomeia colunas e eixos para português, apenas para exibição."""
    t = table.rename(columns=PT_LABELS).rename_axis("Agente")
    if columns_name is not None:        # ex.: RQ3 também nomeia o eixo das colunas
        t = t.rename_axis(columns=columns_name)
    return t


def main():    
    # ---- RQ1: carrega TODOS os PRs de agentes e mede aceitação/rejeição ----
    print("\n========================= RQ1. ACEITAÇÃO / REJEIÇÃO POR AGENTE =========================\n")
    agents = load_prs("all_pull_request.parquet")  # Carrega uma tabela de PRs e garante as colunas que usamos (aceitação/rejeição/open).
    tab_acc = acceptance_by_agent(agents) # calcula taxa de aceitação/rejeição/open/IC por agente
    print(to_portuguese(tab_acc))         # imprime a tabela (rótulos em português)
    bar_chart(tab_acc["merge_rate"],      # e salva o gráfico de taxa de merge
              "Taxa de merge por agente (%)", "% mesclado",
              "RQ1_taxa_merge.png")

    # ---- RQ2: esforço de revisão (só existe no subset AIDev-pop) ----
    print("\n\n============================ RQ2. ESFORÇO DE REVISÃO POR AGENTE ============================\n")
    try:
        pop = load_prs("pull_request.parquet")  # PRs de repos populares
        tab_review = review_effort(pop) # calcula média de revisões, CHANGES_REQUESTED e comentários inline por PR
        print(to_portuguese(tab_review))
        bar_chart(tab_review["mean_changes_requested"],
                  "Média de revisões 'CHANGES_REQUESTED' por PR",
                  "média por PR", "RQ2_changes_requested.png")
    except FileNotFoundError as e:
        # Se as tabelas de revisão não existirem, apenas avisa e segue em frente.
        print(f"[!] pulei esforço de revisão: {e}")

    # ---- RQ3: taxa de merge por agente e tipo de tarefa (feat/fix/docs...) ----
    print("\n\n============================= RQ3. TAXA DE MERGE POR TIPO DE TAREFA (%) =============================\n")
    try:
        tab_type = acceptance_by_type(agents)         # calcula taxa de merge por agente e tipo de tarefa
        print(to_portuguese(tab_type, columns_name="tipo"))
        grouped_bar_chart(tab_type,                   # barras agrupadas por agente
                          "Taxa de merge por tipo de tarefa (%)",
                          "% mesclado", "RQ3_taxa_merge_por_tipo.png",
                          legend_title="Tipo de tarefa")
    except FileNotFoundError as e:
        print(f"[!] pulei controle por tipo: {e}")

    # ---- RQ4: adiciona o baseline humano e compara na mesma métrica ----
    print("\n\n========================= RQ4. AGENTES vs HUMANOS (taxa de merge) =========================\n")
    humans = load_prs("human_pull_request.parquet", agent_label="Human")
    # concat empilha os dois DataFrames; ignore_index renumera as linhas.
    combined = pd.concat([agents, humans], ignore_index=True)

    # Reaproveita a mesma função de RQ1, agora com 'Human' incluso.
    tab_humans = acceptance_by_agent(combined)[["n_prs", "merge_rate"]]
    print(to_portuguese(tab_humans))
    # Gráfico da taxa de merge com o baseline 'Human' na mesma escala dos agentes.
    bar_chart(tab_humans["merge_rate"],
              "Taxa de merge: agentes vs. humanos (%)", "% mesclado",
              "RQ4_agentes_vs_humanos.png")

    # Conclusão: síntese das quatro questões de pesquisa (qualidade PERCEBIDA).
    print("\nInterprete (RQ1-RQ4): a aceitação varia muito entre agentes (RQ1) e "
          "acompanha menos retrabalho de revisão (RQ2); cai nas tarefas que mexem na "
          "lógica (RQ3); e os melhores já se equiparam aos humanos (RQ4). É qualidade "
          "percebida (o que o mantenedor aceita) — a intrínseca fica nas RQ5/RQ6.")
    # print("\n===================== CONCLUSÃO (RQ1-RQ4) =====================\n")
    # print(
    #     "RQ1 (aceitação): a taxa de merge varia muito entre agentes (~59% a ~88%).\n"
    #     "     'Não mesclado' não é 'rejeitado' — distinga recusa (rejeitado) de\n"
    #     "     abandono (em aberto), que pesa em alguns agentes.\n"
    #     "RQ2 (esforço de revisão): quem é mais aceito tende a gerar menos retrabalho\n"
    #     "     (menos 'changes_requested' e menos comentários inline por PR).\n"
    #     "RQ3 (tipo de tarefa): para TODOS os agentes a aceitação cai nas tarefas que\n"
    #     "     mexem na lógica (feat/fix/refactor) vs. docs/build — a lacuna de\n"
    #     "     qualidade aparece justamente onde o código é mais crítico.\n"
    #     "RQ4 (vs. humanos): os melhores agentes já são competitivos com o baseline\n"
    #     "     humano (~76,8% de aceitação); os demais ficam abaixo.\n\n"
    #     "Em conjunto: estas RQs medem qualidade PERCEBIDA (o que o mantenedor aceita),\n"
    #     "não corretude. Leia-as juntas e controle o tipo de tarefa antes de comparar\n"
    #     "agentes; a qualidade intrínseca do código é avaliada na RQ5 (estática) e na\n"
    #     "RQ6 (testes)."
    # )


# Só executa main() quando o arquivo é rodado direto (não quando importado).
if __name__ == "__main__":
    main()
