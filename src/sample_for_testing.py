r"""
sample_for_testing.py - RQ6 (passo 1): seleciona a amostra de PRs para os
testes dinâmicos (cobertura + mutação).

Cobertura e teste de mutação precisam EXECUTAR a suíte de testes em um repo
reconstruído — o que é pesado e instável. Por isso a RQ6 roda sobre uma AMOSTRA
curada de PRs. Este script faz a parte 100% minerável: escolhe os PRs e gera um
CSV com tudo que o runner (mutation_coverage.py) precisa para clonar, dar
checkout e medir.

Critérios de inclusão (framing B - quão bem-testado é o código do agente):
  - PR MESCLADO (o código entrou no projeto);
  - em repositório popular (subset AIDev-pop / pull_request.parquet);
  - que adiciona ao menos 1 arquivo .py de CÓDIGO (algo para cobrir/mutar); e
  - ao menos 1 arquivo .py de TESTE (indício de que a suíte existe).

Saída: outputs/test_sample.csv com uma linha por PR selecionado:
  agent, repo_url, pr_number, html_url, pr_id, head_sha,
  n_code_files, n_test_files, added_code_loc, code_files, test_files

Uso (Git Bash):
  AIDEV_DATA="C:\Users\Ruth\Downloads\aidev" python src/sample_for_testing.py

A amostra é estratificada por agente e limitada por AIDEV_TEST_SAMPLE
(padrão 25 = 5 por agente; use 0 para exportar todos os candidatos).
"""

import os                      # lê variáveis de ambiente (AIDEV_DATA, AIDEV_TEST_SAMPLE)
import re                      # expressões regulares p/ reconhecer arquivos de teste
import sys                     # usado para forçar a saída do console em UTF-8
import pandas as pd           # leitura dos .parquet e agregações

# Reaproveita o caminho dos dados e a pasta de saída já definidos na RQ1-RQ4.
from RQ1_to_RQ4_quality_analysis import path, OUTPUT_DIR

# Console do Windows usa cp1252; força UTF-8 para não quebrar com acentos/emojis.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Tamanho da amostra final (estratificada por agente). 0 = todos os candidatos.
# Padrão 25 (5 por agente), igual ao documentado no guia (§6.6).
SAMPLE = int(os.environ.get("AIDEV_TEST_SAMPLE", "25"))

# Regex que reconhece arquivos de TESTE pelas convenções comuns em Python:
#   - pasta tests/ ou test/ ou __tests__/      -> (^|/)(tests?|__tests__)/
#   - nome começando com test_                 -> (^|/)test_
#   - nome terminando em _test.py ou _tests.py -> _test\.py$ | _tests\.py$
TEST_RE = re.compile(r"(^|/)(tests?|__tests__)/|(^|/)test_|_test\.py$|_tests\.py$",
                     re.IGNORECASE)


def is_python(name):
    # Verdadeiro só se for string e terminar em .py (cobertura/mutação são de Python).
    return isinstance(name, str) and name.endswith(".py")


def is_test_file(name):
    # É arquivo de teste se for .py E o caminho/nome casar com a regex de teste.
    return is_python(name) and bool(TEST_RE.search(name))


def head_sha_by_pr():
    """Descobre o commit de topo (head) de cada PR via pr_timeline.

    Usa o último evento 'committed' (maior created_at) de cada pr_id — é a forma
    confiável de ordenar os commits, já que pr_commits não tem data.
    """
    tl = pd.read_parquet(path("pr_timeline.parquet"))          # carrega a linha do tempo dos PRs
    committed = tl[tl["event"].eq("committed")]                # mantém só eventos de commit
    committed = committed.dropna(subset=["commit_id"])         # descarta linhas sem SHA do commit
    committed = committed.sort_values("created_at")            # ordena do mais antigo ao mais novo
    # Após ordenar, .last() de cada PR = o commit mais recente (o head do PR).
    return committed.groupby("pr_id")["commit_id"].last().rename("head_sha")


def main():
    # Mensagem inicial mostrando o alvo de amostra escolhido.
    print(f"[i] selecionando amostra de PRs para RQ6 (alvo = {SAMPLE or 'todos'})")

    # 1. PRs mesclados do subset popular (AIDev-pop) -------------------------
    prs = pd.read_parquet(path("pull_request.parquet"))        # PRs de repos > 100 estrelas
    prs = prs[prs["merged_at"].notna()].copy()                 # mantém só os MESCLADOS
    prs = prs[["id", "agent", "number", "repo_url", "html_url"]]  # colunas que vamos usar

    # 2. Classifica os arquivos de cada PR (código .py vs teste .py) ---------
    commits = pd.read_parquet(                                  # detalhes por arquivo de cada commit
        path("pr_commit_details.parquet"),
        columns=["pr_id", "filename", "additions"],            # só as colunas necessárias (economia)
    )
    commits["is_py"] = commits["filename"].map(is_python)      # marca arquivos Python
    commits = commits[commits["is_py"]].copy()                 # descarta o que não é .py
    commits["is_test"] = commits["filename"].map(is_test_file) # marca quais são de teste
    commits["is_code"] = ~commits["is_test"]                   # o resto é código de produção

    # 3. Agrega por PR: listas de arquivos e linhas de código adicionadas ----
    def join_files(s):
        # Junta os nomes de arquivo num texto único, sem repetição e ordenado.
        return ";".join(sorted(set(s)))

    code = commits[commits["is_code"]]                         # subconjunto de arquivos de código
    test = commits[commits["is_test"]]                         # subconjunto de arquivos de teste
    per_pr = pd.DataFrame({                                     # uma linha por PR, com:
        "code_files": code.groupby("pr_id")["filename"].apply(join_files),   # lista de arquivos de código
        "n_code_files": code.groupby("pr_id")["filename"].nunique(),         # qtde de arquivos de código
        "added_code_loc": code.groupby("pr_id")["additions"].sum(),          # linhas de código adicionadas
        "test_files": test.groupby("pr_id")["filename"].apply(join_files),   # lista de arquivos de teste
        "n_test_files": test.groupby("pr_id")["filename"].nunique(),         # qtde de arquivos de teste
    })
    # Mantém só PRs que têm CÓDIGO e TESTE em Python (os dois são obrigatórios).
    per_pr = per_pr.dropna(subset=["code_files", "test_files"])

    # 4. Junta metadados do PR + head_sha -----------------------------------
    cand = prs.merge(per_pr, left_on="id", right_index=True, how="inner")        # PR + arquivos
    cand = cand.merge(head_sha_by_pr(), left_on="id", right_index=True, how="inner")  # + head_sha
    cand = cand.rename(columns={"id": "pr_id", "number": "pr_number"})           # nomes mais claros

    # Relatório dos candidatos antes de amostrar.
    print(f"[i] {len(cand):,} PRs candidatos (mesclado + código.py + teste.py + head_sha)")
    print("[i] candidatos por agente:")
    print(cand["agent"].value_counts().to_string())            # distribuição por agente

    # 5. Amostra estratificada por agente -----------------------------------
    # (para não ficar dominada pelo Codex, que tem muito mais PRs)
    if SAMPLE:                                                  # se houver limite (SAMPLE != 0)
        per_agent = max(1, SAMPLE // cand["agent"].nunique())  # cota igual por agente
        parts = [d.sample(min(len(d), per_agent), random_state=42)  # sorteia cada agente
                 for _, d in cand.groupby("agent")]            # random_state fixa p/ reprodutibilidade
        cand = pd.concat(parts)                                # junta as cotas de volta

    # Ordem das colunas no CSV de saída.
    cols = ["agent", "repo_url", "pr_number", "html_url", "pr_id", "head_sha",
            "n_code_files", "n_test_files", "added_code_loc",
            "code_files", "test_files"]
    cand = cand[cols].sort_values(["agent", "pr_number"])      # reordena linhas e colunas

    out = OUTPUT_DIR / "test_sample.csv"                       # caminho do arquivo de saída
    cand.to_csv(out, index=False, encoding="utf-8")           # grava o CSV
    print(f"\n[i] amostra final: {len(cand)} PRs -> {out}")    # confirma onde salvou
    print(cand[["agent", "repo_url", "pr_number", "n_code_files",  # prévia da amostra
                "n_test_files"]].to_string(index=False))


# Só executa main() quando o arquivo é rodado direto (não quando importado).
if __name__ == "__main__":
    main()
