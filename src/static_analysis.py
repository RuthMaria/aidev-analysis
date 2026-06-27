r"""
static_analysis.py - RQ5: qualidade estática intrínseca do código dos agentes.

Diferente das RQ1-RQ4 (que medem ACEITAÇÃO/esforço de revisão), aqui olhamos
o PRÓPRIO código submetido, rodando ferramentas de análise estática sobre os
patches da tabela pr_commit_details:

  - ruff   -> warnings de lint (smells / estilo) por linha adicionada
  - lizard -> complexidade ciclomática média (tolerante, multi-linguagem)
  - taxa de "parse" -> % de fragmentos que sequer compilam (sinal de qualidade)

LIMITAÇÃO (ver §8 do guia): o dataset traz só os PATCHES (diffs), não o arquivo
completo. Analisamos as LINHAS ADICIONADAS, que podem ser um fragmento sintático
incompleto. Por isso reportamos a taxa de parse e calculamos a densidade de
warnings apenas sobre fragmentos que compilam.

Começamos por Python (.py), a linguagem com melhor suporte de linter.

Uso (PowerShell):
  $env:AIDEV_DATA = "C:\Users\Ruth\Downloads\aidev"   # opcional (dados locais)
  python src/static_analysis.py

Uso (Git Bash):
  AIDEV_DATA="C:\Users\Ruth\Downloads\aidev" python src/static_analysis.py

A amostra é limitada por padrão (AIDEV_RQ5_SAMPLE arquivos .py) para a análise
ser rápida. Defina AIDEV_RQ5_SAMPLE=0 para rodar no dataset inteiro.
"""

import os
import json
import tempfile
import subprocess
from pathlib import Path

import pandas as pd
import lizard

# Reaproveita a infraestrutura já pronta da RQ1-RQ4 (mesma pasta src/).
from quality_analysis import path, OUTPUT_DIR, load_prs, bar_chart

# Quantos arquivos .py analisar. 0 = todos (lento). Padrão = amostra.
SAMPLE = int(os.environ.get("AIDEV_RQ5_SAMPLE", "5000"))


def added_lines(patch):
    """Extrai do diff unificado apenas as linhas ADICIONADAS (começam com '+').

    Ignora o cabeçalho '+++' e o prefixo '+' de cada linha, devolvendo o texto
    do código novo introduzido pelo commit.
    """
    if not isinstance(patch, str) or not patch:
        return ""
    out = []
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            out.append(line[1:])
    return "\n".join(out)


def run_ruff(fragments):
    """Roda o ruff em lote sobre os fragmentos e devolve métricas por índice.

    Escreve cada fragmento como <i>.py num diretório temporário, roda o ruff uma
    única vez (mais rápido que um processo por arquivo) e mapeia os resultados de
    volta pelo nome do arquivo. Para cada índice devolve:
      - n_warnings  : nº de avisos de lint (exclui erro de sintaxe)
      - syntax_error: True se o fragmento nem compila
    """
    tmp = Path(tempfile.mkdtemp(prefix="rq5_ruff_"))
    try:
        # Grava os fragmentos em disco com nome = índice (zero-padded).
        for i, code in fragments.items():
            (tmp / f"{i:08d}.py").write_text(code, encoding="utf-8")

        # Uma única chamada ao ruff, saída em JSON. O ruff retorna código != 0
        # quando encontra problemas; por isso não usamos check=True.
        proc = subprocess.run(
            ["ruff", "check", "--no-cache", "--output-format=json",
             "--select=ALL", str(tmp)],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",   # ruff emite UTF-8; evita cp1252
        )
        try:
            issues = json.loads(proc.stdout) if proc.stdout.strip() else []
        except json.JSONDecodeError:
            issues = []

        # Inicializa todo índice com zero warnings e sem erro de sintaxe.
        result = {i: {"n_warnings": 0, "syntax_error": False} for i in fragments.index}
        for it in issues:
            stem = Path(it.get("filename", "")).stem
            if not stem.isdigit():
                continue
            i = int(stem)
            # No ruff, erro de sintaxe vem sem 'code' (code == None).
            if it.get("code") is None:
                result[i]["syntax_error"] = True
            else:
                result[i]["n_warnings"] += 1
        return result
    finally:
        # Limpa o diretório temporário (apaga os .py gerados).
        for f in tmp.glob("*.py"):
            f.unlink()
        tmp.rmdir()


def mean_complexity(code):
    """Complexidade ciclomática média das funções do fragmento (via lizard).

    O lizard é tolerante a código parcial. Devolve NaN se não houver função.
    """
    try:
        info = lizard.analyze_file.analyze_source_code("frag.py", code)
    except Exception:
        return float("nan")
    if not info.function_list:
        return float("nan")
    ccns = [f.cyclomatic_complexity for f in info.function_list]
    return sum(ccns) / len(ccns)


def load_python_files():
    """Carrega pr_commit_details, filtra .py e anexa agente, tipo e merged.

    Devolve um DataFrame com uma linha por arquivo (.py) alterado em um PR,
    já com o código adicionado e os metadados do PR para agregação.
    """
    commits = pd.read_parquet(path("pr_commit_details.parquet"))
    # Só arquivos Python que de fato adicionaram código.
    py = commits[commits["filename"].str.endswith(".py", na=False)].copy()
    py = py[py["additions"].fillna(0) > 0]

    # Amostra para a análise ser rápida (SAMPLE=0 desliga o limite).
    if SAMPLE and len(py) > SAMPLE:
        py = py.sample(n=SAMPLE, random_state=42)
    py = py.reset_index(drop=True)

    # Extrai o código adicionado e conta as linhas adicionadas reais.
    py["code"] = py["patch"].map(added_lines)
    py["added_loc"] = py["code"].map(lambda c: c.count("\n") + 1 if c else 0)
    py = py[py["added_loc"] > 0].reset_index(drop=True)

    # Liga ao PR para descobrir o agente e se foi mesclado (RQ1).
    prs = load_prs("all_pull_request.parquet")[["id", "agent", "merged"]]
    py = py.merge(prs, left_on="pr_id", right_on="id", how="inner")

    # Tipo de tarefa (feat/fix/docs...) para o recorte por tipo.
    types = pd.read_parquet(path("pr_task_type.parquet"))[["id", "type"]]
    py = py.merge(types, left_on="pr_id", right_on="id", how="left",
                  suffixes=("", "_type"))
    return py


# Rótulos em português para EXIBIR as tabelas (iguais ao estilo do guia).
PT_LABELS = {
    "n_files": "#arquivos",
    "parse_rate": "% compila",
    "warnings_per_100loc": "Warnings/100 linhas",
    "mean_ccn": "Complexidade média",
}


def to_portuguese(table):
    """Renomeia colunas e o eixo para português, apenas para exibição."""
    return table.rename(columns=PT_LABELS).rename_axis("Agente")


def static_quality_by_agent(df):
    """Agrega as métricas estáticas por agente (RQ5)."""
    g = df.groupby("agent").agg(
        n_files=("code", "size"),                 # nº de arquivos analisados
        parse_rate=("syntax_error", lambda s: (1 - s.mean()) * 100),  # % que compila
    )
    # Densidade de warnings só sobre o que compila (exclui erro de sintaxe).
    ok = df[~df["syntax_error"]]
    dens = ok.groupby("agent").apply(
        lambda d: d["n_warnings"].sum() / d["added_loc"].sum() * 100,
        include_groups=False,
    ).rename("warnings_per_100loc")
    ccn = ok.groupby("agent")["mean_ccn"].mean().rename("mean_ccn")

    g = g.join(dens).join(ccn)
    return g.round(2).sort_values("warnings_per_100loc")


def main():
    print(f"[i] RQ5 - análise estática (amostra = {SAMPLE or 'todos'} arquivos .py)")
    print("[i] carregando patches de pr_commit_details ...")
    df = load_python_files()
    print(f"[i] {len(df):,} arquivos .py com código adicionado para analisar")

    print("[i] rodando ruff (lint) ...")
    ruff = run_ruff(df["code"])
    df["n_warnings"] = df.index.map(lambda i: ruff[i]["n_warnings"])
    df["syntax_error"] = df.index.map(lambda i: ruff[i]["syntax_error"])

    print("[i] rodando lizard (complexidade) ...")
    df["mean_ccn"] = df["code"].map(mean_complexity)

    print("\n===================== RQ5. QUALIDADE ESTÁTICA POR AGENTE =====================\n")
    table = static_quality_by_agent(df)
    print(to_portuguese(table))
    bar_chart(table["warnings_per_100loc"],
              "Densidade de warnings (ruff) por agente",
              "warnings por 100 linhas", "RQ5_static_warnings.png")

    print("\nInterprete: menor densidade de warnings e maior taxa de 'compila' "
          "sugerem código intrinsecamente mais limpo, independente da aceitação.")


if __name__ == "__main__":
    main()
