r"""
RQ6_step2_mutation_and_coverage.py - RQ6 (passo 2): mede COBERTURA e MUTAÇÃO do código dos
agentes sobre a amostra selecionada por RQ6_step1_sample_for_testing.py.

Para cada PR da amostra (outputs/RQ6_step1_test_sample.csv) o runner:
  1. Converte a URL da API do GitHub para URL de clone;
  2. Clona só o commit do PR (head_sha) e dá checkout;
  3. Cria um venv isolado e instala o projeto + ferramentas de teste;
  4. Roda a suíte com cobertura e calcula a COBERTURA das linhas que o agente
     adicionou (patch coverage), cruzando coverage.json com os patches;
  5. Roda o cosmic-ray nos arquivos do agente e calcula o MUTATION SCORE.

Cobertura = pytest-cov; Mutação = cosmic-ray (multiplataforma; o mutmut não roda
nativamente no Windows). Tudo é defensivo: cada PR pode falhar em qualquer etapa
(clone, instalação, testes) — registramos o status e seguimos. A tese reporta o
subconjunto que rodou.

Uso (Git Bash):
  AIDEV_DATA="C:\Users\Ruth\Downloads\aidev" python src/RQ6_step2_mutation_and_coverage.py

Variáveis de ambiente:
  AIDEV_RUN_LIMIT     quantos PRs processar (padrão: todos do CSV)
  AIDEV_MUT_MAX_FILES máx. de arquivos a mutar por PR (padrão 3; controla o tempo)
  AIDEV_MUT_TIMEOUT   timeout por mutante, em segundos (padrão 30)
  AIDEV_SKIP_MUTATION se definido, mede só cobertura (mutação é lenta)
"""

import os                       # variáveis de ambiente e caminhos
import re                       # parse dos cabeçalhos de hunk do diff
import sys                      # saída UTF-8 no console do Windows
import json                    # leitura do coverage.json e do dump do cosmic-ray
import shutil                  # remover diretórios temporários
import tempfile                # criar a pasta de trabalho (clones)
import subprocess              # rodar git, pip, pytest, cosmic-ray
from pathlib import Path        # manipular caminhos de forma portável

import pandas as pd            # ler o CSV da amostra e o pr_commit_details

from RQ1_to_RQ4_perceived_quality import path, OUTPUT_DIR, grouped_bar_chart  # caminhos + helper de gráfico

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # evita erro de cp1252

# --- Configuração via ambiente -------------------------------------------------
RUN_LIMIT = int(os.environ.get("AIDEV_RUN_LIMIT", "0"))          # 0 = todos
MUT_MAX_FILES = int(os.environ.get("AIDEV_MUT_MAX_FILES", "3"))  # limita mutação
MUT_TIMEOUT = float(os.environ.get("AIDEV_MUT_TIMEOUT", "30"))   # s por mutante
SKIP_MUTATION = bool(os.environ.get("AIDEV_SKIP_MUTATION"))      # pular mutação?

SAMPLE_CSV = OUTPUT_DIR / "RQ6_step1_test_sample.csv"                  # entrada (passo 1)
RESULTS_CSV = OUTPUT_DIR / "RQ6_step2_results.csv"              # saída por PR
LOGS_DIR = OUTPUT_DIR / "rq6_logs"                               # logs do pytest por PR

# Caminho do python dentro de um venv difere entre Windows e Unix.
VENV_PY = "Scripts/python.exe" if os.name == "nt" else "bin/python"

# Cabeçalho de hunk de diff unificado: captura o início do arquivo NOVO (+).
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def clone_url(api_url):
    """Converte 'https://api.github.com/repos/owner/name' em URL de clone .git."""
    owner_name = api_url.split("/repos/", 1)[1]        # pega 'owner/name'
    return f"https://github.com/{owner_name}.git"      # monta a URL de clone


def run(cmd, cwd=None, timeout=None):
    """Roda um comando e devolve (ok, saída). Nunca lança — captura tudo."""
    try:
        p = subprocess.run(cmd, cwd=cwd, timeout=timeout,           # executa
                           capture_output=True, text=True,          # captura stdout/err
                           encoding="utf-8", errors="replace")      # força UTF-8
        return p.returncode == 0, (p.stdout or "") + (p.stderr or "")  # ok? + log
    except Exception as e:                                          # timeout/erro de SO
        return False, str(e)


def added_lines_by_file(pr_id, commits):
    """Mapeia, para um PR, cada arquivo -> conjunto de LINHAS adicionadas (no novo).

    Lê os patches de pr_commit_details e percorre os hunks: linha '+' = adicionada;
    ' ' = contexto (avança); '-' = removida (não avança no arquivo novo).
    """
    out = {}                                                   # {filename: {linhas}}
    sub = commits[commits["pr_id"] == pr_id]                   # commits desse PR
    for _, row in sub.iterrows():                              # cada arquivo do PR
        patch = row["patch"]                                   # texto do diff
        fname = row["filename"]                                # caminho do arquivo
        if not isinstance(patch, str) or not fname.endswith(".py"):
            continue                                           # só .py com patch
        lines = out.setdefault(fname, set())                   # conjunto do arquivo
        new_line = None                                        # nº da linha atual (novo)
        for ln in patch.splitlines():                          # percorre o diff
            m = HUNK_RE.match(ln)                              # é cabeçalho de hunk?
            if m:                                              # sim:
                new_line = int(m.group(1))                     # reinicia o contador
            elif new_line is None:                             # antes do 1º hunk
                continue                                       # ignora
            elif ln.startswith("+"):                           # linha adicionada
                lines.add(new_line)                            # registra a linha
                new_line += 1                                  # avança no arquivo novo
            elif ln.startswith("-"):                           # linha removida
                pass                                           # não avança (não existe no novo)
            else:                                              # contexto
                new_line += 1                                  # avança no arquivo novo
    return out                                                 # mapa final


def patch_coverage(repo_dir, venv_py, added_map, log_path):
    """Cobertura SÓ das linhas adicionadas (patch coverage) -> (cobertas, total)."""
    cov_json = repo_dir / "cov.json"                           # arquivo de saída
    # Roda a suíte com cobertura, exportando o relatório em JSON.
    ok, out = run([str(venv_py), "-m", "pytest", "-q",         # pytest silencioso
                   "--cov", "--cov-report", f"json:{cov_json}"], # relatório JSON
                  cwd=repo_dir, timeout=900)                   # teto de 15 min
    # Salva o log do pytest p/ diagnóstico (mostra POR QUE falhou, se falhar).
    log_path.write_text(out, encoding="utf-8", errors="replace")
    if not cov_json.exists():                                  # sem relatório?
        return None                                            # cobertura indisponível
    data = json.loads(cov_json.read_text(encoding="utf-8"))    # lê o JSON
    files = data.get("files", {})                              # mapa de arquivos
    covered = total = 0                                        # acumuladores
    for fname, lines in added_map.items():                     # cada arquivo do agente
        base = os.path.basename(fname)                         # nome p/ casar caminhos
        # Acha a entrada de cobertura cujo caminho termina no arquivo do PR.
        match = next((v for k, v in files.items()
                      if k.replace("\\", "/").endswith(fname)
                      or os.path.basename(k) == base), None)
        if not match:                                          # arquivo não coberto/visto
            continue
        executed = set(match.get("executed_lines", []))        # linhas executadas
        missing = set(match.get("missing_lines", []))          # executáveis não executadas
        executable = lines & (executed | missing)              # linhas add. que contam
        covered += len(lines & executed)                       # add. cobertas
        total += len(executable)                               # add. executáveis
    return covered, total                                      # (cobertas, total)


def mutation_score(repo_dir, venv_py, code_files, log_path):
    """Mutation score dos arquivos do agente -> (mortos, sobreviventes)."""
    killed = survived = 0                                       # acumuladores
    logbuf = []                                                 # diagnóstico do cosmic-ray
    for fname in code_files[:MUT_MAX_FILES]:                    # limita p/ caber no tempo
        target = repo_dir / fname                               # arquivo a mutar
        if not target.exists():                                 # arquivo ausente?
            logbuf.append(f"### {fname}: arquivo não encontrado no checkout")
            continue
        cfg = repo_dir / "cr.toml"                              # config do cosmic-ray
        session = repo_dir / "cr.sqlite"                        # sessão (resultados)
        if session.exists():                                    # limpa sessão anterior
            session.unlink()
        # test-command usa o python do venv para achar as deps do projeto.
        cfg.write_text(
            "[cosmic-ray]\n"
            f'module-path = "{fname}"\n'                        # só este arquivo
            f"timeout = {MUT_TIMEOUT}\n"                        # timeout por mutante
            "excluded-modules = []\n"
            # as_posix() usa barras '/': em TOML, '\' é escape e quebra o caminho no Windows.
            f'test-command = "{Path(venv_py).as_posix()} -m pytest -x -q"\n'  # roda a suíte
            "\n[cosmic-ray.distributor]\n"
            'name = "local"\n', encoding="utf-8")
        cr = [str(venv_py), "-m", "cosmic_ray.cli"]            # CLI do cosmic-ray
        _, oi = run(cr + ["init", str(cfg), str(session)], cwd=repo_dir, timeout=120)   # cria sessão
        _, oe = run(cr + ["exec", str(cfg), str(session)], cwd=repo_dir, timeout=3600)  # roda mutantes
        ok, dump = run(cr + ["dump", str(session)], cwd=repo_dir, timeout=120)  # extrai resultados
        # Guarda a cauda da saída p/ diagnóstico (init costuma falhar se não houver mutações).
        logbuf.append(f"### {fname}\n[init]\n{oi[-400:]}\n[exec]\n{oe[-1000:]}")
        if not ok:                                             # falhou o dump?
            continue
        for line in dump.splitlines():                         # cada linha = 1 mutante
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)                         # formato: [job, result]
            except json.JSONDecodeError:
                continue
            # O desfecho fica em result["test_outcome"], em MINÚSCULAS.
            result = rec[1] if isinstance(rec, list) and len(rec) > 1 else {}
            outcome = result.get("test_outcome") if isinstance(result, dict) else None
            if outcome == "killed":                            # teste matou o mutante
                killed += 1
            elif outcome == "survived":                        # mutante sobreviveu
                survived += 1
    # Salva o diagnóstico do cosmic-ray (resumo + caudas de init/exec).
    log_path.write_text(f"killed={killed} survived={survived}\n\n" + "\n".join(logbuf),
                        encoding="utf-8", errors="replace")
    return killed, survived                                    # (mortos, sobreviventes)


def install_project(venv_py, repo_dir):
    """Instala o projeto e, se houver, suas dependências de TESTE.

    A maioria das falhas de 'tests_failed' vem de deps de teste ausentes: elas
    costumam ficar em extras (.[test]/.[dev]) ou num requirements-dev/test — não
    no install padrão. Tentamos todos; o que não existir falha em silêncio.
    """
    pip = [str(venv_py), "-m", "pip", "install", "-q"]
    run(pip + ["-e", "."], cwd=repo_dir, timeout=1800)         # projeto (editable)
    for extra in ("test", "tests", "testing", "dev"):          # extras comuns de teste
        run(pip + ["-e", f".[{extra}]"], cwd=repo_dir, timeout=1800)
    for req in ("requirements.txt", "requirements-dev.txt",    # arquivos de requirements
                "requirements-test.txt", "test-requirements.txt",
                "dev-requirements.txt"):
        if (repo_dir / req).exists():                          # só se o arquivo existir
            run(pip + ["-r", req], cwd=repo_dir, timeout=1800)
    # Ferramentas de teste no MESMO venv (para enxergar as deps do projeto).
    ok, _ = run(pip + ["pytest", "pytest-cov", "cosmic-ray"], timeout=600)
    return ok                                                  # ok = ferramentas instaladas


def process_pr(row, commits, workdir):
    """Processa um PR: clona, instala, mede cobertura e mutação. Devolve dict."""
    res = {"agent": row["agent"], "repo": clone_url(row["repo_url"]),  # base do resultado
           "pr_number": row["pr_number"], "status": "ok",
           "patch_cov": None, "mut_score": None, "reason": None}
    repo_dir = Path(workdir) / f"pr_{row['pr_id']}"            # pasta deste PR
    repo_dir.mkdir(parents=True, exist_ok=True)
    sha = row["head_sha"]                                      # commit a reconstruir
    url = clone_url(row["repo_url"])                           # URL de clone

    # 1. Clona apenas o commit do PR (rápido) e dá checkout.
    run(["git", "init", "-q"], cwd=repo_dir)                   # repo vazio
    run(["git", "remote", "add", "origin", url], cwd=repo_dir) # aponta para o GitHub
    ok, log = run(["git", "fetch", "--depth", "1", "origin", sha],  # busca só o sha
                  cwd=repo_dir, timeout=600)
    if not ok:                                                 # commit indisponível?
        res["status"] = "clone_failed"
        return res
    run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=repo_dir) # estado do PR

    # 2. Cria venv e instala o projeto + ferramentas de teste.
    run([sys.executable, "-m", "venv", str(repo_dir / ".venv")], timeout=120)
    venv_py = repo_dir / ".venv" / VENV_PY                     # python do venv
    run([str(venv_py), "-m", "pip", "install", "-q", "-U", "pip"], timeout=300)
    # Instala o projeto + deps de teste (extras/requirements).
    if not install_project(venv_py, repo_dir):
        res["status"] = "install_failed"
        return res

    # 3. Cobertura das linhas adicionadas pelo agente.
    added_map = added_lines_by_file(row["pr_id"], commits)     # linhas add. por arquivo
    log_path = LOGS_DIR / f"pr_{row['pr_id']}.log"             # log do pytest deste PR
    cov = patch_coverage(repo_dir, venv_py, added_map, log_path)
    if cov is None:                                            # suíte não rodou
        res["status"] = "tests_failed"
        res["reason"] = log_path.name                          # aponta o log p/ diagnóstico
        return res
    covered, total = cov
    res["patch_cov"] = round(100 * covered / total, 1) if total else None

    # 4. Mutação (opcional, pesada).
    if not SKIP_MUTATION:
        code_files = [f for f in str(row["code_files"]).split(";") if f.endswith(".py")]
        mut_log = LOGS_DIR / f"pr_{row['pr_id']}_mutation.log"  # diagnóstico do cosmic-ray
        killed, survived = mutation_score(repo_dir, venv_py, code_files, mut_log)
        if killed + survived:                                 # houve mutantes válidos?
            res["mut_score"] = round(100 * killed / (killed + survived), 1)
    return res


def plot_results(df):
    """Gera os gráficos da RQ6 reaproveitando grouped_bar_chart (RQ1-RQ4)."""
    # 1. Status por agente: quantos PRs rodaram (ok) vs. falharam.
    status = df.copy()
    status["ran"] = status["status"].eq("ok")                 # True se a suíte rodou
    by_agent = status.groupby("agent")["ran"].agg(Rodou="sum", total="size")
    by_agent["Falhou"] = by_agent["total"] - by_agent["Rodou"]
    by_agent = by_agent[["Rodou", "Falhou"]].sort_values("Rodou", ascending=False)
    by_agent.index = by_agent.index.str.replace("_", " ")     # rótulos legíveis
    grouped_bar_chart(by_agent, "RQ6: PRs que rodaram vs. falharam, por agente",
                      "nº de PRs", "RQ6_status_por_agente.png", legend_title="Status")

    # 2. Cobertura e mutação dos PRs que rodaram (um grupo por PR).
    ok = df[df["status"].eq("ok")].copy()
    if ok.empty:                                              # nada rodou -> sem 2º gráfico
        return
    ok["PR"] = ok["agent"].str.replace("_", " ") + " #" + ok["pr_number"].astype(str)
    metrics = ok.set_index("PR")[["patch_cov", "mut_score"]]
    metrics.columns = ["Cobertura", "Mutação"]               # nomes da legenda
    grouped_bar_chart(metrics, "RQ6: cobertura e mutação dos PRs que rodaram",
                      "%", "RQ6_cobertura_mutacao.png", legend_title="Métrica")


def main():
    print(f"\n =============== RQ6 runner - COBERTURA + MUTAÇÂO (mutação {'OFF' if SKIP_MUTATION else 'ON'}) ==============\n")
    if not SAMPLE_CSV.exists():                                # precisa do passo 1
        sys.exit(f"[!] amostra não encontrada: {SAMPLE_CSV}. Rode RQ6_step1_sample_for_testing.py antes.")
    sample = pd.read_csv(SAMPLE_CSV)                           # lê a amostra
    if RUN_LIMIT:                                              # limita nº de PRs?
        sample = sample.head(RUN_LIMIT)                        # pega só os primeiros N PRs
    LOGS_DIR.mkdir(exist_ok=True)                             # pasta dos logs do pytest

    # Carrega os patches só dos PRs da amostra (para a cobertura por linha).
    commits = pd.read_parquet(path("pr_commit_details.parquet"),
                              columns=["pr_id", "filename", "patch"])
    commits = commits[commits["pr_id"].isin(sample["pr_id"])]  # filtra a amostra

    workdir = tempfile.mkdtemp(prefix="rq6_")                  # pasta de trabalho
    rows = []                                                  # resultados por PR
    try:
        for _, row in sample.iterrows():                      # cada PR da amostra
            print(f"[i] {row['agent']:<12} {clone_url(row['repo_url'])} #{row['pr_number']} ...")
            r = process_pr(row, commits, workdir)             # processa o PR e devolve o resultado
            print(f"    -> status={r['status']} cobertura={r['patch_cov']} mutação={r['mut_score']}\n")
            rows.append(r)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)            # limpa os clones

    df = pd.DataFrame(rows)                                    # tabela de resultados
    df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")     # salva por PR
    print(f"\n -> Resultados por PR salvos em {RESULTS_CSV}")

    # Agrega por agente só os PRs que rodaram (status == ok).
    ok = df[df["status"] == "ok"]
    print("\n===================== RQ6. COBERTURA E MUTAÇÃO POR AGENTE =====================\n")
    if ok.empty:
        print("[!] nenhum PR rodou até o fim (clone/instalação/testes falharam).")
    else:
        agg = ok.groupby("agent").agg(
            n_prs=("pr_number", "size"),                       # quantos PRs rodaram
            cobertura_media=("patch_cov", "mean"),            # % linhas add. cobertas
            mutation_score_medio=("mut_score", "mean"),       # % mutantes mortos
        ).round(1)
        print(agg)
    # Relatório de status (quantos falharam em cada etapa).
    print("\nStatus da amostra: \n")
    print(df["status"].value_counts().to_string())
    # Onde diagnosticar as falhas de 'tests_failed'.
    print(f"\n[i] logs do pytest (por que cada PR falhou) em: {LOGS_DIR}")

    # Gera os gráficos da RQ6 a partir dos resultados.
    plot_results(df)


if __name__ == "__main__":
    main()
