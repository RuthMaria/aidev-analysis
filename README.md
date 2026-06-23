# AIDev — Análise de qualidade de código gerado por agentes de IA

Scripts de análise sobre o dataset [**AIDev**](https://huggingface.co/datasets/hao-li/AIDev),
que reúne ~1 milhão de pull requests de agentes de programação com IA
(OpenAI Codex, Devin, GitHub Copilot, Cursor e Claude Code) em projetos
open-source do GitHub, além de PRs de humanos como baseline.

## O que estes scripts medem

O dataset não tem uma coluna "qualidade"; ela é estimada por **proxies**:

| Proxy | Sinal de qualidade |
|---|---|
| Taxa de merge | PR aceito pelos mantenedores |
| Taxa de rejeição | PR fechado sem merge |
| `CHANGES_REQUESTED` por PR | quanto retrabalho o código exigiu |
| Comentários inline de revisão | volume de problemas apontados |
| Comparação com humanos | baseline para todas as métricas |
| Tipo de tarefa (`feat`/`fix`/`docs`) | controle para comparações justas |

## Scripts

- **`explore.py`** — mostra colunas e amostras de cada tabela.
- **`quality_analysis.py`** — calcula as métricas acima e gera gráficos PNG.

## Como rodar

```bash
pip install pandas pyarrow matplotlib huggingface_hub
python explore.py
python quality_analysis.py
```

### Origem dos dados

Por padrão os scripts leem direto do Hugging Face (`hf://datasets/hao-li/AIDev`).
Se você já tem os `.parquet` baixados localmente, aponte para a pasta:

```bash
# Windows (PowerShell)
$env:AIDEV_DATA = "C:\Users\Ruth\Downloads\aidev"
python quality_analysis.py
```

## Créditos

Dataset AIDev: Hao Li, Haoxiang Zhang, Ahmed E. Hassan —
[arXiv:2507.15003](https://arxiv.org/abs/2507.15003).
Cada repositório de origem mantém sua licença original.
