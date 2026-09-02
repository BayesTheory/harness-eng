# CLI ou API?

**As duas.** A CLI é composição em cima da API Python, não um caminho paralelo — tudo que ela faz existe como função importável.

## As duas superfícies

```bash
harness-eng analyze [dir]              # métricas sobre traces
harness-eng compare A.json B.json      # comparação pareada
harness-eng power [dir]                # tamanho de amostra
harness-eng run "tarefa" --level 2     # roda o agente
harness-eng sources                    # origens de trace disponíveis
```

```python
from harness_eng import Harness, tool, level, RESEARCHER
```

## Paridade

| Subcomando | Equivalente em Python |
|---|---|
| `analyze` | `metrics.tools.analyse_tools`, `metrics.loops.detect_loops`, `metrics.context.profile_context`, `metrics.cost.estimate_cost`, `metrics.policy.analyse_policy` + `trace.sources.*` |
| `compare` | `stats.compare.compare_paired` |
| `power` | `stats.design.required_pairs`, `stats.design.estimate_power` |
| `run` | `harness.Harness` (fachada) ou `core.loop.AgentLoop` (peça a peça) |

## Onde elas **não** se equivalem

Duas assimetrias reais, e as duas são da CLI para menos:

### 1. A CLI não registra ferramenta própria

`_run()` em [`cli.py`](../harness_eng/cli.py) monta o registro por `policy_registry()` ou `workspace_registry()` — só as ferramentas embutidas dos níveis (`read_file`, `list_dir`, `find_files`, `write_file`, `fetch_url`, `run_command`). **Não há entrada por linha de comando para uma função sua.**

```python
# só em Python
h = Harness(model="claude-opus-5")

@h.tool
def consultar_estoque(sku: str) -> int:
    """Quantas unidades há em estoque."""
    return banco.query(sku)
```

### 2. A CLI só fala com a Anthropic

`_run()` instancia `AnthropicClient` fixo. Não existe `--provider` nem `--base-url`. `--model` troca apenas o *nome do modelo dentro da Anthropic* — passar `gpt-5` ali envia a string "gpt-5" para a API da Anthropic e toma erro.

```python
# só em Python
from harness_eng.core.clients import OpenAIClient
h = Harness(client=OpenAIClient(model="llama3", base_url="http://localhost:11434/v1"))
```

> **Isto é uma lacuna conhecida, não uma decisão.** O README anuncia "funciona com qualquer modelo" logo no começo, e isso é verdade só para a API Python. Fechar essa diferença é um dos pontos de entrada listados no [CONTRIBUTING.md](../CONTRIBUTING.md) — a porta `ModelClient` já existe e o `OpenAIClient` já está pronto e testado, então é trabalho de composition root, não de arquitetura.

### Resumo

| | CLI | Python |
|---|---|---|
| Analisar traces existentes | ✅ | ✅ |
| Comparação pareada, análise de poder | ✅ | ✅ |
| Níveis, allowlist de domínio e comando | ✅ | ✅ |
| Rodar agente com **ferramenta própria** | ❌ | ✅ |
| Trocar de **provedor** | ❌ | ✅ |
| Compor pipeline próprio sobre as métricas | ❌ | ✅ |

## Quando usar qual

**CLI** quando o alvo é trace que já existe. `harness-eng analyze` mede os transcripts do Claude Code que você acumulou, sem instrumentar nada e sem escrever uma linha.

**Python** quando você está construindo o agente. É onde estão as ferramentas próprias, os provedores e a composição.

## A fachada não é o único caminho

`Harness` é um *composition root* — ela monta peças, não implementa nada. Tudo que ela faz continua possível na mão, e isso é de propósito: a fachada existe para o caminho comum ser curto, não para ser o único.

```python
# equivalente ao que Harness(level=1) monta por dentro
from harness_eng.core.clients import AnthropicClient
from harness_eng.core.loop import AgentLoop
from harness_eng.core.policy import READER
from harness_eng.core.toolkit import policy_registry
from harness_eng.trace.sources.native import NativeSink

loop = AgentLoop(
    AnthropicClient(model="claude-opus-5"),
    policy_registry(READER, Path(".")),
    policy=READER,
)
outcome = loop.run("...")
NativeSink().write(outcome.session, Path("trace.jsonl"))
```

Use a forma explícita quando precisar trocar uma peça que a fachada não expõe — um `TraceSink` seu, um relógio injetado para teste, um registro montado a partir de outra fonte.
