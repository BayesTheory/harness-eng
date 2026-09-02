# Arquitetura

```
harness_eng/
├── harness.py       a porta da frente: Harness, Run, quick
├── cli.py           os 5 subcomandos
│
├── trace/           o vocabulário comum entre harnesses
│   ├── model.py     Session, Turn, ToolCall, ToolResult, Usage  ← PURO
│   ├── ports.py     TraceSource, TraceSink                      ← PURO
│   └── sources/     claude_code · native · (o seu)
│
├── metrics/         recebem dado, devolvem número               ← PURO
│   ├── tools.py     erro, retry, falha silenciosa, chamada sem resposta
│   ├── loops.py     repetição, retry cego, oscilação
│   ├── context.py   crescimento, concentração, cache
│   ├── cost.py      custo por modelo, por sessão, por chamada
│   └── policy.py    o nível serviu? concedeu demais? de menos?
│
├── stats/           pareado, bootstrap, efeito, poder           ← PURO
│
└── core/            o harness
    ├── ports.py     ModelClient, ModelResponse, ToolSpec
    ├── loop.py      o loop e os seis desfechos
    ├── tools.py     registro, @tool, derivação de schema
    ├── toolkit.py   as ferramentas de cada nível
    ├── policy.py    os eixos, os graus, a contagem de negativas
    └── clients.py   Anthropic · formato OpenAI · Scripted
```

## A regra, em uma frase

**`core/` importa `trace/`. `trace/`, `metrics/` e `stats/` não sabem que `core/` existe.**

A seta aponta num sentido só. Inverter é tentador — o harness tem o `ToolSpec`, e uma métrica sobre descrição de ferramenta ficaria "natural" importando dele. No dia em que isso acontecesse, medir um harness de **terceiro** passaria a arrastar o loop, o cliente de modelo e o SDK junto, e a promessa de rodar a suíte sem chave de API morreria sem que nenhum teste reclamasse.

## O que "puro" significa aqui

Os módulos em `PURE_MODULES` — `trace/model.py`, `trace/ports.py`, `metrics/` e `stats/` — recebem dado e devolvem número. **Não conhecem provedor, não tocam disco, não abrem rede.**

Isso não é purismo. É o que permite:

- montar uma `Session` à mão num teste, indistinguível de uma lida do disco
- testar toda a camada de medição sem transcript, sem chave e sem instalar nada além do `pytest`
- rodar `metrics/policy.py` sobre trace de **qualquer** harness que grave as chaves certas, inclusive um que não seja este

## As sete regras verificadas

Todas em [`tests/test_layering.py`](../tests/test_layering.py). Cada uma existe porque protege algo concreto:

| Regra | O que protege |
|---|---|
| `test_pure_layer_has_no_heavy_dependency` | Camada pura não importa `anthropic`, `openai`, `httpx`, `scipy`, `numpy`, `pandas`, `django`… (16 pacotes). É o que mantém `dependencies = []`. |
| `test_pure_layer_does_not_touch_the_filesystem` | Nem `open(`, nem `.read_text(`, nem `os.environ`. Uma métrica que lê arquivo sozinha não é testável sem arquivo — é assim que suíte rápida vira suíte com fixture no disco. |
| `test_the_canonical_model_knows_no_specific_harness` | `trace/model.py` não pode ter caso especial de nenhuma origem. No momento em que o vocabulário comum carrega um caso especial, deixa de ser comum e o segundo adapter passa a lutar contra ele. |
| `test_metrics_do_not_import_sources` | Métrica fala com o formato canônico, nunca com um leitor concreto. |
| `test_stats_do_not_import_trace` | A estatística recebe `Sequence[float]` e `Mapping[str, float]`. Isso a torna reutilizável para qualquer comparação pareada, e testável contra distribuições sintéticas com resposta conhecida. |
| `test_nothing_measurable_depends_on_the_harness` | A seta acima. |
| `test_the_harness_talks_to_one_provider_in_one_place` | Só `core/clients.py` importa SDK de provedor. Se o loop importar `anthropic` direto, a porta vira decoração e o próximo adapter vira reescrita em vez de acréscimo. |

## Duas lições já pagas por essas regras

**Regra de arquitetura precisa ser capaz de falhar.** Uma regra que nunca falhou não protege nada — pode estar quebrada e ninguém saber. As sete foram verificadas plantando uma violação de propósito e conferindo que o teste ficava vermelho. Se você acrescentar uma oitava, faça o mesmo antes de confiar nela.

**Grep de texto dá falso positivo, e teste que dá falso positivo é desligado numa semana.** Aconteceu duas vezes neste arquivo:

1. A primeira versão de `test_the_canonical_model_knows_no_specific_harness` grepava o arquivo inteiro e **reprovou o próprio docstring**, que citava `claude_code` como exemplo legítimo de valor de `source`. Trocada por análise de AST que separa identificador de código de texto de docstring.

2. A regra `test_nothing_measurable_depends_on_the_harness` nasceu com o mesmo defeito — e o primeiro módulo a citar `harness_eng.core` num docstring, **justamente para dizer que não o importa**, foi reprovado. Trocada por AST, que pega as duas formas de escrever a dependência (`from harness_eng.core.x import` e `from ..core.x import`).

A lição estava escrita no arquivo, para a regra 1, quando a regra 6 a repetiu.

## Onde a pureza **não** é exigida

`trace/sources/` lê disco por definição. `cli.py` e `core/` falam com o mundo. A pureza é exigida onde ela compra alguma coisa — impô-la em todo lugar viraria cerimônia.

## O formato canônico é lossy para replay, de propósito

Uma decisão que confunde à primeira leitura. `ModelResponse.replay_content` carrega os blocos crus do provedor, e o loop os guarda em `Turn.raw` — mas o trace gravado no disco **não os persiste**.

Motivo: um bloco de pensamento carrega assinatura do provedor. Reconstruí-lo a partir do texto canônico entrega um bloco sem assinatura, que a API descarta. Ou seja:

- **medição** quer o que dá para comparar entre provedores
- **replay** quer fidelidade de byte com um provedor específico

Forçar um tipo a servir aos dois estraga os dois. Daí o campo opaco, que o loop carrega sem olhar dentro e o sink deliberadamente ignora — um trace é registro de medição, não checkpoint retomável.

## Antes de mexer numa camada

```bash
pytest tests/test_layering.py -q     # 29 testes, roda em 0,1s
```

Se ele ficar vermelho, a pergunta certa não é "como faço o teste passar" — é "a seta deveria mesmo apontar para lá?". Nas duas vezes em que a resposta foi "não", o conserto foi no teste, e está documentado acima.
