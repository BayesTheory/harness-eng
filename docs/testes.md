# Testes

```bash
pytest -q          # 170 testes, ~5 segundos
```

**Nenhum deles chama um modelo.** Nenhum precisa de chave de API, de rede ou de transcript. Os SDKs `anthropic` e `openai` não precisam nem estar instalados — o CI tem um passo que **falha** se eles aparecerem no ambiente.

Isso é o que faz contribuir aqui não custar nada.

## Onde os testes estão

| Arquivo | Cobre |
|---|---|
| `test_core.py` | o loop, o executor de ferramenta, o trace nativo, o round-trip |
| `test_facade.py` | `@tool`, `Harness`, tradução de/para o formato *chat completions* |
| `test_policy.py` | os eixos, os níveis, a fronteira de política, a métrica de ajuste |
| `test_metrics.py` | as métricas sobre sessões construídas à mão |
| `test_stats.py` | a camada estatística contra implementação de referência |
| `test_layering.py` | as sete regras de dependência |

## Como o modelo é substituído

`ScriptedClient` responde de um roteiro fixo. Não é acessório de teste — é o que permite exercitar os modos de falha que **não dá para provocar sob demanda pagando por eles**:

```python
from harness_eng.core.clients import ScriptedClient
from harness_eng.core.ports import ModelResponse
from harness_eng.trace.model import StopReason, ToolCall

client = ScriptedClient([
    ModelResponse(
        tool_calls=(ToolCall(id="c1", name="soma", arguments={"a": 2, "b": 3}),),
        stop_reason=StopReason.TOOL_USE,
    ),
    ModelResponse(text="dá 5", stop_reason=StopReason.END_TURN),
])

h = Harness(client=client)
```

Duas propriedades importam:

- **Determinístico.** O mesmo roteiro produz o mesmo trace, então um teste sobre o trace testa o loop e não o humor do modelo.
- **`client.seen` guarda as conversas recebidas.** É o que permite verificar o que o loop **enviou** — inclusive que os resultados paralelos foram numa mensagem só, que é uma propriedade invisível se você só olhar o conteúdo.

Roteiro curto demais levanta `ModelError` em vez de devolver `end_turn`. É deliberado: devolver fim de turno faria um loop que devia estourar o teto terminar limpo, e o teste passaria medindo a coisa errada.

## Por que isto não é uma limitação

Um comportamento de borda que só aparece em produção é um comportamento que ninguém verifica. Os quatro modos de falha do loop — `pause_turn` tratado como fim, teto de iteração, resposta truncada executando ferramenta pela metade, resultados paralelos partidos — são **silenciosos**: dão errado sem levantar erro. Provocá-los com modelo real seria caro, lento e não-reprodutível.

Com roteiro, cada um vira um teste de dez linhas que roda em milissegundos.

## A lacuna

**`.complete()` dos clientes reais nunca é executado em teste nenhum.** Medido:

```
harness_eng/core/clients.py    169 linhas    67% de cobertura
```

As 55 linhas descobertas são justamente `AnthropicClient.complete()`, `OpenAIClient.complete()` e a construção dos clientes de verdade. Os testes tocam essas classes só por dois caminhos:

- `AnthropicClient(client=object())` — injeta um cliente falso, o construtor curto-circuita antes de importar o SDK
- monkeypatch de `facade.AnthropicClient` num teste da fachada

O que **está** coberto são as funções puras de tradução — `to_messages`, `from_message`, `to_openai_messages`, `from_completion`, `_openai_usage` — exercitadas com `SimpleNamespace`. É a parte onde os erros interessantes moram (argumento como string JSON, `prompt_tokens` já incluindo cache, `pause_turn` sobrevivendo à tradução), e ela está testada.

O que **não** está coberto é o fio que sai para a rede: montagem final do payload, streaming acima do limiar, tradução de exceção do SDK.

> Fechar isso é um dos pontos de entrada listados no [CONTRIBUTING.md](../CONTRIBUTING.md). O caminho provável é um cliente falso com a forma do SDK (não `object()`), verificando o payload que `complete()` monta — sem rede, mantendo a propriedade de o CI não precisar de chave.

Está escrito aqui porque um repositório que prega medição e esconde a própria lacuna de medição não vale nada.

## Convenções

**Teste nomeia o comportamento, não o método.** `test_teto_de_iteracoes_nao_e_sucesso`, não `test_max_iterations`. O nome deve dizer o que quebra quando ele fica vermelho.

**O docstring explica por que o teste existe**, com preferência por casos que já aconteceram:

```python
def test_pause_turn_nao_e_fim_de_turno() -> None:
    """
    O caso que este pacote não sabia representar até o harness precisar dele.

    ``pause_turn`` significa "pausei, me retome" — e um loop que trata tudo que não é
    ``tool_use`` como fim devolve trabalho pela metade **sem erro nenhum**.
    """
```

**Estatística é testada contra propriedade, não contra "roda sem levantar".** Código estatístico sutilmente errado produz número plausível, ninguém desconfia, e a conclusão errada circula. Então `test_stats.py` confere contra implementação de referência ingênua, contra casos com resposta conhecida, e contra a propriedade que define cada estatística — a cobertura de um IC de 95% é verificada rodando 200 repetições e contando.

**Relógio injetado, nunca `sleep`.** `AgentLoop(clock=...)` aceita um relógio determinístico. Duração medida sem dormir.

## Cobertura

```bash
pytest --cov=harness_eng --cov-report=term-missing
```

Não há meta de porcentagem, e não vai haver: número de cobertura vira alvo e para de medir o que deveria. O que importa é se o comportamento que quebra em produção tem teste — e a seção [A lacuna](#a-lacuna) é a resposta honesta a essa pergunta hoje.
