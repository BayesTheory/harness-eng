# Contribuindo

Obrigado pelo interesse. Este arquivo é curto de propósito — o material longo está em [`docs/`](docs/index.md).

## Três comandos

```bash
git clone https://github.com/BayesTheory/harness-eng
cd harness-eng
pip install -e ".[dev]"
pytest -q                 # 170 passam, ~5s
```

**Contribuir aqui não custa nada.** Nenhum dos 170 testes precisa de chave de API, de rede ou de transcript — os SDKs de provedor não precisam nem estar instalados. O modelo é substituído por um roteiro fixo, e é isso que permite testar os modos de falha que só apareceriam pagando. Ver [`docs/testes.md`](docs/testes.md).

Para experimentar o harness de ponta a ponta sem gastar:

```bash
harness-eng run "o que tem aqui?" --workspace harness_eng --level 1 --dry-run
harness-eng analyze reports/native
```

## Checklist de PR

- [ ] `pytest -q` passa (170, ou mais se você acrescentou)
- [ ] `ruff check .` sai limpo — é o que o CI roda
- [ ] comportamento novo tem teste; correção de bug tem teste que falharia antes
- [ ] se mexeu em camada, `pytest tests/test_layering.py` continua verde

O CI roda em Python 3.10, 3.11 e 3.12.

> Não rodamos `ruff format`. O repositório é formatado à mão, com alinhamento deliberado nos comentários `#:` de atributo e nas tabelas em docstring; o formatador reescreveria 28 dos 34 arquivos sem ganho.

## Onde encostar primeiro

Quatro buracos reais, bem delimitados, do menor para o maior:

**1. `--provider` e `--base-url` na CLI.** Hoje `_run()` instancia `AnthropicClient` fixo, então a CLI só fala com a Anthropic — enquanto o README anuncia "funciona com qualquer modelo". A porta `ModelClient` já existe e o `OpenAIClient` já está pronto e testado; é trabalho de composition root, umas 20 linhas mais teste. Junto vem ler o `OPENAI_API_KEY` que o `.env.example` promete e ninguém lê.

**2. Teste do `.complete()` real.** Os clientes de verdade nunca são executados em teste (`clients.py` em 67%). O caminho provável é um cliente falso com a forma do SDK, verificando o payload que `complete()` monta — sem rede. Ver [`docs/testes.md`](docs/testes.md#a-lacuna).

**3. Cliente Bedrock.** Para quem já tem acesso AWS. Entra ao lado dos dois existentes, sem tocar no loop nem nas métricas.

**4. `report/`.** O pacote existe vazio desde o scaffold; a ideia era relatório em HTML sobre o que `analyze` já calcula.

**Um adapter de trace novo** (LangGraph, Agents SDK, o seu harness) é a contribuição de maior alcance: transforma todas as métricas e toda a estatística em ferramentas para mais um ecossistema. Ver [`docs/adicionar-adapter.md`](docs/adicionar-adapter.md).

Abra uma issue antes de começar algo grande — as quatro acima mexem em pontos próximos e dá para colidir.

## Estilo

### Código

**Escreva o porquê, não o quê.** Comentário que diz "incrementa o contador" acima de `contador += 1` não entra. O que diz *por que* o contador precisa existir, sim.

Os docstrings de módulo aqui são longos (11 a 41 linhas) e carregam decisões — vários documentam o erro que a decisão consertou. Se a sua mudança tem uma razão não óbvia, ela pertence ao docstring.

**Ausência é ausência.** Devolva `None` para "não deu para medir", nunca `0`. Zero é uma medição; ausente não é, e confundir os dois envenena toda média que vier depois. Quem formata para o relatório é que trata o `None`.

### Commits

Assunto imperativo em minúscula, com escopo quando ajuda:

```
feat(core): harness minimo — loop, ferramentas e trace nativo
fix(trace): pause_turn e um stop reason real, e faltava no enum
docs: README reescrito para quem chega — uso antes de argumento
```

**O corpo explica por que, e o que foi medido.** Este repositório mede em vez de supor; o commit é onde a medição fica registrada. Se você trocou uma abordagem por outra, diga o número que sustentou a troca.

**Erro próprio se registra, não se esconde.** Vários commits aqui documentam bugs que o autor introduziu e os testes acharam — inclusive erros na própria camada de estatística e numa regra de arquitetura. Um repositório que prega medição e esconde os próprios erros de medição não vale nada, e isso vale para o histórico também.

## Documentação com número

Se você escrever um número em README ou docs, **execute o que o produziu antes de commitar**. Os exemplos de código do README foram todos rodados contra cliente falso antes de entrar, e a saída mostrada é a saída real. Número inventado num repositório sobre rigor de medição é a pior coisa que pode acontecer com ele.

## Dúvidas

Abra uma issue. Se a pergunta for "como isto funciona?", a resposta provavelmente pertence a [`docs/`](docs/index.md) — e transformá-la em PR de documentação também é contribuição.
