# Adicionar uma ferramenta

**Uma ferramenta é uma função.** Sem classe base, sem registro global, sem JSON Schema escrito à mão.

```python
from harness_eng import Harness

h = Harness(model="claude-opus-5")

@h.tool
def buscar(termo: str, limite: int = 10) -> list[str]:
    """Busca no catálogo.

    Args:
        termo: o que procurar
        limite: máximo de resultados
    """
    return catalogo.query(termo)[:limite]
```

A função continua uma função: dá para chamar, testar e importar como sempre. O decorador só pendura `.spec` e `.handler` nela.

## O schema sai da assinatura

| Você escreve | O modelo recebe |
|---|---|
| `termo: str` | `{"type": "string"}`, obrigatório |
| `limite: int = 10` | `{"type": "integer"}`, opcional (tem default) |
| `peso: float \| None = None` | `{"type": "number"}`, opcional |
| `tags: list[str]` | `{"type": "array", "items": {"type": "string"}}` |
| `config: dict` | `{"type": "object"}` |
| primeira parte do docstring | descrição da ferramenta |
| linha `termo: o que procurar` | descrição do parâmetro |

Descrição de parâmetro é lida de duas convenções — `Args:` do Google e `:param x:` do reST — porque obrigar um formato só faria a maioria dos docstrings existentes não render nada.

**Por que derivar em vez de escrever:** schema à mão é uma segunda fonte de verdade. Ela sai de sincronia com a função na primeira vez que alguém renomeia um parâmetro, e o sintoma é o modelo mandando um argumento que a função não aceita — longe da causa.

## Três formas de registrar

```python
@h.tool                      # decorador da fachada: declara e registra
def soma(a: int, b: int) -> int: ...

h.add(soma, subtrai)         # função já existente (decorada ou crua)

registry.register(spec, handler)   # explícito: você traz o schema
```

`ToolRegistry.add()` aceita função crua e deriva na hora. `register()` é o escape para quando o schema não pode vir da assinatura.

## O contrato de erro

Isto é o que faz as métricas deste repositório significarem alguma coisa.

**Falha se sinaliza levantando exceção**, nunca devolvendo `"erro: ..."`. Um executor que transforma toda falha em string produz um trace onde nada nunca falha — e aí a taxa de erro por ferramenta, a métrica mais reveladora do pacote, mede zero em toda ferramenta.

```python
from harness_eng import ToolError

@h.tool
def ler_config(nome: str) -> str:
    """Lê um arquivo de configuração."""
    if not nome.endswith(".toml"):
        raise ToolError(f"só aceito .toml, recebi {nome}")   # falha esperada
    return Path(nome).read_text()                            # se explodir, também vira erro
```

| O que acontece | Como fica no trace |
|---|---|
| `raise ToolError(...)` | `is_error=True` — a ferramenta recusou corretamente |
| qualquer outra exceção | `is_error=True`, com o tipo preservado — bug do executor |
| `raise PolicyDenied(...)` | `is_error=True` **e contado à parte** — ver [níveis](#e-se-a-ferramenta-precisar-de-permissão) |
| retorno `""` ou `None` | sucesso **vazio** — falha silenciosa, contada separado |

`ToolRegistry.execute()` **nunca levanta**. Uma exceção que subisse dali mataria a sessão inteira por causa de uma chamada, perdendo junto o trace que explicaria o que houve.

### A mensagem vai para o modelo

Argumento faltando vira `"argumentos inválidos para soma(): missing a required argument: 'b'"` — texto que o modelo lê e corrige na chamada seguinte. Um `TypeError` cru não ajudaria ninguém.

### Retorno vazio é falha silenciosa

Uma ferramenta que "funciona" e não devolve nada deixa o modelo sem sinal para o próximo passo. É um jeito de o loop travar sem nunca registrar erro — e portanto sem aparecer em painel nenhum que só conte erro.

Por isso `content_kinds` fica vazio quando não há saída, o que faz `ToolResult.is_empty` valer `True`. Marcar `("text",)` num retorno vazio esconderia exatamente o que a métrica existe para achar.

## O que o retorno vira

| Você devolve | O modelo lê |
|---|---|
| `str` | como está |
| `dict` / `list` / `tuple` | JSON indentado (não `repr` do Python — aspas duplas e `null` o modelo lê melhor) |
| `int`, `float`, outros | `str(valor)` |
| `None` | string vazia → conta como falha silenciosa |

Saída acima de 20.000 caracteres é truncada, **e o corte é anunciado** no texto. Truncar é escolha de harness; deixar o modelo supor que leu o arquivo inteiro é bug.

## Falhas na hora de decorar

Erro de schema aparece na decoração, não na execução:

```python
@h.tool
def quebrado(a) -> str:      # ValueError: o parâmetro 'a' não tem anotação de tipo
    ...
```

Um schema adivinhado passa no teste, chega ao modelo com o tipo errado e só reaparece em produção como *"o modelo mandou string onde eu queria número"* — caro de diagnosticar, barato de prevenir.

Duas mensagens que você pode encontrar:

- **"não tem anotação de tipo"** — anote o parâmetro.
- **"não consegui resolver uma anotação"** — tipo declarado dentro de outra função, ou importado só sob `TYPE_CHECKING`. Com `from __future__ import annotations` as anotações viram string, e resolvê-las exige que o nome exista no módulo.

## E se a ferramenta precisar de permissão?

Ferramenta que toca rede, disco ou processo deve consultar a política **dentro do handler**, imediatamente antes do efeito:

```python
from harness_eng.core.toolkit import PolicyDenied

def handler(arguments):
    decision = policy.check_url(url)
    if not decision.allowed:
        raise PolicyDenied(decision)      # contado como negativa, não como falha
    ...
```

A checagem mora no handler e não na montagem do registro **de propósito**: o agente recebe a ferramenta, tenta, e aprende qual parede bateu — e a parede fica contada. Filtrar a ferramenta da lista esconderia a informação dos dois lados.

`PolicyDenied` é subclasse de `ToolError`, então o modelo vê erro normalmente. A diferença é na medição: negativa significa que a política funcionou, falha significa que algo quebrou. Somá-las produziria uma taxa de erro que **sobe quando você aperta a segurança**.

Ver [`core/toolkit.py`](../harness_eng/core/toolkit.py) para as três ferramentas que já fazem isso.

## Testando

Nenhum modelo envolvido — chame o registro direto:

```python
from harness_eng.core.tools import ToolRegistry
from harness_eng.trace.model import ToolCall

def test_soma():
    registry = ToolRegistry().add(soma)
    result = registry.execute(ToolCall(id="c1", name="soma", arguments={"a": 2, "b": 3}))
    assert result.content == "5"
    assert not result.is_error
```

Para testar como o **modelo** usaria a ferramenta, use `ScriptedClient` — ver [testes.md](testes.md).
