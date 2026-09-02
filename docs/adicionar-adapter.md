# Adicionar um adapter de trace

Um adapter ensina o pacote a **medir um harness que ele ainda não conhece**. É a contribuição de maior alcance: cada adapter novo transforma todas as métricas e toda a camada estatística em ferramentas para mais um ecossistema, sem escrever métrica nenhuma.

Hoje existem dois: `claude_code` (lê `~/.claude/projects/**/*.jsonl`) e `native` (o formato que o harness deste repositório escreve).

## O contrato

```python
class TraceSource(Protocol):
    @property
    def name(self) -> str: ...

    def discover(self, root: Path) -> list[Path]: ...
    def load(self, path: Path) -> Session | None: ...
    def sessions(self, root: Path) -> Iterator[Session]: ...
```

Quatro regras que os dois adapters existentes seguem, e que valem mais que a assinatura:

**1. `load()` devolve `None` quando o arquivo não é seu.** Não levanta. Varrer um diretório heterogêneo é o caso normal — o carregador oferece cada arquivo a cada origem e a primeira que reconhece fica com ele. Uma exceção aqui derruba a varredura inteira por causa de um arquivo alheio.

**2. Discrimine por conteúdo, não por extensão.** `~/.claude/projects` também é cheio de `.jsonl`. O leitor nativo exige uma marca de formato na primeira linha; o do Claude Code simplesmente não encontra turno nenhum num arquivo nativo. Adivinhar pela extensão faz um leitor engolir transcript alheio e devolver sessão vazia em vez de `None`.

**3. Seja tolerante por arquivo, não por linha.** Sessão morta no meio deixa a última linha truncada — acontece sempre que alguém mata o processo. Recusar o arquivo inteiro por causa dela joga fora 1.668 turnos de dado bom.

**4. Conte o que você descartou.** Ambos os adapters têm um contador `skipped`:

```python
self.skipped: Counter[str] = Counter()
...
self.skipped["json inválido"] += 1
```

Um adapter que joga fora 30% das linhas está errado, e **sem esse contador ninguém descobre** — o relatório sai bonito, calculado sobre o pedaço que sobrou. `harness-eng analyze` imprime o total ao final justamente por isso.

## `sessions()` é iterador, não lista

Não é estilo: os 54 transcripts que motivaram o projeto somam 44.152 linhas, e a maior sessão sozinha tem 1.669 turnos. Materializar tudo antes de contar uma média é desperdício que cresce com o histórico do usuário — e histórico de agente só cresce.

## O trabalho de verdade é o mapeamento

O `Protocol` é fácil. O que consome tempo é traduzir o formato de origem para [`trace/model.py`](../harness_eng/trace/model.py) sem perder nem inventar informação.

Comece lendo o docstring de `ClaudeCodeSource`: o schema dele foi extraído **empiricamente** de 54 transcripts reais, com a contagem de cada tipo de registro. Essa é a abordagem recomendada — assinatura adivinhada passa em teste e quebra em execução.

### Quatro armadilhas já pagas

- **Nem todo conteúdo é texto.** A primeira versão do adapter do Claude Code lia só `type == "text"` e reportou *"58 de 60 chamadas de `ToolSearch` voltaram vazias"* — um achado alarmante sobre o harness que era defeito da ferramenta de medição. `ToolSearch` devolve blocos `tool_reference`; `Read` de imagem devolve `image`. Use `ToolResult.content_kinds` para separar *"não devolveu nada"* de *"não sei ler o que devolveu"*: são conclusões opostas.

- **Ausência é ausência.** Turno sem uso de token tem `usage=None`, não `Usage(0,0,0,0)`. Zero é uma medição; ausente não é.

- **Timestamp com fuso.** O adapter do Claude Code produz `datetime` com fuso; o harness nativo também. Misturar *aware* com *naive* levanta `TypeError` na primeira comparação — e a primeira comparação costuma ser um `min()`/`max()` numa métrica de duração, meses depois.

- **`stop_reason` desconhecido vira `UNKNOWN`, e isso é perigoso.** `pause_turn` não existia no enum e virava `UNKNOWN`; um harness que trata "não é `tool_use`" como "acabou" encerraria no meio registrando fim normal. Se o formato que você está lendo tem um motivo de parada que o enum não cobre, **acrescente ao enum** em vez de deixar cair em `UNKNOWN`.

## Teste: o round-trip

Para adapter que também **escreve** (um `TraceSink`), o teste mais valioso é `Session → disco → Session`, campo a campo:

```python
original = ...                                    # sessão montada à mão
path = MeuSink().write(original, tmp_path / "s.jsonl")
recuperada = MinhaSource().load(path)

assert [replace(t, raw={}) for t in original.turns] == list(recuperada.turns)
```

Se um campo se perde no caminho, o **formato canônico** tem um buraco — e é melhor descobrir com um teste de 40 linhas do que com um relatório que reporta zero onde havia dado. Ver `test_o_round_trip_nao_perde_campo` em [`tests/test_core.py`](../tests/test_core.py).

> `raw` fica de fora da comparação de propósito: um trace é registro de **medição**, não checkpoint retomável. Gravar o payload do provedor multiplicaria o arquivo por conteúdo que nenhuma métrica lê.

Para adapter só de leitura, os testes que importam são: um arquivo real reconhecido, um arquivo de outra origem recusado com `None`, e uma linha corrompida que não custa o arquivo inteiro.

## Registrando

Adapter novo entra em `_load()` na [`cli.py`](../harness_eng/cli.py), na tupla de origens, e no subcomando `sources`. As origens se excluem sozinhas pela regra 2, então a ordem não importa muito — mas coloque a mais específica primeiro.

## Uma nota sobre o valor disso

O formato canônico só **prova** ser comum quando alguém que não o escreveu tenta usá-lo. Foi exatamente ao escrever o segundo consumidor que apareceu o buraco do `pause_turn` — que não tinha aparecido em revisão de código nem nos 54 transcripts.

Se você escrever o terceiro adapter e ele **não** achar nenhum buraco, isso também é resultado, e vale registrar no PR.
