# harness-eng

**Rode agentes de IA — e saiba o que eles fizeram.**

Escrever um agente é fácil. Saber se ele está funcionando bem, não. Este pacote faz as duas coisas: um harness mínimo para rodar o agente, e as ferramentas de medição que quase ninguém tem.

Funciona com **qualquer modelo** — Claude, GPT, Llama local, o que você quiser. O núcleo não tem uma dependência sequer.

---

## Em 30 segundos

```bash
pip install -e ".[harness]"
```

```python
from harness_eng import Harness

h = Harness(model="claude-opus-5")

@h.tool
def clima(cidade: str) -> str:
    """Diz o tempo agora numa cidade."""
    return f"Em {cidade}: 24°C, sol."

run = h.run("como está o tempo em Recife?")

print(run.final_text)   # a resposta
print(run.report())     # o que aconteceu no caminho
```

```
completed — end_turn
  iterações   2
  ferramentas 1 chamadas, 0 erros
  tokens      3.104
  cache       92,8% de acerto
```

É isso. Sem JSON Schema escrito à mão, sem classe base, sem registro global. **Uma ferramenta é uma função** — o `@h.tool` lê a assinatura e o docstring e monta o schema sozinho.

> Quer só experimentar? `harness-eng run "..." --dry-run` roda o loop inteiro com um roteiro fixo: sem chave de API, sem gastar nada.

---

## Espera — o que é um "harness"?

O modelo não faz nada sozinho. Ele recebe texto e devolve texto — inclusive quando esse texto é "quero chamar a ferramenta `X` com estes argumentos". Alguém precisa ler esse pedido, **executar** a ferramenta, devolver o resultado e perguntar de novo. Esse alguém é o harness.

```
   você ──▶ prompt
              │
              ▼
        ┌───────────┐   "quero chamar clima('Recife')"
        │  MODELO   │ ──────────────────────┐
        └───────────┘                       ▼
              ▲                      ┌─────────────┐
              │   "24°C, sol."       │   HARNESS   │  ◀── é isto aqui
              └──────────────────────│  executa a  │
                                     │  ferramenta │
                                     └─────────────┘
                          repete até o modelo dizer "terminei"
```

O harness é o `while` no meio. E é onde mora quase todo problema real de agente: ele parou cedo demais? tarde demais? executou uma chamada que veio pela metade? gastou 40 mil tokens repetindo o mesmo comando?

**Nada disso é culpa do modelo — e nada disso aparece se ninguém medir.** Daí o pacote.

---

## Qualquer IA

`Harness(model=...)` usa a Anthropic. Para o resto, troque o cliente:

```python
from harness_eng.core.clients import OpenAIClient

# OpenAI
h = Harness(client=OpenAIClient(model="gpt-5"))

# Llama rodando local no Ollama — mesmo código, mesmas métricas
h = Harness(client=OpenAIClient(
    model="llama3",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
))
```

O formato *chat completions* é o que Ollama, vLLM, Groq, Together e OpenRouter falam, então esse cliente já cobre a maior parte do mundo.

**E um provedor que ninguém suporta?** A porta pede dois membros. É isto, inteiro:

```python
from harness_eng import ModelResponse
from harness_eng.trace.model import StopReason

class MeuModelo:
    model = "meu-modelo"

    def complete(self, conversa, tools):
        resposta = minha_api(conversa, tools)      # o que você já tem
        return ModelResponse(
            text=resposta.texto,
            stop_reason=StopReason.END_TURN,
        )

h = Harness(client=MeuModelo())
```

Pronto: loop, trace, métricas e estatística funcionam igual. Não há caso especial de provedor em nenhum outro arquivo do pacote — e um teste quebra o build se alguém tentar botar um.

---

## Ferramentas são funções

```python
@h.tool
def buscar(termo: str, limite: int = 10) -> list[str]:
    """Busca no catálogo.

    Args:
        termo: o que procurar
        limite: máximo de resultados
    """
    return catalogo.query(termo)[:limite]
```

O que o pacote faz por você:

| Você escreve | O modelo recebe |
|---|---|
| `termo: str` | `{"type": "string"}`, obrigatório |
| `limite: int = 10` | `{"type": "integer"}`, opcional (tem default) |
| `peso: float \| None = None` | número, opcional |
| `tags: list[str]` | array de strings |
| a linha do docstring | a descrição da ferramenta |
| `termo: o que procurar` | a descrição do parâmetro |

E mais três coisas que evitam dor de cabeça:

- **Esqueceu de anotar um tipo? Falha na hora de decorar**, com mensagem. Schema adivinhado passa no teste, chega ao modelo com o tipo errado e só reaparece em produção como "o modelo mandou string onde eu queria número".
- **Retornou `dict` ou `list`?** Vira JSON, não `repr` do Python. Aspas duplas e `null` o modelo lê melhor.
- **Modelo mandou argumento errado?** Ele recebe *"faltou o argumento 'b'"* e corrige na chamada seguinte — em vez de um traceback, que não ajuda ninguém.

A função continua uma função normal: dá para chamar, testar e importar como sempre.

---

## O `while` é fácil. As bordas é que não são.

Estas quatro estão implementadas e testadas — e as quatro são silenciosas, isto é, dão errado **sem levantar erro nenhum**:

| Borda | O que acontece se errar |
|---|---|
| `pause_turn` não é fim de turno | O loop encerra no meio e devolve trabalho parcial, registrando "terminou normalmente" |
| Bater no teto de iterações **não é sucesso** | Sem teto, um modelo que nunca diz "terminei" roda até a conta acabar |
| Resposta cortada por `max_tokens` não executa a ferramenta | O último bloco pode estar pela metade: é rodar um comando que o modelo não acabou de escrever |
| Resultados paralelos voltam numa mensagem só | Parti-los é aceito pela API e ensina o modelo a parar de pedir chamadas em paralelo |

`run.ok` é `True` só quando o modelo terminou por vontade própria. Na linha de comando, o código de saída segue a mesma regra — num script de CI, "bati no teto" e "terminei" não são a mesma coisa.

---

## Níveis: quanto poder esse agente precisa?

Um harness não é uma coisa só. É um conjunto de **capacidades concedidas** — ler arquivo, sair para a internet, rodar comando — e cada uma tem grau. "Rodei o agente" não descreve um experimento. "Rodei o agente no nível 2" descreve.

```python
h = Harness(model="claude-opus-5", level=2)      # lê arquivo + internet
```

| Nível | Nome | Arquivo | Rede | Comando | Ferramentas concedidas |
|---|---|---|---|---|---|
| **0** | `sealed` | — | — | — | nenhuma: só o modelo |
| **1** | `reader` | lê | — | — | `read_file` `list_dir` `find_files` |
| **2** | `researcher` | lê | allowlist | — | `+ fetch_url` |
| **3** | `builder` | escreve | allowlist | — | `+ write_file` |
| **4** | `operator` | escreve | allowlist | allowlist | `+ run_command` |

O nível 0 não é inútil: é a linha de base. *"Dar acesso a arquivo melhorou?"* só tem resposta se existir a execução sem acesso a arquivo.

### Internet e bloqueio de sites

A allowlist nasce **vazia**. Quem sobe para o nível de rede tem de dizer onde o agente pode ir:

```python
from harness_eng import RESEARCHER

politica = RESEARCHER.allowing("docs.python.org", "pypi.org").blocking("interno.pypi.org")
h = Harness(model="claude-opus-5", policy=politica)
```

```bash
harness-eng run "pesquise X" --level 2 --allow docs.python.org --block interno.exemplo.com
harness-eng run "rode os testes" --level 4 --allow-command pytest --allow-command git
```

Quatro detalhes que decidem se a política é real ou decorativa:

- **Bloqueio vence allowlist.** Você quer "tudo em `exemplo.com`, menos `interno.exemplo.com`"; a ordem inversa faria a exceção não valer nada.
- **Subdomínio casa com ponto.** `pypi.org` libera `api.pypi.org` e **não** libera `naopypi.org`. Esquecer o ponto é o jeito clássico de uma allowlist deixar passar um domínio parecido.
- **Redirecionamento revalida.** Um domínio liberado responde 302 para um bloqueado e o cliente HTTP segue sozinho — a forma clássica de furar allowlist, e invisível para qualquer teste que só busque URLs bem-comportadas.
- **`run_command` roda sem shell.** A allowlist casa o primeiro token, e isso só significa algo se o primeiro token for de fato o executável: com shell, `pytest && curl ...` passaria pela checagem. Perder pipe e redirecionamento é o preço de a checagem não ser teatro.

### A parte que ninguém mede: a parede

Uma política que bloqueia **em silêncio** não ensina nada. Você fica sabendo que o agente falhou — não que ele bateu numa parede. Aqui toda negativa é contada, com motivo e alvo, e vira métrica:

```bash
harness-eng analyze traces/
```

```
POLÍTICA
  o nível apertou: 23 negativas em 6 de 8 sessões, mais em 'stackoverflow.com'.
     14x  tentou stackoverflow.com
      9x  tentou github.com
```

Duas leituras, as duas caras e as duas invisíveis sem isto:

| O relatório diz | Significa |
|---|---|
| **"o nível sobrou"** — `fetch_url` concedida e nunca usada | Risco carregado de graça. Ninguém percebe, porque nada dá errado — é por nada dar errado que o excesso sobrevive por anos. |
| **"o nível apertou"** — 23 negativas, mais em `stackoverflow.com` | O agente contornou. Aparece como execução mais longa e mais cara, e é atribuído ao modelo, ao prompt, à tarefa — a qualquer coisa menos à política, porque a parede não estava no relatório. |

E o veredito nunca afirma além da amostra: abaixo de 5 sessões ele diz *"indício, não conclusão"*. Um relatório que chama n=2 de evidência comete o mesmo erro que este repositório documenta na própria camada estatística.

> **Negativa é contada separado de falha.** Para o modelo as duas são erro — ele precisa ver as duas. Para a medição são opostas: falha significa que algo quebrou, negativa significa que a política funcionou. Somá-las produziria uma taxa de erro que **sobe quando você aperta a segurança**.

O nível inteiro vai para o trace, não só o número — nível é um rótulo que depende de uma tabela que muda entre versões, e o trace precisa dizer o que estava concedido *naquela* execução.

---

## Medir

Toda execução vira um **trace**: um registro do que aconteceu, turno a turno.

```python
run = h.run("...", save_to="traces/hoje.jsonl")
```

```bash
harness-eng analyze traces/            # métricas sobre o que rodou
harness-eng analyze --redact           # troca caminho e comando por hash estável
harness-eng compare antes.json depois.json --metric "custo/sessão"
harness-eng power                      # quantas execuções preciso para concluir algo?
```

O `analyze` também lê os transcripts do **Claude Code** direto de `~/.claude/projects` — dá para medir o agente que você usa todo dia sem instrumentar nada.

### O que ele já encontrou

Rodando sobre **54 transcripts reais** (26.271 turnos, 8.412 chamadas de ferramenta):

| Achado | Número | Por que importa |
|---|---|---|
| **PowerShell erra 5x mais que Bash** | 14,3% vs 3,1% | Mesmo trabalho, ferramentas diferentes. É conserto de harness, não de modelo. |
| **Pico de contexto: 998.281 tokens** | o limite é 1M | A 1,7 mil tokens de estourar a janela. |
| Crescimento de contexto concentrado | 27% vem de 5% dos turnos | Diz *qual* otimização vale: cortar leitura grande, não sumarizar. |
| `ExitPlanMode` lidera o ranking de erro | 17,2% (29 chamadas) | Só aparece sem o filtro de amostra mínima — some de qualquer painel que só olhe volume. |
| Cache: 96,5% de acerto, rewrite 0,036 | — | O *rewrite ratio* é o número que denuncia cache mal usado. |
| 41 padrões de loop, 190 chamadas desperdiçadas | 2,3% do total | Um comando repetido 11x na mesma janela. |
| Custo estimado | US$ 4.496 | Com lacunas explícitas quando um modelo não tem preço. |

Nenhum desses números veio de ferramenta existente.

---

## "Mudei o prompt e melhorou" — será?

Esta é a parte que separa o pacote de um logger bonito. Comparar duas configurações de agente é um problema estatístico, e quase todo mundo erra do mesmo jeito: roda cinco vezes, olha a média, publica.

**Pareie por tarefa.** A variância entre tarefas é enorme — no baseline real, mediana US$ 67 e máximo US$ 548. Comparar 20 execuções de A contra 20 de B mede principalmente *quais tarefas caíram em qual grupo*.

**Quantas execuções você precisa?** Calculado por simulação sobre o baseline real, não por fórmula:

| Detectar melhora de | Tarefas pareadas |
|---|---|
| 5% | 64 |
| 10% | 12 |
| 20% ou mais | 4 |

Se nem 200 bastam, `required_pairs` devolve `None` em vez do teto. "Preciso de mais de 200 para detectar 5%" é uma resposta útil: quase sempre significa que a pergunta deve mudar, não o `n`.

<details>
<summary><b>As decisões estatísticas — e a que eu errei</b> (clique)</summary>

### O veredito usa teste `t`. Esta era a decisão errada na primeira versão.

A justificativa original foi: *"as distribuições deste domínio são assimétricas, logo o teste `t` mente"*. Premissa certa — o baseline real tem assimetria **+1,66**. Conclusão errada.

O teste `t` **pareado** não supõe que os dados sejam normais. Supõe que as *diferenças* sejam aproximadamente simétricas. E o pareamento é exatamente o que produz isso: a assimetria das diferenças do mesmo baseline é **−0,13**.

Calibração medida, 3.000 repetições sob a hipótese nula (nominal 5%, ±0,8%):

| n | teste `t` | bootstrap percentil |
|---|---|---|
| 12 | 2,7% (conservador) | **6,8%** (liberal) |
| 20 | 3,1% | **6,4%** |
| 40 | **5,2%** ✓ | 6,1% |

O bootstrap da mediana é liberal em toda a faixa testada — e **BCa não corrige** (7,5% em n=12, medido). A mediana é uma estatística não-suave e a teoria de bootstrap para ela é fraca em amostra pequena.

Consequência desconfortável: parte do "poder maior" que o bootstrap exibia era só ele rejeitar mais vezes, inclusive quando não devia.

O bootstrap continua no pacote, para o que ele faz bem: intervalo em torno da **mediana** — que responde outra pergunta ("a tarefa típica melhorou?") e não é dominado por uma sessão cara.

### O portão que escolhe o método também precisou de conserto

A verificação de simetria usava o momento de terceira ordem. Num dado de diferenças **simétrico por construção** sobre cauda pesada, ele devolveu **−10,76**: um único outlier elevado ao cubo domina o estimador. Trocado por assimetria robusta de quartis (Bowley), que dá −0,12 no mesmo dado. Quartis colapsados devolvem `None` — *não consegui verificar* — e isso encaminha ao método mais robusto, nunca ao mais frágil.

### Duas medidas de efeito, porque medem coisas diferentes

Delta de Cliff descreve sobreposição das distribuições; **dominância pareada** descreve consistência. Este módulo errou isso também:

> Num teste em que **toda** tarefa ficou 25-45% mais barata, o delta de Cliff saiu +0,32 — "pequeno". Não por bug: o delta compara todas as observações contra todas e ignora o pareamento, então a tarefa cara melhorada ainda custa mais que a tarefa barata original. A dominância pareada, no mesmo teste: **1,00**.

Tudo isso sem `numpy`, `scipy` ou `pandas`. Bootstrap, delta de Cliff e a distribuição `t` (via beta incompleta) são `statistics`, `math` e `random` da biblioteca padrão — e os valores críticos da `t` são conferidos contra tabela publicada no teste. Num repositório cujo argumento é rigor de medição, cálculo auditável linha a linha vale mais que a conveniência de importar.

</details>

---

## Arquitetura

```
harness_eng/
├── harness.py       a porta da frente: Harness, Run, quick
├── trace/           formato canônico — o vocabulário comum entre harnesses
│   ├── model.py     Session, Turn, ToolCall, ToolResult, Usage
│   └── sources/     claude_code · native · (o seu, se quiser)
├── metrics/         puras: recebem dado, devolvem número
│   ├── tools.py     erro, retry, falha silenciosa, chamada sem resposta
│   ├── loops.py     repetição, retry cego, oscilação
│   ├── context.py   crescimento, concentração, cache
│   ├── cost.py      custo por modelo, por sessão, por chamada
│   └── policy.py    o nível serviu? concedeu demais? concedeu de menos?
├── stats/           pareado, bootstrap, tamanho de efeito, poder
├── core/            o harness: loop, ferramentas, clientes
│   ├── policy.py    níveis: os eixos, os graus e a contagem de negativas
│   └── toolkit.py   as ferramentas de cada nível e onde a política é aplicada
└── cli.py
```

A regra: `trace/model.py`, `metrics/` e `stats/` são **puros**. Não conhecem provedor, não tocam disco, não abrem rede. `core/` importa `trace/`; nunca o contrário.

Isso não é purismo — é o que faz **170 testes rodarem em 5 segundos** sem transcript, sem chave de API e sem instalar nada além do `pytest`. `tests/test_layering.py` quebra o build quando a seta inverte, e as regras foram verificadas contra violação plantada: regra de arquitetura que nunca falhou não protege nada.

---

## Honestidade sobre a medição

O repositório aplica em si o que cobra dos outros.

**Um bug da ferramenta pego pelos dados reais.** A primeira versão reportou "58 de 60 chamadas de `ToolSearch` voltaram vazias" — um achado alarmante sobre o harness. Era defeito meu: `ToolSearch` devolve blocos `tool_reference` e `Read` de imagem devolve blocos `image`, e meu extrator só lia `type == "text"`. `ToolResult.content_kinds` agora separa *"não devolveu nada"* de *"não sei ler o que devolveu"* — conclusões opostas.

**O detector de loop contradiz o baseline manual, e está certo.** Um script ingênuo contava 124 padrões repetidos; o detector conta 41. A diferença é a janela: repetição espalhada por uma sessão de 800 chamadas é trabalho legítimo, não loop. O baseline inflava 3x.

**Modelo sem preço vira lacuna, não estimativa.** `CostReport.is_complete` diz quando o total é piso. Inventar um preço "próximo" produz um número que parece exato e está errado — exatamente o padrão que este repositório existe para detectar.

**O adapter conta o que descarta.** `ClaudeCodeSource.skipped` registra cada registro ignorado e por quê. Um adapter que joga fora 30% das linhas está errado, e sem esse contador ninguém descobre.

**O formato canônico tinha um buraco, e escrever o harness foi o que revelou.** `pause_turn` é um motivo de parada real e não existia no `StopReason`; virava `UNKNOWN`. Não apareceu em revisão de código nem nos 54 transcripts — apareceu quando um consumidor novo precisou do caso. É o argumento a favor do segundo adapter: um formato só prova ser comum quando alguém que não o escreveu tenta usá-lo.

**O segundo formato cobrou três diferenças, e as três viraram teste.** Escrever o cliente do formato *chat completions* mostrou que ali um resultado de ferramenta é uma mensagem por resultado, os argumentos chegam como string JSON, e `prompt_tokens` **já inclui** o cache lido — esta última erra em silêncio: repassar o número cru contaria o cache duas vezes e inflaria o tamanho de contexto medido, num relatório cujo assunto é crescimento de contexto.

**A regra de camadas nova nasceu errada, e reprovou o próprio docstring.** A primeira versão grepava o texto do arquivo; o primeiro módulo a citar `harness_eng.core` num docstring — justamente para dizer que **não** o importa — foi reprovado. Trocada por leitura de AST. A lição já estava escrita no mesmo arquivo, para outra regra, e eu a repeti mesmo assim: *um teste de arquitetura que dá falso positivo é desligado numa semana, e aí não protege nada.*

**`analyze` quebrava num trace curto, e o defeito era da apresentação.** As métricas devolvem `None` quando não há o que medir — "ausência é ausência" atravessa o pacote inteiro. O formatador do relatório não sabia disso e estourava `TypeError` no meio da saída. A métrica seguia a regra da casa; a apresentação, não.

**Três erros do próprio pacote de estatística, todos achados medindo.** Estão na seção recolhida acima e no histórico de commits, porque um repositório que prega medição e esconde os próprios erros de medição não vale nada.

---

## Privacidade

Os transcripts **nunca** entram no repositório — `.gitignore` cobre `*.jsonl`, `.env` e saídas de relatório. As ferramentas os leem localmente.

`--redact` substitui caminho e conteúdo de comando por hash SHA-256 estável (estável para o mesmo comando continuar comparável entre execuções; redação que randomiza destrói a análise que justifica o relatório).

---

## Estado

| Fase | Estado |
|---|---|
| Formato canônico + adapter Claude Code | pronto, validado em 54 transcripts |
| Métricas (ferramenta, loop, contexto, custo) | pronto |
| Estatística (pareado, bootstrap, poder) | pronto, validado contra implementação de referência |
| Harness mínimo + trace nativo | pronto, com round-trip verificado |
| Clientes Anthropic e formato OpenAI | pronto |
| API amigável (`Harness`, `@tool`) | pronto — 30 linhas viraram 8 |
| Níveis de harness (arquivo, rede, comando, orçamento) | pronto, com negativas contadas |
| CLI (`analyze`, `compare`, `power`, `run`) | pronto |
| Relatório em HTML (`report/`) | próximo |

MIT.
