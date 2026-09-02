# harness-eng

Ferramentas para **medir** harnesses de agente.

O campo de harness engineering hoje é anedótico: "esse prompt parece melhor", "esse loop parece mais estável". Quase ninguém publica intervalo de confiança, quase ninguém pareia, e praticamente ninguém sabe quantas execuções precisaria rodar para a comparação significar alguma coisa.

Este repositório é a parte que falta: um formato de trace agnóstico, métricas que rodam sobre ele, e uma camada estatística cujas escolhas foram **medidas em vez de supostas** — inclusive quando a medição contradisse o que eu tinha escrito.

---

## O que ele já encontrou

Rodando sobre **54 transcripts reais** (26.271 turnos, 8.412 chamadas de ferramenta):

| Achado | Número | Por que importa |
|---|---|---|
| **PowerShell erra 5x mais que Bash** | 14,3% vs 3,1% | Mesmo trabalho, ferramentas diferentes. É conserto de harness, não de modelo. |
| `ExitPlanMode` lidera o ranking | 17,2% (29 chamadas) | Só aparece sem o filtro de amostra mínima — e some de qualquer painel que só olhe volume. |
| **Pico de contexto: 998.281 tokens** | limite é 1M | A 1,7 mil tokens de estourar a janela. |
| Crescimento de contexto concentrado | 27% vem de 5% dos turnos | Diz *qual* otimização vale: cortar leitura grande, não sumarizar. |
| Cache: 96,5% de acerto, rewrite 0,036 | — | Prefixo bem estável. O rewrite ratio é o número que denuncia cache mal usado. |
| 41 padrões de loop, 190 chamadas desperdiçadas | 2,3% do total | Um comando repetido 11x na mesma janela. |
| Custo estimado | US$ 4.496 | Com lacunas explícitas quando um modelo não tem preço. |

Nenhum desses números veio de ferramenta existente.

---

## Uso

```bash
pip install -e ".[dev]"

harness-eng analyze                    # ~/.claude/projects por padrão
harness-eng analyze ./traces --json
harness-eng analyze --redact           # troca caminho e comando por hash estável
harness-eng power                      # quantas tarefas para detectar uma melhora
harness-eng compare atual.json novo.json --metric "custo/sessão"

harness-eng run "resuma este pacote" --workspace harness_eng     # o harness mínimo
harness-eng run "..." --dry-run        # loop inteiro, sem chave e sem custo
```

O núcleo **não tem dependência nenhuma** — nem `numpy`, nem `scipy`, nem `pandas`. Bootstrap, delta de Cliff e a distribuição `t` (via beta incompleta) são `statistics`, `math` e `random` da biblioteca padrão. Num repositório cujo argumento é rigor de medição, cálculo auditável linha a linha vale mais que a conveniência de importar — e os valores críticos da `t` são conferidos contra tabela publicada no teste.

---

## Arquitetura

```
harness_eng/
├── trace/           formato canônico — o vocabulário comum entre harnesses
│   ├── model.py     Session, Turn, ToolCall, ToolResult, Usage, TraceSet
│   ├── ports.py     TraceSource, TraceSink
│   └── sources/     claude_code (pronto) · native (pronto) · openai
├── metrics/         puras sobre o formato canônico
│   ├── tools.py     erro, retry, falha silenciosa, chamada sem resposta
│   ├── loops.py     repetição, retry cego, oscilação
│   ├── context.py   crescimento, concentração, cache
│   └── cost.py      custo por modelo, por sessão, por chamada
├── stats/           o diferencial
│   ├── compare.py    pareado, escolha de método, efeito, dominância
│   ├── parametric.py teste t + distribuição t sem scipy, assimetria robusta
│   └── design.py     poder por simulação, tamanho de amostra
├── core/            o harness mínimo — o que fecha o círculo
│   ├── ports.py     ModelClient, ModelResponse, ToolSpec
│   ├── loop.py      o loop e os cinco desfechos possíveis
│   ├── tools.py     registro, execução e as ferramentas de leitura
│   └── clients.py   Anthropic (import preguiçoso) · Scripted (sem rede)
└── cli.py
```

`trace/model.py`, `metrics/` e `stats/` são **puros**: recebem dado, devolvem número. Não conhecem provedor, não tocam disco, não abrem rede. `tests/test_layering.py` verifica isso e quebra o build quando a seta inverte.

A seta aponta num sentido só: `core/` importa `trace/`, e `trace/`, `metrics/` e `stats/` não sabem que ele existe. Inverter seria tentador — o harness tem o `ToolSpec`, e uma métrica sobre descrição de ferramenta ficaria "natural" importando dele. No dia em que isso acontecesse, medir um harness de terceiro passaria a arrastar o loop, o cliente de modelo e o SDK junto. Também é teste, e os dois foram verificados contra violação plantada — regra de arquitetura que nunca falhou não protege nada.

Consequência prática: **114 testes rodam em 6 segundos** sem transcript, sem chave de API e sem instalar nada além do `pytest`.

---

## As decisões estatísticas, e a que eu errei

**1. Pareamento por tarefa.** A variância entre tarefas é enorme — no baseline real, mediana US$ 67 e máximo US$ 548. Comparar a média de 20 execuções de A contra 20 de B mede principalmente quais tarefas caíram em qual grupo.

**2. O veredito usa teste `t`. Esta era a decisão errada na primeira versão.**

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

O bootstrap continua no pacote, para o que ele faz bem: intervalo em torno da **mediana** — que responde outra pergunta ("a tarefa típica melhorou?") e não é dominado por uma sessão cara — e estatísticas sem teoria paramétrica pronta.

**3. O portão que escolhe o método também precisou de conserto.** A verificação de simetria usava o momento de terceira ordem. Num dado de diferenças **simétrico por construção** sobre cauda pesada, ele devolveu **−10,76**: um único outlier elevado ao cubo domina o estimador. Trocado por assimetria robusta de quartis (Bowley), que dá −0,12 no mesmo dado. Quartis colapsados devolvem `None` — *não consegui verificar* — e isso encaminha ao método mais robusto, nunca ao mais frágil.

**4. Duas medidas de efeito.** Delta de Cliff descreve sobreposição das distribuições; **dominância pareada** descreve consistência. Os dois saem juntos porque medem coisas diferentes — e porque este módulo errou isso também:

> Num teste em que **toda** tarefa ficou 25-45% mais barata, o delta de Cliff saiu +0,32 — "pequeno". Não por bug: o delta compara todas as observações contra todas e ignora o pareamento, então a tarefa cara melhorada ainda custa mais que a tarefa barata original. A dominância pareada, no mesmo teste: **1,00**.

### Quantas execuções você precisa

Calculado por simulação sobre o baseline real, não por fórmula:

| Detectar melhora de | Tarefas pareadas |
|---|---|
| 5% | 64 |
| 10% | 12 |
| 20% ou mais | 4 |

`required_pairs` devolve `None` — não o teto — quando nem 200 tarefas bastam. "Preciso de mais de 200 para detectar 5%" é uma resposta útil: quase sempre significa que a pergunta deve mudar, não o `n`.

---

## Honestidade sobre a medição

O repositório aplica em si o que cobra dos outros.

**Um bug da ferramenta pego pelos dados reais.** A primeira versão reportou "58 de 60 chamadas de `ToolSearch` voltaram vazias" — um achado alarmante sobre o harness. Era defeito meu: `ToolSearch` devolve blocos `tool_reference` e `Read` de imagem devolve blocos `image`, e meu extrator só lia `type == "text"`. `ToolResult.content_kinds` agora separa *"não devolveu nada"* de *"não sei ler o que devolveu"* — conclusões opostas.

**O detector de loop contradiz o baseline manual, e está certo.** Um script ingênuo contava 124 padrões repetidos; o detector conta 41. A diferença é a janela: repetição espalhada por uma sessão de 800 chamadas é trabalho legítimo, não loop. O baseline inflava 3x.

**Modelo sem preço vira lacuna, não estimativa.** `CostReport.is_complete` diz quando o total é piso. Inventar um preço "próximo" produz um número que parece exato e está errado — exatamente o padrão que este repositório existe para detectar.

**O adapter conta o que descarta.** `ClaudeCodeSource.skipped` registra cada registro ignorado e por quê. Um adapter que joga fora 30% das linhas está errado, e sem esse contador ninguém descobre.

**Três erros do próprio pacote de estatística, todos achados medindo.** A escolha do bootstrap sobre o teste `t` estava mal justificada; o delta de Cliff estava errado como critério para desenho pareado; e o diagnóstico de simetria explodia com um outlier. Nenhum apareceu por revisão de código — os três apareceram quando a simulação contradisse o que o README afirmava. Está tudo na seção acima e no histórico de commits, porque um repositório que prega medição e esconde os próprios erros de medição não vale nada.

---

## Privacidade

Os transcripts **nunca** entram no repositório — `.gitignore` cobre `*.jsonl`, `.env` e saídas de relatório. As ferramentas os leem localmente.

`--redact` substitui caminho e conteúdo de comando por hash SHA-256 estável (estável para o mesmo comando continuar comparável entre execuções; redação que randomiza destrói a análise que justifica o relatório). Os exemplos de comando neste README estão redigidos — os agregados não precisam de redação, e são eles que sustentam os achados.

---

## Estado

| Fase | Estado |
|---|---|
| Formato canônico + adapter Claude Code | pronto, validado em 54 transcripts |
| Métricas (ferramenta, loop, contexto, custo) | pronto |
| Estatística (pareado, bootstrap, poder) | pronto, validado contra implementação de referência |
| CLI (`analyze`, `compare`, `power`) | pronto |
| Harness mínimo (`core/`) | próximo |
| Adapter OpenAI | depois do harness — a estrutura agnóstica precisa provar que aguenta o segundo formato |

MIT.
