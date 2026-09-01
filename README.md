# harness-eng

Ferramentas para **medir** harnesses de agente.

O campo de harness engineering hoje é anedótico: "esse prompt parece melhor", "esse loop parece mais estável". Quase ninguém publica intervalo de confiança, quase ninguém pareia, e praticamente ninguém sabe quantas execuções precisaria rodar para a comparação significar alguma coisa.

Este repositório é a parte que falta: um formato de trace agnóstico, métricas que rodam sobre ele, e uma camada estatística feita para as distribuições que este domínio realmente tem — assimétricas, com cauda pesada, e onde teste `t` dá resposta errada com aparência de precisão.

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
```

O núcleo **não tem dependência nenhuma** — nem `numpy`, nem `scipy`, nem `pandas`. Bootstrap e delta de Cliff são ~80 linhas de `statistics` e `random` da biblioteca padrão. Num repositório cujo argumento é rigor de medição, cálculo auditável linha a linha vale mais que a conveniência de importar.

---

## Arquitetura

```
src/harness_eng/
├── trace/           formato canônico — o vocabulário comum entre harnesses
│   ├── model.py     Session, Turn, ToolCall, ToolResult, Usage, TraceSet
│   ├── ports.py     TraceSource, TraceSink
│   └── sources/     claude_code (pronto) · openai · native
├── metrics/         puras sobre o formato canônico
│   ├── tools.py     erro, retry, falha silenciosa, chamada sem resposta
│   ├── loops.py     repetição, retry cego, oscilação
│   ├── context.py   crescimento, concentração, cache
│   └── cost.py      custo por modelo, por sessão, por chamada
├── stats/           o diferencial
│   ├── compare.py   pareado, bootstrap, Cliff's delta, dominância
│   └── design.py    poder por simulação, tamanho de amostra
└── cli.py
```

`trace/model.py`, `metrics/` e `stats/` são **puros**: recebem dado, devolvem número. Não conhecem provedor, não tocam disco, não abrem rede. `tests/test_layering.py` verifica isso e quebra o build quando a seta inverte.

Consequência prática: **76 testes rodam em 6 segundos** sem transcript, sem chave de API e sem instalar nada além do `pytest`.

---

## As três decisões estatísticas

Cada uma tem uma razão que o dado exige, não uma preferência.

**1. Pareamento por tarefa.** A variância entre tarefas é enorme — no baseline real, mediana US$ 67 e máximo US$ 548. Comparar a média de 20 execuções de A contra 20 de B mede principalmente quais tarefas caíram em qual grupo.

**2. Bootstrap, não teste `t`.** O baseline real tem **média/mediana = 1,63**. Acima de ~1,2 a distribuição é assimétrica o bastante para o `t` produzir intervalo confiante e errado. O bootstrap não supõe forma nenhuma.

**3. Duas medidas de efeito, não uma.** Delta de Cliff descreve sobreposição das distribuições. **Dominância pareada** descreve consistência. Os dois saem juntos porque medem coisas diferentes — e porque este módulo errou isso antes de ser medido:

> Num teste em que **toda** tarefa ficou 25-45% mais barata, o delta de Cliff saiu +0,32 — "pequeno". Não por bug: o delta compara todas as observações contra todas e ignora o pareamento, então com variância grande entre tarefas a tarefa cara melhorada ainda custa mais que a tarefa barata original. A dominância pareada, no mesmo teste: **1,00**.

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
