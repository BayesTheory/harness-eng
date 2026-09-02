# Começando

## Requisitos

**Python 3.10 ou mais novo.** É só isso.

O núcleo do pacote tem `dependencies = []` — nem `numpy`, nem `scipy`, nem `pandas`. Bootstrap, delta de Cliff e a distribuição `t` são `statistics`, `math` e `random` da biblioteca padrão.

## Instalação

```bash
git clone https://github.com/BayesTheory/harness-eng
cd harness-eng
pip install -e ".[dev]"
```

O extra `[dev]` traz `pytest`, `pytest-cov`, `ruff` e `mypy`. **É tudo que você precisa para contribuir com qualquer parte do projeto.**

## Rodando

```bash
pytest -q          # 170 testes, ~5 segundos
ruff check .       # precisa sair limpo — é o que o CI roda
```

Nenhum dos dois precisa de chave de API, de rede ou de transcript. Se `pytest -q` passou, seu ambiente está pronto.

## O extra `[harness]` — e quando você **não** precisa dele

```bash
pip install -e ".[harness]"     # anthropic, openai, python-dotenv
```

Isto só é necessário para **chamar um modelo de verdade**. Você não precisa dele para:

- rodar a suíte de testes (nenhum dos 170 precisa)
- trabalhar no loop, nas ferramentas, na política, no formato de trace, nas métricas ou na estatística
- analisar transcripts que você já tem (`harness-eng analyze`)
- rodar o harness de ponta a ponta com `--dry-run`

O import dos SDKs é **preguiçoso**, dentro do construtor dos clientes. Importar `harness_eng` sem o extra funciona; a mensagem de erro só aparece para quem de fato tentou falar com um provedor.

> O CI tem um passo que **falha** se `anthropic` ou `openai` aparecerem instalados no ambiente de teste. Não é paranoia: sem essa trava, uma dependência de provedor entraria pela porta dos fundos — um `import` no topo de um módulo que algum teste importa — e a promessa "roda sem chave" morreria em silêncio.

## Experimentando sem gastar nada

`--dry-run` troca o modelo por um roteiro fixo e exercita **o pipeline inteiro**: loop, executor de ferramenta, política, trace nativo e métricas.

```bash
harness-eng run "o que tem neste projeto?" --workspace harness_eng --level 1 --dry-run
```

```
COMPLETED — end_turn
  política        nível 1 (reader): arquivo=read · rede=none · shell=none
  iterações       2/30
  turnos          4
  ferramentas     1 chamadas
  erros           0 (0%)
  tokens          0 (cache lido 0, escrito 0)
  trace           reports/native/native-20260902T193901.jsonl

  medir: harness-eng analyze reports/native
```

E aí o círculo fecha — o trace que o harness acabou de escrever passa pelas mesmas métricas que medem harness de terceiro:

```bash
harness-eng analyze reports/native
```

## Analisando o que você já tem

Se você usa Claude Code, há transcripts em `~/.claude/projects` esperando para serem medidos, sem instrumentar nada:

```bash
harness-eng analyze            # ~/.claude/projects por padrão
harness-eng analyze --redact   # troca caminho e comando por hash estável
```

> Os transcripts **nunca** entram no repositório: o `.gitignore` cobre `*.jsonl`, `.env`, `reports/` e `out/`. As ferramentas os leem localmente.

## Onde as coisas ficam

```
harness_eng/
├── harness.py       a porta da frente: Harness, Run, quick
├── cli.py           os 5 subcomandos
├── trace/           o formato canônico + os adapters
├── metrics/         puras: recebem dado, devolvem número
├── stats/           comparação pareada, bootstrap, poder
└── core/            o harness: loop, ferramentas, política, clientes
```

Antes de mexer em qualquer um deles, vale ler [arquitetura.md](arquitetura.md) — há sete regras de dependência verificadas por teste, e elas quebram o build quando a seta inverte.
