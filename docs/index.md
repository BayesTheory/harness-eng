# Documentação do harness-eng

O [README](../README.md) explica **o que** o projeto é e **por que** existe — é o material de quem está avaliando. Estas páginas são de quem vai mexer no código.

## Comece por aqui

| Página | Responde |
|---|---|
| [Começando](comecando.md) | Como montar o ambiente e rodar tudo — **sem chave de API, sem custo** |
| [CLI ou API?](cli-vs-api.md) | As duas superfícies, o que cada uma faz, e onde elas **não** se equivalem |
| [Chamando o modelo](chamando-o-modelo.md) | Anthropic, OpenAI, Ollama, o seu provedor — e a questão das credenciais |

## Estendendo

| Página | Responde |
|---|---|
| [Adicionar uma ferramenta](adicionar-ferramenta.md) | `@tool`, o schema derivado da assinatura, e o contrato de erro |
| [Adicionar um adapter](adicionar-adapter.md) | Como fazer o pacote medir um harness que ele ainda não conhece |

## Entendendo

| Página | Responde |
|---|---|
| [Arquitetura](arquitetura.md) | As camadas, as sete regras que as protegem, e por que a seta aponta assim |
| [Testes](testes.md) | Por que 170 testes nunca chamam um modelo — e o que **não** está testado |

---

## O caminho mais curto

```bash
git clone https://github.com/BayesTheory/harness-eng
cd harness-eng
pip install -e ".[dev]"
pytest -q                 # 170 passam, ~5s, sem chave nenhuma
```

Se isso funcionou, você já consegue contribuir com qualquer parte do projeto. Ver [CONTRIBUTING.md](../CONTRIBUTING.md) para o fluxo de PR.

---

## Uma convenção que vale saber antes de ler o código

Este repositório escreve o **porquê** nos docstrings, não o **o quê**. Um comentário que diz "incrementa o contador" acima de `contador += 1` não existe aqui; um que diz *por que* o contador precisa existir, sim.

A consequência prática: os docstrings de módulo são longos (11 a 41 linhas) e valem a leitura antes de mexer no arquivo. Eles carregam as decisões e, com frequência, o erro que a decisão consertou — vários módulos documentam bugs reais que só apareceram quando a medição contradisse o que o código afirmava.
