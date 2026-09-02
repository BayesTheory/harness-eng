# Chamando o modelo

## A porta tem dois membros

Todo o resto do pacote fala com esta interface, nunca com um SDK:

```python
class ModelClient(Protocol):
    @property
    def model(self) -> str: ...

    def complete(self, conversation: Sequence[Turn], tools: Sequence[ToolSpec]) -> ModelResponse: ...
```

É isso. Um provedor novo é uma classe com um atributo e um método — ver [Um provedor que ninguém suporta](#um-provedor-que-ninguém-suporta) abaixo.

## Os três clientes que já existem

| Cliente | Fala com | Precisa de |
|---|---|---|
| `AnthropicClient` | API da Anthropic | chave de API da Anthropic |
| `OpenAIClient` | formato *chat completions* | chave do provedor, ou nada (local) |
| `ScriptedClient` | um roteiro fixo na memória | **nada** |

```python
from harness_eng import Harness
from harness_eng.core.clients import OpenAIClient

Harness(model="claude-opus-5")                  # Anthropic

Harness(client=OpenAIClient(model="gpt-5"))     # OpenAI

Harness(client=OpenAIClient(                    # Llama local no Ollama
    model="llama3",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
))
```

`OpenAIClient` cobre bem mais que a OpenAI: o formato *chat completions* é o que **Ollama, vLLM, Groq, Together, OpenRouter e LM Studio** falam. `base_url` aponta para qualquer um deles.

## Os caminhos não se cruzam

Uma dúvida que já apareceu: *"a Anthropic permite usar outro provedor?"*

A pergunta não se aplica, porque **a Anthropic não participa**. Os imports são preguiçosos e por cliente:

```python
class AnthropicClient:
    def __init__(...):
        import anthropic          # só aqui

class OpenAIClient:
    def __init__(...):
        import openai             # só aqui
```

Usar `OpenAIClient` nem carrega o módulo `anthropic` — é HTTP para `api.openai.com` (ou para o seu `localhost`), com a sua chave, na sua conta. Não é integração entre provedores; são dois adapters que implementam a mesma porta e nunca se encontram.

Uma regra de camada em `tests/test_layering.py` garante que continue assim: **só `core/clients.py` pode importar SDK de provedor.** Se o loop importar `anthropic` direto, o build quebra.

## Credenciais

### Anthropic

O SDK resolve nesta ordem, primeira que existir vence:

1. `ANTHROPIC_API_KEY`
2. `ANTHROPIC_AUTH_TOKEN`
3. perfil OAuth salvo por `ant auth login`
4. federação de identidade (variáveis de WIF)
5. perfil padrão em disco

> **Assinatura do Claude.ai (Pro/Max) não dá acesso à API.** São contas separadas: o endpoint `/v1/messages` exige credencial de API. Se você tem só a assinatura, o caminho Anthropic não vai funcionar — use `--dry-run`, `ScriptedClient`, ou um modelo local via `OpenAIClient`.

### Outros provedores

`OpenAIClient(api_key=...)`, ou a variável que o SDK da OpenAI lê. Para servidor local, qualquer string serve (`api_key="ollama"`), porque não há autenticação de verdade do outro lado.

### `.env`

A CLI carrega `.env` automaticamente **se** `python-dotenv` estiver instalado (vem no extra `[harness]`). Sem ele, só variáveis de ambiente.

> **Uma promessa quebrada, registrada aqui para não virar mistério:** o `.env.example` pede `OPENAI_API_KEY` desde o primeiro commit e **nada no código lê essa variável**. A CLI só instancia `AnthropicClient`, e o `OpenAIClient` recebe a chave por argumento. Consertar isso faz parte do mesmo trabalho de dar `--provider` à CLI.

## Um provedor que ninguém suporta

Este é o cliente inteiro. Não há classe base, registro global nem decorador:

```python
from harness_eng import Harness, ModelResponse
from harness_eng.trace.model import StopReason, ToolCall

class MeuModelo:
    model = "meu-modelo"

    def complete(self, conversa, tools):
        resposta = minha_api(conversa, tools)     # o que você já tem

        return ModelResponse(
            text=resposta.texto,
            tool_calls=tuple(
                ToolCall(id=c.id, name=c.nome, arguments=c.args)
                for c in resposta.chamadas
            ),
            stop_reason=StopReason.TOOL_USE if resposta.chamadas else StopReason.END_TURN,
        )

h = Harness(client=MeuModelo())
```

Pronto: loop, níveis, trace, métricas e estatística funcionam igual. Não existe caso especial de provedor em nenhum outro arquivo do pacote.

### Quatro detalhes que os dois adapters existentes tiveram de resolver

Se o seu provedor for parecido com algum deles, você vai encontrar os mesmos:

- **`stop_reason` importa mais do que parece.** `pause_turn` significa "pausei, me retome" — traduzi-lo como fim de turno faz o loop devolver trabalho pela metade sem erro nenhum. Foi um buraco real no `StopReason` deste pacote, achado justamente ao escrever o segundo consumidor.
- **Argumentos de ferramenta podem chegar como string JSON.** A Anthropic manda dicionário; o formato *chat completions* manda texto. Repassar a string faria `ToolCall.arguments` virar `str` onde o resto do pacote espera mapa — e o detector de loop, que assina a chamada pelos argumentos ordenados, pararia de casar repetições.
- **Contagem de token pode contar duas vezes.** No formato *chat completions*, `prompt_tokens` **já inclui** os tokens lidos de cache. Como `Usage.context_size` soma input + cache lido + cache escrito, repassar o número cru infla o tamanho de contexto medido — num relatório cujo assunto é crescimento de contexto. Ver `_openai_usage()`.
- **Ausência é ausência.** Se a resposta não trouxe uso de token, devolva `usage=None`, não `Usage(0, 0, 0, 0)`. Zero é uma medição; ausente não é, e confundir os dois envenena toda média que vier depois.

### Como testar o seu cliente sem gastar

Separe **tradução** de **transporte**. Os dois adapters existentes fazem isso: `from_message()` e `from_completion()` são funções puras que recebem o objeto de resposta e devolvem `ModelResponse`. Dá para testá-las com um `SimpleNamespace`:

```python
from types import SimpleNamespace

def test_traducao():
    falso = SimpleNamespace(content=[...], stop_reason="tool_use", usage=...)
    resposta = from_message(falso)
    assert resposta.tool_calls[0].name == "read_file"
```

E o construtor de ambos aceita `client=` para injetar um cliente falso sem tocar a rede.

> **Onde a cobertura acaba, e vale saber:** `.complete()` dos clientes reais **nunca é executado** em teste nenhum. As funções de tradução são cobertas com objetos falsos; o envio de verdade, não. Ver [testes.md](testes.md#a-lacuna).
