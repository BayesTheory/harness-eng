"""
Clientes de modelo: as implementações concretas da porta :class:`ModelClient`.

Este é o único módulo do pacote que importa um SDK de provedor, e o import é preguiçoso —
dentro do construtor, não no topo. Consequência prática: ``pip install harness-eng`` sem
o extra ``[harness]`` continua servindo para analisar transcript, que é o uso principal, e
a mensagem de erro só aparece para quem de fato tentou rodar o loop.

:class:`ScriptedClient` é o outro lado disso: um cliente que responde a partir de um
roteiro fixo. Não é acessório de teste — é o que permite exercitar o loop inteiro, com
seus modos de falha, sem chave, sem rede e sem custo. Um harness cujo comportamento de
borda só dá para observar gastando dinheiro é um harness que ninguém verifica.
"""
from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable, Sequence
from typing import Any

from ..trace.model import Role, StopReason, ToolCall, Turn, Usage
from .ports import ModelError, ModelResponse, ToolSpec

#: Padrão do harness. Acima disto o SDK exige streaming para a requisição não morrer no
#: timeout de HTTP antes de o modelo terminar de escrever.
DEFAULT_MAX_TOKENS = 16_000

#: Acima deste teto a requisição vai por streaming. É o limiar do SDK, não uma preferência.
STREAMING_THRESHOLD = 16_000

DEFAULT_MODEL = "claude-opus-5"


class ScriptedClient:
    """
    Responde de um roteiro. Implementa :class:`~harness_eng.core.ports.ModelClient`.

    Determinístico de propósito: o mesmo roteiro produz o mesmo trace, então um teste
    sobre o trace testa o loop e não o humor do modelo.
    """

    def __init__(self, responses: Iterable[ModelResponse], model: str = "scripted") -> None:
        self._responses = deque(responses)
        self._model = model
        #: As conversas recebidas, em ordem. É o que deixa um teste verificar o que o loop
        #: **enviou** — inclusive que os resultados paralelos foram num turno só.
        self.seen: list[tuple[Turn, ...]] = []

    @property
    def model(self) -> str:
        return self._model

    def complete(self, conversation: Sequence[Turn], tools: Sequence[ToolSpec]) -> ModelResponse:
        self.seen.append(tuple(conversation))
        if not self._responses:
            # Roteiro curto demais é erro de teste, não fim de conversa. Devolver
            # ``end_turn`` aqui faria um loop que devia estourar o teto terminar limpo, e
            # o teste passaria medindo a coisa errada.
            raise ModelError("roteiro esgotado: o loop pediu mais turnos do que o script tem")
        return self._responses.popleft()


class AnthropicClient:
    """
    Fala com a API da Anthropic. Implementa :class:`~harness_eng.core.ports.ModelClient`.

    Duas escolhas que valem explicação, porque as duas aparecem nas métricas depois:

    **Cache no prefixo.** ``cache_control`` no topo da requisição faz a API cachear o
    último bloco cacheável, e num loop de agente o prefixo cresce a cada turno — é
    exatamente a forma que o cache foi feito para atender. O número que prova que
    funcionou é ``cache_read_tokens``, que este cliente grava em cada turno e o
    ``metrics/context.py`` agrega. Se ele vier zero em execuções seguidas, alguma coisa no
    prefixo está variando, e o relatório mostra isso sem ninguém precisar suspeitar antes.

    **Pensamento adaptativo.** ``budget_tokens`` foi removido nos modelos atuais e é
    rejeitado com 400; ``thinking={"type": "adaptive"}`` com ``effort`` é o controle. O
    ``.env.example`` já falava em esforço — o campo existe na API, e é este.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        system: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str = "high",
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        self._model = model
        self._system = system
        self._max_tokens = max_tokens
        self._effort = effort

        if client is not None:
            # Injeção para teste: exercita a tradução de resposta sem tocar a rede.
            self._client = client
            return
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depende do ambiente
            raise ModelError(
                "o SDK da Anthropic não está instalado. "
                'Instale o extra: pip install -e ".[harness]"'
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    @property
    def model(self) -> str:
        return self._model

    def complete(self, conversation: Sequence[Turn], tools: Sequence[ToolSpec]) -> ModelResponse:
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": to_messages(conversation),
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self._effort},
            "cache_control": {"type": "ephemeral"},
        }
        if self._system:
            request["system"] = self._system
        if tools:
            request["tools"] = [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": dict(spec.input_schema),
                }
                for spec in tools
            ]

        try:
            if self._max_tokens > STREAMING_THRESHOLD:
                with self._client.messages.stream(**request) as stream:
                    message = stream.get_final_message()
            else:
                message = self._client.messages.create(**request)
        except Exception as exc:  # noqa: BLE001 — traduzido logo abaixo
            raise _translate(exc) from exc

        return from_message(message)


def _translate(exc: Exception) -> ModelError:
    """
    Exceção de SDK vira :class:`ModelError`, preservando se vale retentar.

    A cadeia é por nome de classe em vez de ``except`` tipado porque ``anthropic`` pode
    não estar instalado — e um ``import`` no topo deste helper reintroduziria a dependência
    que o construtor toma o cuidado de deixar preguiçosa. O que interessa ao loop é a
    distinção entre "tenta de novo" e "não adianta", não a taxonomia do SDK.
    """
    name = type(exc).__name__
    retryable = name in {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
    }
    if name == "APIStatusError":
        retryable = getattr(exc, "status_code", 0) >= 500
    return ModelError(f"{name}: {exc}", retryable=retryable)


def to_messages(conversation: Sequence[Turn]) -> list[dict[str, Any]]:
    """
    Converte a conversa canônica no formato de mensagem da API.

    O turno de assistant volta pelos **blocos originais** quando o loop os guardou em
    ``raw['replay_content']``. Não é otimização: um bloco de pensamento carrega uma
    assinatura do provedor, e reconstruí-lo a partir do texto canônico entrega um bloco
    sem assinatura, que a API descarta. O canônico é a visão de medição; o replay quer
    fidelidade de byte. Ver :class:`~harness_eng.core.ports.ModelResponse`.
    """
    messages: list[dict[str, Any]] = []
    for turn in conversation:
        if turn.role is Role.ASSISTANT:
            replay = turn.raw.get("replay_content") if turn.raw else None
            if replay:
                messages.append({"role": "assistant", "content": replay})
                continue
            blocks: list[dict[str, Any]] = []
            if turn.text:
                blocks.append({"type": "text", "text": turn.text})
            blocks.extend(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": dict(call.arguments),
                }
                for call in turn.tool_calls
            )
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue

        if turn.tool_results:
            # Todos os resultados numa mensagem só. Ver a nota sobre resultados paralelos
            # em ``core/loop.py``: partir isso é aceito pela API e degrada o modelo em
            # silêncio.
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": result.call_id,
                            "content": result.content,
                            "is_error": result.is_error,
                        }
                        for result in turn.tool_results
                    ],
                }
            )
        elif turn.text:
            messages.append({"role": "user", "content": turn.text})
    return messages


def from_message(message: Any) -> ModelResponse:
    """
    Traduz a resposta da API para o vocabulário canônico.

    Percorre os blocos em vez de assumir que o primeiro é texto. A lição está registrada
    no README: a primeira versão do adapter do Claude Code lia só ``type == "text"`` e
    reportou "58 de 60 chamadas voltaram vazias" — um achado alarmante sobre o harness
    que era, na verdade, defeito da ferramenta de medição.
    """
    texts: list[str] = []
    thoughts: list[str] = []
    calls: list[ToolCall] = []

    for block in getattr(message, "content", None) or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            texts.append(getattr(block, "text", "") or "")
        elif kind == "thinking":
            thoughts.append(getattr(block, "thinking", "") or "")
        elif kind == "tool_use":
            calls.append(
                ToolCall(
                    id=getattr(block, "id", ""),
                    name=getattr(block, "name", ""),
                    arguments=getattr(block, "input", None) or {},
                )
            )

    stop_details: dict[str, Any] = {}
    details = getattr(message, "stop_details", None)
    if details is not None:
        stop_details = {
            "category": getattr(details, "category", None),
            "explanation": getattr(details, "explanation", None),
        }

    return ModelResponse(
        text="\n".join(t for t in texts if t),
        thinking="\n".join(t for t in thoughts if t),
        tool_calls=tuple(calls),
        usage=_usage(getattr(message, "usage", None)),
        model=getattr(message, "model", None),
        stop_reason=StopReason.parse(getattr(message, "stop_reason", None)),
        replay_content=getattr(message, "content", None),
        stop_details=stop_details,
    )


class OpenAIClient:
    """
    Fala o formato *chat completions*. Implementa :class:`~harness_eng.core.ports.ModelClient`.

    Na prática isto é "qualquer IA": além da OpenAI, o mesmo formato é o que Ollama, vLLM,
    Groq, Together, OpenRouter e a maioria dos servidores locais expõem. ``base_url``
    aponta para qualquer um deles::

        OpenAIClient(model="llama3", base_url="http://localhost:11434/v1", api_key="ollama")

    Este adapter é o teste que o README prometia: a estrutura só é agnóstica se aguentar
    um segundo formato sem que o loop, as métricas ou o trace mudem. Aguentou — e as três
    diferenças reais que apareceram estão marcadas no código, porque são exatamente o tipo
    de detalhe que um formato canônico existe para absorver:

    1. resultado de ferramenta é **uma mensagem por resultado**, não uma com vários blocos;
    2. os argumentos chegam como **string JSON**, não como dicionário;
    3. ``prompt_tokens`` já **inclui** os tokens lidos de cache — somar os dois contaria em
       dobro, e o tamanho de contexto medido sairia inflado.
    """

    def __init__(
        self,
        *,
        model: str,
        system: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any = None,
    ) -> None:
        self._model = model
        self._system = system
        self._max_tokens = max_tokens

        if client is not None:
            self._client = client
            return
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depende do ambiente
            raise ModelError(
                "o SDK da OpenAI não está instalado. "
                'Instale o extra: pip install -e ".[harness]"'
            ) from exc
        options: dict[str, Any] = {}
        if api_key:
            options["api_key"] = api_key
        if base_url:
            options["base_url"] = base_url
        self._client = openai.OpenAI(**options)

    @property
    def model(self) -> str:
        return self._model

    def complete(self, conversation: Sequence[Turn], tools: Sequence[ToolSpec]) -> ModelResponse:
        messages = to_openai_messages(conversation)
        if self._system:
            messages.insert(0, {"role": "system", "content": self._system})

        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_completion_tokens": self._max_tokens,
        }
        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": dict(spec.input_schema),
                    },
                }
                for spec in tools
            ]

        try:
            completion = self._client.chat.completions.create(**request)
        except Exception as exc:  # noqa: BLE001 — traduzido logo abaixo
            raise _translate(exc) from exc

        return from_completion(completion)


def to_openai_messages(conversation: Sequence[Turn]) -> list[dict[str, Any]]:
    """
    Conversa canônica → mensagens no formato *chat completions*.

    Diferença número 1 do docstring da classe: um turno de user canônico que carrega três
    resultados de ferramenta vira **três** mensagens ``role="tool"``. Não é perda — é a
    mesma informação em outra forma, e o formato canônico guarda a forma que preserva o
    agrupamento porque dá para derivar a outra a partir dela, não o contrário.
    """
    messages: list[dict[str, Any]] = []
    for turn in conversation:
        if turn.role is Role.ASSISTANT:
            message: dict[str, Any] = {"role": "assistant", "content": turn.text or None}
            if turn.tool_calls:
                message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(dict(call.arguments), ensure_ascii=False),
                        },
                    }
                    for call in turn.tool_calls
                ]
            if message["content"] is not None or "tool_calls" in message:
                messages.append(message)
            continue

        if turn.tool_results:
            messages.extend(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": result.content or "(sem saída)",
                }
                for result in turn.tool_results
            )
        elif turn.text:
            messages.append({"role": "user", "content": turn.text})
    return messages


#: ``finish_reason`` do formato chat completions → motivo canônico de parada.
_FINISH_REASONS = {
    "stop": StopReason.END_TURN,
    "length": StopReason.MAX_TOKENS,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "content_filter": StopReason.REFUSAL,
}


def from_completion(completion: Any) -> ModelResponse:
    """
    Resposta *chat completions* → vocabulário canônico.

    Diferença número 2: ``function.arguments`` é uma **string JSON**. Repassá-la como está
    faria ``ToolCall.arguments`` virar uma string onde o resto do pacote espera um mapa —
    e o detector de loop, que assina a chamada pelos argumentos ordenados, deixaria de
    casar repetições entre provedores.
    """
    choices = getattr(completion, "choices", None) or []
    if not choices:
        return ModelResponse(
            stop_reason=StopReason.UNKNOWN, model=getattr(completion, "model", None)
        )

    choice = choices[0]
    message = getattr(choice, "message", None)
    calls: list[ToolCall] = []
    for raw in (getattr(message, "tool_calls", None) or []):
        function = getattr(raw, "function", None)
        try:
            arguments = json.loads(getattr(function, "arguments", "") or "{}")
        except json.JSONDecodeError:
            # Argumento inválido é do modelo, não do adapter. Vira chamada com mapa vazio
            # e o executor devolve erro legível — melhor que estourar a sessão inteira.
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(
            ToolCall(
                id=getattr(raw, "id", ""),
                name=getattr(function, "name", ""),
                arguments=arguments,
            )
        )

    return ModelResponse(
        text=getattr(message, "content", None) or "",
        thinking=getattr(message, "reasoning_content", None) or "",
        tool_calls=tuple(calls),
        usage=_openai_usage(getattr(completion, "usage", None)),
        model=getattr(completion, "model", None),
        stop_reason=_FINISH_REASONS.get(getattr(choice, "finish_reason", None), StopReason.UNKNOWN),
        # Sem concessão de replay aqui: este formato não assina o bloco de raciocínio, então
        # reconstruir a mensagem a partir do canônico é fiel. A concessão em
        # ``ModelResponse.replay_content`` é específica de quem assina — ver ports.py.
        replay_content=None,
    )


def _openai_usage(raw: Any) -> Usage | None:
    """
    Uso do formato chat completions → :class:`Usage` canônico.

    Diferença número 3, e a que erra em silêncio: ``prompt_tokens`` **já inclui** os
    tokens lidos de cache. Como ``Usage.context_size`` soma input + cache lido + cache
    escrito, repassar ``prompt_tokens`` cru contaria o cache duas vezes e inflaria o
    tamanho de contexto medido — num relatório cujo assunto é justamente crescimento de
    contexto. Por isso o cache sai do input aqui.
    """
    if raw is None:
        return None
    prompt = getattr(raw, "prompt_tokens", 0) or 0
    details = getattr(raw, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    return Usage(
        input_tokens=max(prompt - cached, 0),
        output_tokens=getattr(raw, "completion_tokens", 0) or 0,
        cache_write_tokens=0,  # o formato não reporta escrita de cache separadamente
        cache_read_tokens=cached,
    )


def _usage(raw: Any) -> Usage | None:
    """
    ``None`` quando a resposta não trouxe uso — ausência é ausência, não zero.

    ``or 0`` em cada campo porque a API omite ou manda ``null`` nos campos de cache quando
    não houve cache, e ``None`` numa soma de inteiros quebra a agregação da sessão.
    """
    if raw is None:
        return None
    return Usage(
        input_tokens=getattr(raw, "input_tokens", 0) or 0,
        output_tokens=getattr(raw, "output_tokens", 0) or 0,
        cache_write_tokens=getattr(raw, "cache_creation_input_tokens", 0) or 0,
        cache_read_tokens=getattr(raw, "cache_read_input_tokens", 0) or 0,
        service_tier=getattr(raw, "service_tier", None),
    )
