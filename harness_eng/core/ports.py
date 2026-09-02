"""
Portas do harness mínimo.

O harness deste repositório existe por uma razão só: **fechar o círculo**. Todo o resto
do pacote mede harnesses de terceiro através de um adapter, e um adapter é uma tradução —
sempre resta a dúvida de quanto do número é o harness e quanto é a tradução. Aqui não há
adapter: o loop escreve :class:`~harness_eng.trace.model.Session` diretamente, então o
trace é o registro primário e não uma reconstrução dele.

:class:`ModelClient` é a porta que mantém isso agnóstico. O loop nunca importa
``anthropic`` nem ``openai``; recebe um cliente pronto no construtor. Consequência
medível: a suíte exercita o loop inteiro — inclusive os modos de falha que só aparecem em
execução, como estourar o teto de iterações — sem chave de API e sem rede.

Nota de vocabulário: a porta fala em :class:`Turn`, o mesmo tipo que as métricas
consomem. Não é economia de tipo, é uma invariante: **a conversa é o trace**. Um harness
que mantém uma lista de mensagens para a API e outra estrutura para o log tem dois
estados que podem divergir — e quando divergem, o log mente sobre a execução. Aqui há um
estado só, e ele é o que sai no relatório.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..trace.model import StopReason, ToolCall, Turn, Usage


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """
    A declaração de uma ferramenta para o modelo: o que ela faz e o que aceita.

    Separada do executor de propósito. O que o modelo lê (:class:`ToolSpec`) e o que a
    máquina roda (o handler) são coisas diferentes, e tratá-las como uma só esconde a
    pergunta que este repositório quer poder fazer: *a descrição induziu ao erro, ou a
    implementação falhou?* Nos transcripts analisados o PowerShell errou 5x mais que o
    Bash fazendo o mesmo trabalho — e a resposta ali era a descrição, não o executor.
    """

    name: str
    description: str
    input_schema: Mapping[str, Any]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
        }


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """
    Uma resposta do modelo, já no vocabulário canônico.

    ``replay_content`` é a única concessão ao provedor, e é deliberada. Os campos
    canônicos (``text``, ``thinking``) são a visão de **medição**: texto legível, contável,
    comparável entre harnesses. Só que devolver o pensamento à API no turno seguinte exige
    o bloco **original**, com a assinatura que o provedor emitiu — reconstruí-lo a partir
    da string canônica perde essa assinatura, e o provedor descarta ou recusa o bloco.

    Ou seja: o formato canônico é lossy para *replay*, e é correto que seja. Medição quer
    o que dá para comparar entre provedores; replay quer fidelidade de byte com um
    provedor específico. Forçar um tipo a servir aos dois estraga os dois. O loop carrega
    ``replay_content`` sem olhar dentro, e o trace gravado no disco não o guarda — um
    trace é registro de medição, não checkpoint retomável.
    """

    text: str = ""
    thinking: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage | None = None
    model: str | None = None
    stop_reason: StopReason = StopReason.UNKNOWN
    #: Blocos crus do provedor, opacos para o loop. Ver docstring da classe.
    replay_content: Any = None
    #: Detalhe estruturado de recusa, quando o provedor manda um.
    stop_details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class ModelClient(Protocol):
    """
    Fala com um modelo. A única porta do harness que toca a rede.

    :meth:`complete` recebe a conversa inteira porque a API é sem estado — o harness
    reenvia o histórico a cada turno, e é justamente isso que torna a estabilidade do
    prefixo (e portanto o acerto de cache) uma propriedade do harness, não do modelo.
    """

    @property
    def model(self) -> str:
        """Identificador do modelo, gravado em cada turno do trace."""
        ...

    def complete(self, conversation: Sequence[Turn], tools: Sequence[ToolSpec]) -> ModelResponse:
        """
        Um turno de modelo sobre a conversa dada.

        Deve levantar :class:`ModelError` para falha de provedor — o loop a captura,
        registra como turno de erro e encerra com estado próprio. Deixar a exceção do SDK
        vazar acoplaria o loop ao provedor pela via de exceção, que é o jeito mais fácil
        de furar uma camada sem que nenhum import denuncie.
        """
        ...


class ModelError(RuntimeError):
    """
    Falha do provedor, traduzida.

    Existe para o loop poder distinguir "o modelo recusou" de "a rede caiu" de "o meu
    código tem bug" sem inspecionar tipo de exceção de SDK.
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable
