"""
O loop do harness mínimo.

Um agente é, no fundo, um ``while``: pede ao modelo, executa o que ele pediu, devolve o
resultado, repete. O que separa um harness de outro não é o ``while`` — é o que cada um
faz nas bordas. Este módulo trata as bordas como o assunto principal, porque é onde os
modos de falha que o resto do pacote mede realmente nascem:

* **parar cedo demais.** ``pause_turn`` não é fim de turno. Tratar tudo que não é
  ``tool_use`` como "acabou" devolve trabalho pela metade sem erro nenhum.
* **parar tarde demais.** Sem teto de iteração, um modelo que nunca emite ``end_turn``
  roda até a conta acabar. O teto existe, e bater nele **não é sucesso** — é um estado
  próprio, que sai no trace e no código de saída.
* **executar chamada truncada.** Se a resposta foi cortada por ``max_tokens``, o último
  bloco de ferramenta pode estar pela metade. Executá-lo é rodar um comando que o modelo
  não terminou de escrever.
* **partir os resultados paralelos.** Um turno de assistant pode pedir várias ferramentas
  de uma vez; os resultados voltam **todos numa mensagem só**. Espalhá-los em mensagens
  separadas é sintaticamente aceito e ensina o modelo, silenciosamente, a parar de pedir
  chamadas em paralelo — uma regressão de desempenho que nenhum teste de unidade pega.

O loop não importa provedor: recebe um :class:`~harness_eng.core.ports.ModelClient` no
construtor. É o que permite testar cada um dos comportamentos acima sem chave de API.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

from ..trace.model import Role, Session, StopReason, ToolCall, ToolResult, Turn
from .ports import ModelClient, ModelError
from .tools import ToolRegistry

#: Teto padrão de iterações. Não é número mágico com pretensão de ser ótimo: é um teto
#: para o modo de falha "o modelo nunca emite ``end_turn``", que é real e não hipotético.
DEFAULT_MAX_ITERATIONS = 50

#: Origem gravada em ``Session.source``. O harness deste repositório emite trace nativo —
#: sem adapter no meio, portanto sem a dúvida de quanto do número é tradução.
SOURCE = "native"


class LoopStatus(str, Enum):
    """
    Como a execução terminou. Estado medido, não suposto.

    Existem cinco valores porque existem cinco desfechos distintos, e achatá-los em
    ``sucesso``/``falha`` apagaria a diferença entre "o modelo terminou" e "eu desliguei o
    modelo no meio" — que é precisamente a diferença que interessa a quem está medindo.
    """

    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    TRUNCATED = "truncated"
    REFUSED = "refused"
    MODEL_ERROR = "model_error"

    @property
    def finished_on_its_own(self) -> bool:
        """Se o modelo terminou por vontade própria. Só um valor qualifica."""
        return self is LoopStatus.COMPLETED


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """
    O resultado de uma execução: a sessão canônica e como ela acabou.

    A sessão sozinha não basta. Um trace que bateu no teto de iterações é indistinguível,
    turno a turno, de um que terminou normalmente — a diferença está só no motivo da
    última parada, e é justamente o que se perde quando o harness devolve apenas a
    conversa.
    """

    session: Session
    status: LoopStatus
    iterations: int
    detail: str = ""

    @property
    def turns(self) -> int:
        return len(self.session)

    @property
    def tool_calls(self) -> int:
        return len(self.session.tool_calls())

    def as_dict(self) -> dict:
        return {
            "status": self.status.value,
            "iterations": self.iterations,
            "detail": self.detail,
            "session": self.session.as_dict(),
        }


class AgentLoop:
    """
    Pede, executa, devolve, repete — e registra tudo no formato canônico enquanto isso.

    O trace **é** a conversa: a mesma tupla de :class:`~harness_eng.trace.model.Turn` que
    vai para o modelo é a que sai no relatório. Um harness que mantém uma lista de
    mensagens para a API e um log separado para a análise tem dois estados que podem
    divergir, e quando divergem é o log que mente — sempre no sentido de parecer melhor
    do que foi, porque ninguém escreve código de log que invente falha.
    """

    def __init__(
        self,
        client: ModelClient,
        registry: ToolRegistry,
        *,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        cwd: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations precisa ser pelo menos 1")
        self._client = client
        self._registry = registry
        self._max_iterations = max_iterations
        # Onde a execução aconteceu. Campo canônico, e o que permite comparar sessões do
        # mesmo projeto entre harnesses — sem ele o trace nativo entra na análise sem a
        # dimensão que agrupa o resto.
        self._cwd = cwd
        # Injetável para o teste controlar duração sem dormir. Padrão consciente de fuso:
        # o adapter do Claude Code produz datetime com fuso, e misturar aware com naive
        # levanta TypeError na primeira comparação — numa métrica de duração, meses depois.
        self._now = clock or (lambda: datetime.now(timezone.utc))

    def run(self, prompt: str, *, session_id: str | None = None) -> RunOutcome:
        """Roda uma tarefa até o fim, ou até um dos desfechos de :class:`LoopStatus`."""
        started = self._now()
        turns: list[Turn] = [
            Turn(index=0, role=Role.USER, timestamp=started, text=prompt)
        ]
        specs = self._registry.specs
        status = LoopStatus.MAX_ITERATIONS
        detail = f"teto de {self._max_iterations} iterações atingido sem end_turn"
        iteration = 0

        while iteration < self._max_iterations:
            iteration += 1

            try:
                response = self._client.complete(tuple(turns), specs)
            except ModelError as exc:
                # Registrada como turno para não sumir do trace: uma execução que morreu
                # na terceira chamada é dado, e apagá-la deixaria o custo medido sem a
                # explicação de por que a sessão é curta.
                turns.append(
                    Turn(
                        index=len(turns),
                        role=Role.ASSISTANT,
                        timestamp=self._now(),
                        text=str(exc),
                        stop_reason=StopReason.ERROR,
                        model=self._client.model,
                    )
                )
                status, detail = LoopStatus.MODEL_ERROR, str(exc)
                break

            index = len(turns)
            calls = tuple(replace(call, turn_index=index) for call in response.tool_calls)
            turns.append(
                Turn(
                    index=index,
                    role=Role.ASSISTANT,
                    timestamp=self._now(),
                    text=response.text,
                    thinking=response.thinking,
                    tool_calls=calls,
                    usage=response.usage,
                    model=response.model or self._client.model,
                    stop_reason=response.stop_reason,
                    # Blocos crus do provedor, guardados só para o replay do turno
                    # seguinte precisar da assinatura do bloco de pensamento. Não são
                    # gravados no trace: ver ``ModelResponse.replay_content``.
                    raw={"replay_content": response.replay_content},
                )
            )

            if response.stop_reason is StopReason.REFUSAL:
                status = LoopStatus.REFUSED
                detail = str(response.stop_details.get("category") or "recusa sem categoria")
                break

            if response.stop_reason is StopReason.MAX_TOKENS:
                # Antes do ramo de ferramenta de propósito: com a resposta cortada, o
                # último bloco de ferramenta pode estar incompleto, e executar argumento
                # pela metade é pior que parar.
                status = LoopStatus.TRUNCATED
                detail = "resposta cortada por max_tokens"
                break

            if calls:
                turns.append(self._execute(calls, len(turns)))
                continue

            if response.stop_reason is StopReason.PAUSE_TURN:
                # O modelo pausou um turno longo. Segue sem turno de user: a próxima
                # requisição leva o que ele já produziu e ele retoma dali.
                continue

            status = LoopStatus.COMPLETED
            detail = response.stop_reason.value
            break

        ended = self._now()
        session = Session(
            id=session_id or f"{SOURCE}-{started.strftime('%Y%m%dT%H%M%S')}",
            source=SOURCE,
            turns=tuple(turns),
            cwd=self._cwd,
            started_at=started,
            ended_at=ended,
            metadata={
                "status": status.value,
                "detail": detail,
                "iterations": iteration,
                "max_iterations": self._max_iterations,
                "model": self._client.model,
                "tools": [spec.name for spec in specs],
            },
        )
        return RunOutcome(session=session, status=status, iterations=iteration, detail=detail)

    def _execute(self, calls: tuple[ToolCall, ...], index: int) -> Turn:
        """
        Executa todas as chamadas do turno e junta os resultados num **único** turno de user.

        A unicidade é o ponto. Ver a nota sobre resultados paralelos no topo do módulo:
        partir os resultados em mensagens separadas passa em qualquer teste que só olhe
        conteúdo, e degrada o comportamento do modelo sem deixar rastro.
        """
        results: list[ToolResult] = [self._registry.execute(call) for call in calls]
        return Turn(
            index=index,
            role=Role.USER,
            timestamp=self._now(),
            tool_results=tuple(results),
        )
