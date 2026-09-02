"""
Detecção de loop patológico: quando o agente repete sem progredir.

O modo de falha mais caro de um harness, e o menos visível em qualquer painel que só
conte erro. O agente repete a mesma chamada, recebe a mesma resposta, e segue — cada
iteração custa contexto e dinheiro, e nenhuma delas registra falha.

Nos 54 transcripts que motivaram este repositório havia **124 comandos repetidos 4 ou
mais vezes na mesma sessão**, um deles **95 vezes**. Nenhuma ferramenta existente
reportava isso.

Três padrões, porque são causas diferentes e pedem correções diferentes:

* **repetição exata** — mesma ferramenta, mesmos argumentos. Quase sempre o agente sem
  sinal de que a chamada não mudou nada.
* **retry cego** — erro seguido da mesma chamada, sem alterar argumento. O agente não
  leu o erro, ou o erro não dizia o que corrigir.
* **oscilação** — alternância entre duas chamadas (A, B, A, B). Duas correções que se
  desfazem, ou dois estados que o agente não consegue distinguir.

Puro sobre o formato canônico: nenhum acesso a disco, nenhuma dependência pesada.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from ..trace.model import Session, ToolCall, TraceSet

#: Repetições a partir das quais vale reportar. Duas chamadas iguais são normais (ler um
#: arquivo antes e depois de editar); quatro já é padrão.
DEFAULT_MIN_REPEATS = 4

#: Janela de chamadas dentro da qual a repetição conta como loop. Sem janela, um comando
#: legítimo repetido no início e no fim de uma sessão de 800 chamadas vira falso positivo.
DEFAULT_WINDOW = 40


@dataclass(frozen=True, slots=True)
class RepeatedCall:
    """Uma chamada que se repetiu além do razoável dentro de uma janela."""

    session_id: str
    tool: str
    signature: str
    count: int
    turn_indices: tuple[int, ...]
    sample_argument: str | None = None

    @property
    def span(self) -> int:
        """Distância em turnos entre a primeira e a última repetição."""
        return self.turn_indices[-1] - self.turn_indices[0] if self.turn_indices else 0

    @property
    def density(self) -> float:
        """
        Repetições por turno no intervalo em que ocorreram.

        Separa "repetiu 5 vezes em 8 turnos" (loop apertado, patológico) de "repetiu 5
        vezes em 400 turnos" (provavelmente trabalho legítimo). A contagem sozinha não
        distingue os dois, e é por isso que ela sozinha gera relatório cheio de ruído.
        """
        return self.count / max(1, self.span)

    def as_dict(self, redact: bool = False) -> dict:
        return {
            "session": self.session_id,
            "tool": self.tool,
            "count": self.count,
            "span_turns": self.span,
            "density": round(self.density, 3),
            "argument": None if redact else self.sample_argument,
        }


@dataclass(frozen=True, slots=True)
class BlindRetry:
    """
    Erro seguido de repetição idêntica: o agente não usou a informação do erro.

    É o padrão mais acionável dos três. Se acontece muito com uma ferramenta específica,
    a mensagem de erro dela não diz o que corrigir — e isso é conserto de harness, não
    de modelo.
    """

    session_id: str
    tool: str
    signature: str
    attempts: int
    turn_indices: tuple[int, ...]
    sample_argument: str | None = None

    def as_dict(self, redact: bool = False) -> dict:
        return {
            "session": self.session_id,
            "tool": self.tool,
            "attempts": self.attempts,
            "argument": None if redact else self.sample_argument,
        }


@dataclass(frozen=True, slots=True)
class Oscillation:
    """Alternância entre duas chamadas — duas correções que se desfazem."""

    session_id: str
    tool_a: str
    tool_b: str
    cycles: int
    turn_indices: tuple[int, ...]

    def as_dict(self, redact: bool = False) -> dict:
        return {
            "session": self.session_id,
            "between": [self.tool_a, self.tool_b],
            "cycles": self.cycles,
        }


@dataclass(frozen=True, slots=True)
class LoopReport:
    """O que a análise de loop encontrou num conjunto de traces."""

    repeats: tuple[RepeatedCall, ...] = ()
    blind_retries: tuple[BlindRetry, ...] = ()
    oscillations: tuple[Oscillation, ...] = ()
    total_calls: int = 0

    @property
    def wasted_calls(self) -> int:
        """
        Chamadas atribuíveis a repetição, contando só as depois da primeira.

        Limite inferior deliberado: a primeira chamada de cada repetição era legítima, e
        a oscilação não entra porque pode ser trabalho real alternando entre dois
        arquivos. Um número que exagera o desperdício é tão inútil quanto um que o ignora.
        """
        return sum(r.count - 1 for r in self.repeats)

    @property
    def waste_rate(self) -> float | None:
        return self.wasted_calls / self.total_calls if self.total_calls else None

    def worst(self, limit: int = 10) -> list[RepeatedCall]:
        return sorted(self.repeats, key=lambda r: (-r.count, -r.density))[:limit]

    def by_tool(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for repeat in self.repeats:
            counter[repeat.tool] += repeat.count - 1
        return dict(counter.most_common())

    def as_dict(self, redact: bool = False) -> dict:
        return {
            "total_calls": self.total_calls,
            "repeated_patterns": len(self.repeats),
            "wasted_calls_lower_bound": self.wasted_calls,
            "waste_rate": round(self.waste_rate, 4) if self.waste_rate is not None else None,
            "blind_retries": len(self.blind_retries),
            "oscillations": len(self.oscillations),
            "waste_by_tool": self.by_tool(),
            "worst": [r.as_dict(redact) for r in self.worst()],
        }


def detect_loops(
    traces: TraceSet | Sequence[Session],
    min_repeats: int = DEFAULT_MIN_REPEATS,
    window: int = DEFAULT_WINDOW,
) -> LoopReport:
    """Roda os três detectores sobre um conjunto de traces."""
    sessions = traces.sessions if isinstance(traces, TraceSet) else tuple(traces)

    repeats: list[RepeatedCall] = []
    retries: list[BlindRetry] = []
    oscillations: list[Oscillation] = []
    total = 0

    for session in sessions:
        calls = session.tool_calls()
        total += len(calls)
        repeats.extend(_detect_repeats(session, calls, min_repeats, window))
        retries.extend(_detect_blind_retries(session, min_attempts=2))
        oscillations.extend(_detect_oscillation(session, calls, min_cycles=3))

    return LoopReport(
        repeats=tuple(repeats),
        blind_retries=tuple(retries),
        oscillations=tuple(oscillations),
        total_calls=total,
    )


def _detect_repeats(
    session: Session, calls: Sequence[ToolCall], min_repeats: int, window: int
) -> list[RepeatedCall]:
    """
    Assinaturas que aparecem ``min_repeats`` ou mais vezes dentro de uma janela.

    A janela é sobre POSIÇÃO na sequência de chamadas, não sobre turno: o que caracteriza
    loop é a densidade de repetição no fluxo de trabalho do agente, e turnos variam muito
    de tamanho.
    """
    positions: dict[str, list[int]] = defaultdict(list)
    for position, call in enumerate(calls):
        positions[call.signature()].append(position)

    found: list[RepeatedCall] = []
    for signature, occurrences in positions.items():
        if len(occurrences) < min_repeats:
            continue
        # Maior aglomerado dentro da janela — evita contar repetições espalhadas pela
        # sessão inteira, que costumam ser trabalho legítimo e não loop.
        best_start = best_count = 0
        for i, start in enumerate(occurrences):
            j = i
            while j < len(occurrences) and occurrences[j] - start <= window:
                j += 1
            if j - i > best_count:
                best_count, best_start = j - i, i
        if best_count < min_repeats:
            continue

        cluster = occurrences[best_start : best_start + best_count]
        example = calls[cluster[0]]
        found.append(
            RepeatedCall(
                session_id=session.id,
                tool=example.name,
                signature=signature,
                count=best_count,
                turn_indices=tuple(calls[p].turn_index for p in cluster),
                sample_argument=example.primary_argument,
            )
        )
    return found


def _detect_blind_retries(session: Session, min_attempts: int = 2) -> list[BlindRetry]:
    """
    Chamadas idênticas que erraram mais de uma vez.

    A condição é forte de propósito: mesma assinatura E resultado de erro. Uma chamada
    que erra, é corrigida e roda de novo tem assinatura diferente e não entra aqui —
    é o comportamento certo, não um loop.
    """
    results = session.results_by_call_id()
    failures: dict[str, list[ToolCall]] = defaultdict(list)

    for call in session.tool_calls():
        result = results.get(call.id)
        if result is not None and result.is_error:
            failures[call.signature()].append(call)

    return [
        BlindRetry(
            session_id=session.id,
            tool=calls[0].name,
            signature=signature,
            attempts=len(calls),
            turn_indices=tuple(c.turn_index for c in calls),
            sample_argument=calls[0].primary_argument,
        )
        for signature, calls in failures.items()
        if len(calls) >= min_attempts
    ]


def _detect_oscillation(
    session: Session, calls: Sequence[ToolCall], min_cycles: int = 3
) -> list[Oscillation]:
    """
    Padrão A,B,A,B,A — o agente alternando entre dois estados sem convergir.

    Detectado sobre assinatura, não sobre nome de ferramenta: dois `Edit` no mesmo
    arquivo desfazendo um ao outro é o caso interessante, e comparar só por nome de
    ferramenta o perderia inteiro.
    """
    found: list[Oscillation] = []
    signatures = [c.signature() for c in calls]

    index = 0
    while index + 3 < len(signatures):
        a, b = signatures[index], signatures[index + 1]
        if a == b:
            index += 1
            continue
        cycles = 0
        cursor = index
        while (
            cursor + 1 < len(signatures)
            and signatures[cursor] == a
            and signatures[cursor + 1] == b
        ):
            cycles += 1
            cursor += 2
        if cycles >= min_cycles:
            found.append(
                Oscillation(
                    session_id=session.id,
                    tool_a=calls[index].name,
                    tool_b=calls[index + 1].name,
                    cycles=cycles,
                    turn_indices=tuple(
                        calls[p].turn_index for p in range(index, min(cursor, len(calls)))
                    ),
                )
            )
            index = cursor
        else:
            index += 1
    return found
