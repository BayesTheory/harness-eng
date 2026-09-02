"""
Saúde de ferramenta: onde o harness realmente falha.

A métrica mais barata e mais reveladora do repositório. Nos 54 transcripts que motivaram
o projeto, PowerShell errou **14,3%** das chamadas contra **3,0%** do Bash — cinco vezes
mais, num harness onde as duas ferramentas fazem essencialmente o mesmo trabalho. Nenhuma
ferramenta existente reporta isso, e é acionável: ou a descrição da ferramenta induz ao
erro, ou ela não devia estar exposta.

Puro sobre o formato canônico. Não importa ``anthropic``, ``openai`` nem toca disco.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ..trace.model import Session, ToolCall, ToolResult, TraceSet

#: Abaixo disto a taxa de erro é ruído amostral e não vale reportar como sinal. Com 5
#: chamadas, um erro vira "20% de falha" e induz conclusão que o dado não sustenta.
MIN_CALLS_FOR_SIGNAL = 30


@dataclass(frozen=True, slots=True)
class ToolHealth:
    """Comportamento observado de uma ferramenta ao longo de um conjunto de traces."""

    name: str
    calls: int = 0
    errors: int = 0
    empty_results: int = 0
    interrupted: int = 0
    unanswered: int = 0

    @property
    def error_rate(self) -> float | None:
        """
        Fração de chamadas que voltaram erro. ``None`` quando não houve chamada.

        ``None`` e não 0.0: uma ferramenta nunca chamada não tem taxa de erro zero, tem
        taxa indefinida. Zero significa "chamada e sempre bem-sucedida", que é outra
        coisa — e a diferença muda a leitura do relatório.
        """
        return self.errors / self.calls if self.calls else None

    @property
    def has_signal(self) -> bool:
        """Se há chamadas suficientes para a taxa significar alguma coisa."""
        return self.calls >= MIN_CALLS_FOR_SIGNAL

    @property
    def silent_failure_rate(self) -> float | None:
        """
        Sucesso sem saída nenhuma.

        Vale separado de erro porque falha silenciosa é pior: o modelo recebe "ok" e
        segue sem o sinal de que precisava. É um jeito de o loop travar sem nunca
        registrar erro — e portanto sem aparecer em nenhum painel que só conte erro.
        """
        return self.empty_results / self.calls if self.calls else None

    def as_dict(self) -> dict:
        return {
            "tool": self.name,
            "calls": self.calls,
            "errors": self.errors,
            "error_rate": round(self.error_rate, 4) if self.error_rate is not None else None,
            "empty_results": self.empty_results,
            "interrupted": self.interrupted,
            "unanswered": self.unanswered,
            "has_signal": self.has_signal,
        }


@dataclass(frozen=True, slots=True)
class ToolHealthReport:
    """Saúde de todas as ferramentas, mais o que dá para concluir do conjunto."""

    tools: Mapping[str, ToolHealth] = field(default_factory=dict)

    @property
    def total_calls(self) -> int:
        return sum(t.calls for t in self.tools.values())

    @property
    def total_errors(self) -> int:
        return sum(t.errors for t in self.tools.values())

    @property
    def overall_error_rate(self) -> float | None:
        return self.total_errors / self.total_calls if self.total_calls else None

    def ranked_by_error_rate(self, only_with_signal: bool = True) -> list[ToolHealth]:
        """
        Ferramentas da pior para a melhor, entre as que têm amostra suficiente.

        O filtro é o que separa relatório útil de lista de ruído: sem ele, o topo do
        ranking é sempre uma ferramenta chamada duas vezes que errou uma.
        """
        candidates = [
            t for t in self.tools.values()
            if t.error_rate is not None and (t.has_signal or not only_with_signal)
        ]
        return sorted(candidates, key=lambda t: (-t.error_rate, -t.calls))

    def outliers(self, factor: float = 2.0) -> list[ToolHealth]:
        """
        Ferramentas que erram muito acima da média geral do harness.

        ``factor=2.0`` significa "erra pelo menos o dobro da média". É comparação contra
        o próprio harness, não contra um limiar absoluto: o que interessa é qual
        ferramenta está fora do padrão *daquele* sistema, e um limiar fixo diria que um
        harness inteiro é ruim sem apontar onde.
        """
        baseline = self.overall_error_rate
        if not baseline:
            return []
        return [
            t for t in self.ranked_by_error_rate()
            if t.error_rate is not None and t.error_rate >= baseline * factor
        ]

    def as_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
            "overall_error_rate": (
                round(self.overall_error_rate, 4) if self.overall_error_rate is not None else None
            ),
            "tools": [t.as_dict() for t in self.ranked_by_error_rate(only_with_signal=False)],
            "outliers": [t.name for t in self.outliers()],
        }


def analyse_tools(traces: TraceSet | Sequence[Session]) -> ToolHealthReport:
    """
    Calcula a saúde de cada ferramenta sobre um conjunto de traces.

    O pareamento chamada↔resultado é o passo que a maioria das análises caseiras erra: o
    erro chega num turno de *user*, e o nome da ferramenta está no turno de *assistant*
    anterior. Sem casar por ``tool_use_id`` a taxa sai atribuída à ferramenta errada — ou,
    pior, atribuída a "desconhecido" e descartada.
    """
    sessions = traces.sessions if isinstance(traces, TraceSet) else tuple(traces)

    calls: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    empty: Counter[str] = Counter()
    interrupted: Counter[str] = Counter()
    unanswered: Counter[str] = Counter()

    for session in sessions:
        for call, result in session.paired_calls():
            calls[call.name] += 1
            if result is None:
                # Chamada sem resultado: sessão interrompida, ou o harness abandonou a
                # chamada. Não é sucesso, não é erro — é uma terceira coisa, e some
                # completamente de qualquer análise que só conte is_error.
                unanswered[call.name] += 1
                continue
            if result.is_error:
                errors[call.name] += 1
            elif result.is_empty:
                empty[call.name] += 1
            if result.interrupted:
                interrupted[call.name] += 1

    return ToolHealthReport(
        tools={
            name: ToolHealth(
                name=name,
                calls=count,
                errors=errors[name],
                empty_results=empty[name],
                interrupted=interrupted[name],
                unanswered=unanswered[name],
            )
            for name, count in calls.items()
        }
    )


def error_rate_by_session(
    traces: TraceSet | Sequence[Session], tool: str | None = None
) -> dict[str, float]:
    """
    Taxa de erro por sessão, opcionalmente de uma ferramenta só.

    É a entrada da camada estatística: comparar dois harnesses exige uma observação por
    unidade experimental (a sessão), não um número agregado sobre tudo. Colapsar tudo
    numa taxa global joga fora a variância, e sem variância não há intervalo de confiança
    nem tamanho de efeito — só uma diferença entre dois números, que não é evidência.
    """
    sessions = traces.sessions if isinstance(traces, TraceSet) else tuple(traces)
    rates: dict[str, float] = {}
    for session in sessions:
        total = failed = 0
        for call, result in session.paired_calls():
            if tool is not None and call.name != tool:
                continue
            total += 1
            if result is not None and result.is_error:
                failed += 1
        if total:
            rates[session.id] = failed / total
    return rates


def calls_by_tool(traces: TraceSet | Sequence[Session]) -> dict[str, list[ToolCall]]:
    """Chamadas agrupadas por ferramenta, para inspeção qualitativa dos erros."""
    sessions = traces.sessions if isinstance(traces, TraceSet) else tuple(traces)
    grouped: dict[str, list[ToolCall]] = defaultdict(list)
    for session in sessions:
        for call in session.tool_calls():
            grouped[call.name].append(call)
    return dict(grouped)


def failing_calls(
    traces: TraceSet | Sequence[Session], tool: str | None = None, limit: int = 20
) -> list[tuple[ToolCall, ToolResult]]:
    """
    Exemplos concretos de chamada que falhou.

    Uma taxa diz que existe problema; um exemplo diz qual é. O relatório precisa dos dois,
    e este é o único ponto do módulo que devolve conteúdo cru de chamada — por isso a
    redação, quando pedida, se aplica aqui.
    """
    sessions = traces.sessions if isinstance(traces, TraceSet) else tuple(traces)
    found: list[tuple[ToolCall, ToolResult]] = []
    for session in sessions:
        for call, result in session.paired_calls():
            if result is None or not result.is_error:
                continue
            if tool is not None and call.name != tool:
                continue
            found.append((call, result))
            if len(found) >= limit:
                return found
    return found
