"""
Contexto: quanto o harness carrega, quão rápido cresce, quanto disso é cache.

A métrica que separa um harness que escala de um que trava na metade da tarefa. Contexto
cresce a cada turno; o que varia entre harnesses é **como** cresce e o que se paga por
isso.

Duas coisas que só aparecem medindo:

* **Crescimento por turno.** Nos transcripts analisados a mediana foi 0 e a média 1.030
  tokens, com p95 em 4.441. Mediana zero com média mil significa que o crescimento é
  concentrado em poucos turnos — quase sempre a leitura de um arquivo grande — e não uma
  subida suave. Otimizar a média aqui seria otimizar a coisa errada.
* **Eficiência de cache.** 96,5% de acerto nos mesmos transcripts. Escrever cache custa
  mais que input normal e ler custa uma fração; um harness que reescreve o prefixo a cada
  turno paga várias vezes pelo mesmo texto e nada no log diz isso.

Puro sobre o formato canônico.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

from ..trace.model import Session, StopReason, TraceSet


@dataclass(frozen=True, slots=True)
class ContextProfile:
    """Como o contexto se comportou ao longo de uma ou mais sessões."""

    samples: tuple[int, ...] = ()
    deltas: tuple[int, ...] = ()
    peak: int = 0
    truncations: int = 0
    turns_measured: int = 0

    @property
    def median_growth(self) -> float | None:
        """
        Crescimento mediano por turno. ``None`` sem amostra.

        Mediana e não média porque a distribuição é fortemente assimétrica: alguns turnos
        despejam um arquivo inteiro no contexto e dominam qualquer média. A média sozinha
        descreve mal o turno típico — é por isso que as duas saem no relatório.
        """
        return statistics.median(self.deltas) if self.deltas else None

    @property
    def mean_growth(self) -> float | None:
        return statistics.mean(self.deltas) if self.deltas else None

    @property
    def p95_growth(self) -> int | None:
        """
        O turno caro. É este número que decide quando o harness estoura a janela.

        Percentil calculado por ordenação direta em vez de interpolação: com poucos
        turnos a interpolação inventa um valor que nenhum turno teve, e aqui o que
        interessa é "existiu um turno assim", não uma estimativa suave.
        """
        if not self.deltas:
            return None
        ordered = sorted(self.deltas)
        return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]

    @property
    def growth_concentration(self) -> float | None:
        """
        Quanto do crescimento total vem dos 5% de turnos mais caros.

        Perto de 1.0 significa que o contexto cresce por poucos eventos grandes — e a
        alavanca é cortar esses eventos (ler menos arquivo, truncar saída). Perto de 0.05
        significa crescimento difuso, e aí a alavanca é outra: sumarização ou janela.
        Sem este número não dá para saber qual das duas otimizações vale.
        """
        if not self.deltas:
            return None
        positive = sorted((d for d in self.deltas if d > 0), reverse=True)
        if not positive:
            return None
        top = positive[: max(1, len(positive) // 20)]
        total = sum(positive)
        return sum(top) / total if total else None

    def as_dict(self) -> dict:
        return {
            "turns_measured": self.turns_measured,
            "peak_context": self.peak,
            "median_growth": round(self.median_growth, 1) if self.median_growth is not None else None,
            "mean_growth": round(self.mean_growth, 1) if self.mean_growth is not None else None,
            "p95_growth": self.p95_growth,
            "growth_concentration": (
                round(self.growth_concentration, 3)
                if self.growth_concentration is not None
                else None
            ),
            "truncations": self.truncations,
        }


@dataclass(frozen=True, slots=True)
class CacheProfile:
    """Eficiência de cache — a diferença entre pagar uma vez e pagar a cada turno."""

    read_tokens: int = 0
    write_tokens: int = 0
    fresh_input_tokens: int = 0

    @property
    def total_input(self) -> int:
        return self.read_tokens + self.write_tokens + self.fresh_input_tokens

    @property
    def hit_rate(self) -> float | None:
        return self.read_tokens / self.total_input if self.total_input else None

    @property
    def rewrite_ratio(self) -> float | None:
        """
        Tokens escritos em cache por token lido.

        O número que denuncia prefixo instável. Perto de 0 é ótimo: escreveu uma vez, leu
        muitas. Subindo, significa que o harness invalida o cache com frequência — cabeçalho
        que muda, ordem de mensagem que varia, timestamp no prompt. Todos consertáveis, e
        nenhum visível sem esta razão.
        """
        return self.write_tokens / self.read_tokens if self.read_tokens else None

    def as_dict(self) -> dict:
        return {
            "cache_read_tokens": self.read_tokens,
            "cache_write_tokens": self.write_tokens,
            "fresh_input_tokens": self.fresh_input_tokens,
            "hit_rate": round(self.hit_rate, 4) if self.hit_rate is not None else None,
            "rewrite_ratio": (
                round(self.rewrite_ratio, 4) if self.rewrite_ratio is not None else None
            ),
        }


def profile_context(traces: TraceSet | Sequence[Session]) -> ContextProfile:
    """
    Perfil de crescimento de contexto sobre um conjunto de traces.

    Deltas são calculados **dentro** de cada sessão, nunca entre sessões: a diferença
    entre o último turno de uma sessão e o primeiro da seguinte é um salto artificial que
    contaminaria toda a distribuição com valores negativos enormes.
    """
    sessions = traces.sessions if isinstance(traces, TraceSet) else tuple(traces)

    samples: list[int] = []
    deltas: list[int] = []
    truncations = 0
    turns_measured = 0

    for session in sessions:
        previous: int | None = None
        for turn in session:
            if turn.stop_reason is StopReason.MAX_TOKENS:
                truncations += 1
            if turn.usage is None:
                continue
            size = turn.usage.context_size
            if size <= 0:
                continue
            turns_measured += 1
            samples.append(size)
            if previous is not None:
                deltas.append(size - previous)
            previous = size

    return ContextProfile(
        samples=tuple(samples),
        deltas=tuple(deltas),
        peak=max(samples) if samples else 0,
        truncations=truncations,
        turns_measured=turns_measured,
    )


def profile_cache(traces: TraceSet | Sequence[Session]) -> CacheProfile:
    """Eficiência de cache agregada."""
    sessions = traces.sessions if isinstance(traces, TraceSet) else tuple(traces)
    usage = TraceSet.of(sessions).total_usage
    return CacheProfile(
        read_tokens=usage.cache_read_tokens,
        write_tokens=usage.cache_write_tokens,
        fresh_input_tokens=usage.input_tokens,
    )


def context_peak_by_session(traces: TraceSet | Sequence[Session]) -> dict[str, int]:
    """
    Pico de contexto por sessão — uma observação por unidade experimental.

    Entrada da camada estatística, pelo mesmo motivo que ``error_rate_by_session``:
    comparar harnesses exige distribuição, não um agregado. Um pico médio não tem
    variância, e sem variância não há evidência, só dois números diferentes.
    """
    sessions = traces.sessions if isinstance(traces, TraceSet) else tuple(traces)
    return {
        session.id: max(
            (t.usage.context_size for t in session if t.usage and t.usage.context_size > 0),
            default=0,
        )
        for session in sessions
    }
