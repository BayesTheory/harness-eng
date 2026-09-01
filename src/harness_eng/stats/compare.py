"""
Comparação estatística entre harnesses.

O motivo de este repositório existir. A literatura de harness engineering é anedótica:
"esse prompt parece melhor", "esse loop parece mais estável". Ninguém publica intervalo
de confiança, e quase ninguém pareia.

Três decisões metodológicas, cada uma com uma razão que o dado exige:

**1. Pareamento por tarefa.** A variância entre tarefas é enorme — uma tarefa fácil custa
centavos, uma difícil custa dólares. Comparar a média de 20 execuções do harness A contra
20 do harness B mede principalmente quais tarefas caíram em qual grupo. Pareando (mesma
tarefa nos dois), essa variância cancela e sobra o efeito do harness.

**2. Bootstrap em vez de teste t.** As distribuições aqui são fortemente assimétricas e
com cauda pesada: custo por sessão, tokens de contexto, contagem de retry. O teste t supõe
normalidade e, nessas distribuições, produz intervalo confiante e errado. O bootstrap não
supõe forma nenhuma — reamostra o que se observou.

**3. Cliff's delta em vez de Cohen's d.** O ``d`` é razão de diferença de médias sobre
desvio padrão; num dado assimétrico o desvio padrão não descreve a dispersão e o ``d``
fica inflado. O delta de Cliff é ordinal: conta quantas vezes uma amostra supera a outra.
Não supõe forma, não é afetado por outlier, e responde à pergunta que se quer fazer —
"com que frequência A é melhor que B".

Sem ``scipy``: tudo abaixo é ``statistics`` e ``random`` da biblioteca padrão. Mantém a
instalação leve e o cálculo auditável linha a linha, que num repositório cujo argumento é
rigor de medição vale mais que a conveniência.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

#: Limiares de magnitude do delta de Cliff (Romano et al., 2006). São convenção de campo,
#: não lei da natureza — servem para rotular, e o número em si é que deve ser reportado.
NEGLIGIBLE, SMALL, MEDIUM = 0.147, 0.33, 0.474

DEFAULT_RESAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95


class EffectSize(str, Enum):
    """Magnitude qualitativa de um efeito, para o relatório."""

    NEGLIGIBLE = "desprezível"
    SMALL = "pequeno"
    MEDIUM = "médio"
    LARGE = "grande"

    @classmethod
    def of(cls, delta: float) -> "EffectSize":
        magnitude = abs(delta)
        if magnitude < NEGLIGIBLE:
            return cls.NEGLIGIBLE
        if magnitude < SMALL:
            return cls.SMALL
        if magnitude < MEDIUM:
            return cls.MEDIUM
        return cls.LARGE


@dataclass(frozen=True, slots=True)
class Interval:
    """Intervalo de confiança."""

    low: float
    high: float
    confidence: float = DEFAULT_CONFIDENCE

    @property
    def excludes_zero(self) -> bool:
        """
        Se o intervalo não contém zero.

        É o mais próximo de "significativo" que este módulo diz, e de propósito: um
        intervalo que exclui zero é evidência de direção, não prova de importância. Um
        efeito pode excluir zero e ser irrelevante na prática — por isso o tamanho de
        efeito sai sempre ao lado, nunca sozinho.
        """
        return not (self.low <= 0.0 <= self.high)

    @property
    def width(self) -> float:
        return self.high - self.low

    def as_dict(self) -> dict:
        return {
            "low": round(self.low, 4),
            "high": round(self.high, 4),
            "confidence": self.confidence,
            "excludes_zero": self.excludes_zero,
        }


@dataclass(frozen=True, slots=True)
class Comparison:
    """
    Resultado de comparar duas configurações de harness numa métrica.

    Carrega o efeito, a incerteza e o tamanho — os três juntos, nunca um sozinho. Reportar
    só a diferença de médias é o que faz a literatura do campo ser anedótica; reportar só
    o p-valor é o que faz a de outros campos ser irreprodutível.
    """

    metric: str
    label_a: str
    label_b: str
    n_pairs: int
    median_a: float
    median_b: float
    median_difference: float
    interval: Interval
    cliffs_delta: float
    dominance: float = 0.5
    lower_is_better: bool = True
    is_paired: bool = True

    @property
    def effect_size(self) -> EffectSize:
        return EffectSize.of(self.cliffs_delta)

    @property
    def relative_change(self) -> float | None:
        """
        Mudança relativa da mediana, para leitura humana. ``None`` se a base é zero.

        Sobre a MEDIANA, não a média: é a mesma estatística do resto do relatório, e
        misturar "mediana subiu 3%" com "média subiu 40%" na mesma tabela confunde mais
        do que informa.
        """
        if self.median_a == 0:
            return None
        return (self.median_b - self.median_a) / abs(self.median_a)

    @property
    def winner(self) -> str | None:
        """
        Qual configuração é melhor, ou ``None`` quando o dado não sustenta uma escolha.

        Exige DUAS condições: o intervalo excluir zero e o efeito não ser desprezível.
        Só a primeira é o erro clássico — com amostra grande, qualquer diferença minúscula
        exclui zero, e a conclusão vira "estatisticamente significativo, praticamente
        irrelevante".
        """
        if not self.interval.excludes_zero:
            return None
        # Em desenho pareado o critério de relevância é a dominância, não o delta: o
        # delta subestima sistematicamente quando a variância entre tarefas é grande,
        # e usá-lo aqui descartaria melhorias reais e consistentes como "desprezíveis".
        if self.is_paired:
            if abs(self.dominance - 0.5) < 0.1:
                return None
        elif self.effect_size is EffectSize.NEGLIGIBLE:
            return None
        b_is_smaller = self.median_difference < 0
        b_wins = b_is_smaller if self.lower_is_better else not b_is_smaller
        return self.label_b if b_wins else self.label_a

    def summary(self) -> str:
        """Uma frase que um humano lê e entende sem saber estatística."""
        if self.winner is None:
            if not self.interval.excludes_zero:
                reason = "o intervalo inclui zero"
            elif self.is_paired:
                reason = f"B vence em apenas {self.dominance:.0%} das tarefas"
            else:
                reason = f"o efeito é {self.effect_size.value}"
            return (
                f"{self.metric}: nenhuma diferença sustentável entre {self.label_a} e "
                f"{self.label_b} ({reason}, n={self.n_pairs})"
            )
        change = self.relative_change
        change_text = f", {abs(change):.0%} " + ("menor" if change and change < 0 else "maior") if change else ""
        consistency = (
            f"vence em {self.dominance:.0%} das tarefas, " if self.is_paired else ""
        )
        return (
            f"{self.metric}: {self.winner} vence ({consistency}"
            f"δ={self.cliffs_delta:+.2f} {self.effect_size.value}, IC95% "
            f"[{self.interval.low:.3g}, {self.interval.high:.3g}]{change_text}, "
            f"n={self.n_pairs})"
        )

    def as_dict(self) -> dict:
        return {
            "metric": self.metric,
            "a": self.label_a,
            "b": self.label_b,
            "n_pairs": self.n_pairs,
            "median_a": round(self.median_a, 4),
            "median_b": round(self.median_b, 4),
            "median_difference": round(self.median_difference, 4),
            "relative_change": (
                round(self.relative_change, 4) if self.relative_change is not None else None
            ),
            "interval": self.interval.as_dict(),
            "cliffs_delta": round(self.cliffs_delta, 4),
            "effect_size": self.effect_size.value,
            "paired_dominance": round(self.dominance, 4),
            "is_paired": self.is_paired,
            "winner": self.winner,
            "summary": self.summary(),
        }


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Delta de Cliff: ``(#(a>b) - #(a<b)) / (n·m)``, em [-1, +1].

    +1 significa que toda observação de ``a`` supera toda de ``b``; 0 significa
    sobreposição completa. Ordinal e sem suposição de forma, o que o torna adequado às
    distribuições assimétricas deste domínio — ao contrário do Cohen's d, que divide por
    um desvio padrão que aqui não descreve a dispersão.

    Implementação por ordenação em O(n log n) em vez do laço duplo ingênuo: com 10.000
    reamostras de bootstrap por cima, o laço quadrático transforma uma comparação de
    segundos em uma de minutos.
    """
    if not a or not b:
        raise ValueError("delta de Cliff exige as duas amostras não vazias")

    ordered_b = sorted(b)
    n, m = len(a), len(ordered_b)
    greater = less = 0

    for value in a:
        greater += _count_less(ordered_b, value)
        less += m - _count_less_or_equal(ordered_b, value)

    return (greater - less) / (n * m)


def _count_less(ordered: Sequence[float], value: float) -> int:
    """Quantos elementos são estritamente menores que ``value`` (busca binária)."""
    low, high = 0, len(ordered)
    while low < high:
        mid = (low + high) // 2
        if ordered[mid] < value:
            low = mid + 1
        else:
            high = mid
    return low


def _count_less_or_equal(ordered: Sequence[float], value: float) -> int:
    low, high = 0, len(ordered)
    while low < high:
        mid = (low + high) // 2
        if ordered[mid] <= value:
            low = mid + 1
        else:
            high = mid
    return low


def paired_dominance(differences: Sequence[float], lower_is_better: bool = True) -> float:
    """
    Fração de pares em que B vence A, ignorando empates. Em [0, 1]; 0,5 é o acaso.

    O tamanho de efeito **correto para desenho pareado**, e a razão de ele existir aqui ao
    lado do delta de Cliff é um erro que este módulo cometeu antes de ser medido:

    Cliff's delta compara todas as observações de A contra todas as de B. Isso é certo
    para amostras independentes e **errado para dados pareados**, porque descarta a
    correspondência tarefa a tarefa. Num teste em que TODA tarefa ficou 25-45% mais
    barata, o delta saiu +0,32 ("pequeno") — não por erro de cálculo, mas porque com
    variância grande entre tarefas as duas distribuições ainda se sobrepõem muito: a
    tarefa cara melhorada continua custando mais que a tarefa barata original.

    A dominância pareada responde a pergunta que o desenho pareado faz: *em que fração
    das tarefas B foi melhor?* No mesmo teste, 1,00.

    Os dois saem no relatório. O delta descreve a sobreposição das distribuições — útil
    para saber se a melhora é perceptível olhando execuções soltas. A dominância descreve
    a consistência — útil para saber se a melhora vale para toda tarefa ou só na média.
    """
    wins = sum(1 for d in differences if (d < 0) == lower_is_better and d != 0)
    decided = sum(1 for d in differences if d != 0)
    return wins / decided if decided else 0.5


def paired_bootstrap(
    differences: Sequence[float],
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int | None = 42,
) -> Interval:
    """
    Intervalo de confiança percentil por bootstrap sobre a mediana das diferenças pareadas.

    Reamostra as DIFERENÇAS, não as duas amostras separadamente — é o que preserva o
    pareamento. Reamostrar independentemente destrói a correspondência tarefa a tarefa e
    devolve o intervalo largo de uma comparação não pareada, jogando fora justamente a
    precisão que o pareamento comprou.

    ``seed`` fixo por padrão: um relatório que muda de número a cada execução não é
    auditável. Passe ``seed=None`` para variar deliberadamente.
    """
    if not differences:
        raise ValueError("bootstrap exige ao menos uma diferença pareada")

    rng = random.Random(seed)
    n = len(differences)
    medians: list[float] = []
    for _ in range(resamples):
        sample = [differences[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(sample))

    medians.sort()
    alpha = (1.0 - confidence) / 2.0
    low = medians[int(alpha * resamples)]
    high = medians[min(resamples - 1, int((1.0 - alpha) * resamples))]
    return Interval(low=low, high=high, confidence=confidence)


def compare_paired(
    a: Mapping[str, float],
    b: Mapping[str, float],
    metric: str = "métrica",
    label_a: str = "A",
    label_b: str = "B",
    lower_is_better: bool = True,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int | None = 42,
) -> Comparison:
    """
    Compara duas configurações medidas nas MESMAS tarefas.

    ``a`` e ``b`` são mapas ``tarefa → valor``. Só as chaves presentes nos dois entram: uma
    tarefa que só uma configuração executou não é comparável, e incluí-la reintroduziria
    exatamente a variância entre tarefas que o pareamento existe para eliminar.

    Levanta quando não há par nenhum — devolver um resultado vazio deixaria o chamador
    reportar "sem diferença" quando o certo é "não foi comparado".
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError(
            f"nenhuma tarefa em comum entre {label_a} ({len(a)}) e {label_b} ({len(b)}): "
            f"comparação pareada exige as mesmas tarefas nos dois lados"
        )

    values_a = [float(a[k]) for k in shared]
    values_b = [float(b[k]) for k in shared]
    differences = [vb - va for va, vb in zip(values_a, values_b)]

    return Comparison(
        metric=metric,
        label_a=label_a,
        label_b=label_b,
        n_pairs=len(shared),
        median_a=statistics.median(values_a),
        median_b=statistics.median(values_b),
        median_difference=statistics.median(differences),
        interval=paired_bootstrap(differences, resamples, confidence, seed),
        cliffs_delta=cliffs_delta(values_a, values_b),
        dominance=paired_dominance(differences, lower_is_better),
        lower_is_better=lower_is_better,
        is_paired=True,
    )


def compare_unpaired(
    a: Sequence[float],
    b: Sequence[float],
    metric: str = "métrica",
    label_a: str = "A",
    label_b: str = "B",
    lower_is_better: bool = True,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int | None = 42,
) -> Comparison:
    """
    Comparação não pareada, para quando as tarefas não são as mesmas.

    Existe porque às vezes é o único dado disponível — dois históricos de uso, por exemplo.
    Mas é estritamente pior: o intervalo sai mais largo pela variância entre tarefas, e
    diferença de composição de carga se confunde com diferença de harness. Use
    :func:`compare_paired` sempre que puder rodar as mesmas tarefas nos dois lados.
    """
    if not a or not b:
        raise ValueError("comparação exige as duas amostras não vazias")

    rng = random.Random(seed)
    medians: list[float] = []
    for _ in range(resamples):
        sample_a = [a[rng.randrange(len(a))] for _ in range(len(a))]
        sample_b = [b[rng.randrange(len(b))] for _ in range(len(b))]
        medians.append(statistics.median(sample_b) - statistics.median(sample_a))

    medians.sort()
    alpha = (1.0 - confidence) / 2.0
    interval = Interval(
        low=medians[int(alpha * resamples)],
        high=medians[min(resamples - 1, int((1.0 - alpha) * resamples))],
        confidence=confidence,
    )

    return Comparison(
        metric=metric,
        label_a=label_a,
        label_b=label_b,
        n_pairs=min(len(a), len(b)),
        median_a=statistics.median(a),
        median_b=statistics.median(b),
        median_difference=statistics.median(b) - statistics.median(a),
        interval=interval,
        cliffs_delta=cliffs_delta(list(a), list(b)),
        lower_is_better=lower_is_better,
        is_paired=False,
    )
