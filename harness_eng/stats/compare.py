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

**2. Teste t para o veredito, bootstrap para a mediana.** Esta era a decisão errada na
primeira versão, e a medição corrigiu.

O argumento original — "as distribuições são assimétricas, logo o teste t mente" — tem
premissa certa (assimetria do baseline real: +1,66) e conclusão errada. O teste t pareado
não supõe que os DADOS sejam normais; supõe que as DIFERENÇAS sejam aproximadamente
simétricas, e o pareamento é exatamente o que produz isso: a assimetria das diferenças do
mesmo baseline é −0,13.

Calibração medida (3.000 repetições sob a nula, nominal 5%): o teste t dá 2,7% / 3,1% /
5,2% em n=12 / 20 / 40; o bootstrap percentil da mediana dá 6,8% / 6,4% / 6,1%. O
bootstrap é liberal em toda a faixa — e BCa não corrige (7,5% em n=12, medido). Parte do
"poder maior" dele era só rejeitar mais, inclusive quando não devia.

Então o veredito usa o teste t quando as diferenças são simétricas, e cai para o bootstrap
quando não são. O intervalo do bootstrap continua sendo reportado: ele é sobre a MEDIANA,
que responde outra pergunta ("a tarefa típica melhorou?") e não é dominado por uma sessão
cara. Ver ``parametric.py`` para os números completos.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from .parametric import TTestResult, difference_skewness, paired_t_test

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
    def of(cls, delta: float) -> EffectSize:
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
    t_test: TTestResult | None = None
    skewness: float | None = None

    @property
    def effect_size(self) -> EffectSize:
        return EffectSize.of(self.cliffs_delta)

    @property
    def differences_are_symmetric(self) -> bool:
        """
        Se o pareamento simetrizou as diferenças o bastante para o teste t valer.

        Assimetria ROBUSTA (por quartis) abaixo de 0,3. Robusta porque o momento de
        terceira ordem explode com um único outlier — num dado simétrico por construção
        ele chegou a −10,76, e o portão seria acionado por ruído amostral.

        ``None`` (quartis colapsados, amostra minúscula) conta como NÃO verificado, e
        portanto não autoriza o teste t. Não conseguir checar a suposição é motivo para o
        método mais robusto, nunca para o mais frágil.
        """
        return self.skewness is not None and abs(self.skewness) < 0.3

    @property
    def uses_parametric_verdict(self) -> bool:
        """O veredito veio do teste t (melhor calibrado) ou do bootstrap (recuo)."""
        return self.t_test is not None and self.differences_are_symmetric

    @property
    def direction_is_supported(self) -> bool:
        """
        Se há evidência de direção, pelo método mais bem calibrado disponível.

        Teste t quando as diferenças são simétricas — medido: 5,2% de falso positivo em
        n=40 contra 6,1% do bootstrap. Bootstrap quando não são, porque aí a suposição do
        t deixou de valer e um teste mal especificado é pior que um liberal.
        """
        if self.uses_parametric_verdict:
            return self.t_test.excludes_zero
        return self.interval.excludes_zero

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
        if not self.direction_is_supported:
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
            if not self.direction_is_supported:
                reason = (
                    f"p={self.t_test.p_value:.3f}" if self.uses_parametric_verdict
                    else "o intervalo do bootstrap inclui zero"
                )
            elif self.is_paired:
                reason = f"B vence em apenas {self.dominance:.0%} das tarefas"
            else:
                reason = f"o efeito é {self.effect_size.value}"
            return (
                f"{self.metric}: nenhuma diferença sustentável entre {self.label_a} e "
                f"{self.label_b} ({reason}, n={self.n_pairs})"
            )
        change = self.relative_change
        direcao = "menor" if change and change < 0 else "maior"
        change_text = f", {abs(change):.0%} {direcao}" if change else ""
        consistency = (
            f"vence em {self.dominance:.0%} das tarefas, " if self.is_paired else ""
        )
        evidence = (
            f"p={self.t_test.p_value:.4f}" if self.uses_parametric_verdict
            else f"IC95% [{self.interval.low:.3g}, {self.interval.high:.3g}]"
        )
        return (
            f"{self.metric}: {self.winner} vence ({consistency}"
            f"δ={self.cliffs_delta:+.2f} {self.effect_size.value}, {evidence}"
            f"{change_text}, n={self.n_pairs})"
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
            "difference_skewness": (
                round(self.skewness, 3) if self.skewness is not None else None
            ),
            "verdict_method": "t-test" if self.uses_parametric_verdict else "bootstrap",
            "t_test": self.t_test.as_dict() if self.t_test else None,
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
    # ``strict`` documenta a invariante: as duas listas vêm do mesmo ``shared``, então têm
    # o mesmo tamanho por construção. Se um dia deixarem de ter, o pareamento silenciosamente
    # descartaria o excesso — e uma comparação pareada que perde pares mede outra coisa.
    differences = [vb - va for va, vb in zip(values_a, values_b, strict=True)]

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
        t_test=paired_t_test(differences, confidence),
        skewness=difference_skewness(differences),
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
