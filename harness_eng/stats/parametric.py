"""
Teste t pareado — e a medição que mostrou por que ele pertence a este pacote.

Este arquivo existe por causa de um erro meu, encontrado do jeito que este repositório
prega: medindo em vez de argumentar.

A primeira versão do pacote usava só bootstrap, com a justificativa de que "as
distribuições deste domínio são assimétricas, e o teste t supõe normalidade". A premissa
estava certa — o baseline real tem assimetria **+1,66** — mas a conclusão não, por um
motivo que só apareceu na simulação:

**O pareamento simetriza.** A assimetria das DIFERENÇAS pareadas do mesmo baseline é
**−0,13**. O teste t pareado não supõe que os dados sejam normais; supõe que as
*diferenças* sejam aproximadamente simétricas. Parear é exatamente o que produz isso.

Calibração medida sobre o baseline real (3.000 repetições sob a hipótese nula, nominal
5%, ±0,8%)::

    n=12    teste t  2,7%   bootstrap percentil  6,8%
    n=20    teste t  3,1%   bootstrap percentil  6,4%
    n=40    teste t  5,2%   bootstrap percentil  6,1%

O teste t é conservador em amostra pequena e correto em n=40. O bootstrap da mediana é
liberal em toda faixa testada — e **BCa não corrige** (7,5% em n=12, medido). A mediana é
uma estatística não-suave e a teoria de bootstrap para ela é fraca em amostra pequena.

Consequência prática desconfortável: parte do "poder maior" que o bootstrap exibia era só
ele rejeitar mais vezes, inclusive quando não devia.

**Por isso o veredito de significância passou a usar o teste t**, e o bootstrap continua
no pacote para o que ele faz bem: intervalo em torno da mediana (que responde outra
pergunta) e estatísticas sem teoria paramétrica pronta, como p95 e razões.

Sem ``scipy``: a CDF da t sai da função beta incompleta, ~40 linhas conferidas contra
valores críticos tabelados no teste.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TTestResult:
    """Resultado de um teste t pareado."""

    n: int
    mean_difference: float
    std_error: float
    t_statistic: float
    p_value: float
    ci_low: float
    ci_high: float
    confidence: float = 0.95

    @property
    def is_significant(self) -> bool:
        return self.p_value < (1.0 - self.confidence)

    @property
    def excludes_zero(self) -> bool:
        """Mesma pergunta que o intervalo do bootstrap responde, para o veredito casar."""
        return not (self.ci_low <= 0.0 <= self.ci_high)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_difference": round(self.mean_difference, 6),
            "t": round(self.t_statistic, 4),
            "p_value": round(self.p_value, 6),
            "ci": [round(self.ci_low, 6), round(self.ci_high, 6)],
            "significant": self.is_significant,
        }


def paired_t_test(
    differences: Sequence[float], confidence: float = 0.95
) -> TTestResult | None:
    """
    Teste t pareado bilateral sobre as diferenças.

    Devolve ``None`` com menos de duas diferenças ou com variância zero — casos em que o
    teste não é definido. ``None`` e não um p-valor de 1,0: "não dá para testar" e "testei
    e não achei nada" são conclusões diferentes, e colapsá-las é o tipo de silêncio que
    este pacote existe para evitar.
    """
    n = len(differences)
    if n < 2:
        return None

    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    if sd == 0.0:
        return None

    std_error = sd / math.sqrt(n)
    t_statistic = mean / std_error
    df = n - 1
    p_value = _t_two_sided_p(t_statistic, df)
    critical = _t_critical(df, confidence)

    return TTestResult(
        n=n,
        mean_difference=mean,
        std_error=std_error,
        t_statistic=t_statistic,
        p_value=p_value,
        ci_low=mean - critical * std_error,
        ci_high=mean + critical * std_error,
        confidence=confidence,
    )


def paired_t_test_of(
    a: Mapping[str, float], b: Mapping[str, float], confidence: float = 0.95
) -> TTestResult | None:
    """Teste t sobre as tarefas que as duas configurações têm em comum."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    return paired_t_test([float(b[k]) - float(a[k]) for k in shared], confidence)


def difference_skewness(differences: Sequence[float]) -> float | None:
    """
    Assimetria **robusta** das diferenças, por quartis (Bowley), em [-1, +1].

    ``(Q3 + Q1 - 2·mediana) / (Q3 - Q1)``. Robusta e limitada, ao contrário do momento
    de terceira ordem — e a escolha é consequência de um teste que falhou:

    A primeira versão usava a assimetria clássica. Num dado de diferenças **simétrico por
    construção** (``v × ruído_uniforme_simétrico``) sobre um baseline de cauda muito
    pesada, ela devolveu **−10,76**. Não era assimetria real: o momento de terceira ordem
    eleva os desvios ao cubo, então um único valor extremo domina o estimador inteiro. O
    portão que decide qual teste usar seria acionado por ruído amostral.

    A versão por quartis dá ~0 no mesmo dado, porque nenhum ponto individual pode mover um
    quartil mais que um lugar. Um diagnóstico que só é confiável quando o dado é bem
    comportado não serve para decidir o que fazer quando ele não é.
    """
    n = len(differences)
    if n < 4:
        return None
    ordered = sorted(differences)
    q1 = _quantile(ordered, 0.25)
    q2 = _quantile(ordered, 0.50)
    q3 = _quantile(ordered, 0.75)
    spread = q3 - q1
    if spread == 0:
        # Quartis colapsados: os 50% centrais são um ponto só, e a medida não distingue
        # simétrico de assimétrico — a cauda inteira fica fora do que ela enxerga. Foi um
        # buraco real: `[-400]*3 + [-1]*27` devolvia 0,0 ("simétrico") para um dado
        # dominado por três outliers. `None` significa "não dá para verificar", e quem
        # decide o método trata isso como motivo para usar o teste mais robusto, não
        # como licença para usar o mais frágil.
        return None
    return (q3 + q1 - 2.0 * q2) / spread


def classical_skewness(values: Sequence[float]) -> float | None:
    """
    Momento de terceira ordem. Mantido para comparação e para o relatório.

    NÃO use para decidir método: ver :func:`difference_skewness` para por quê.
    """
    n = len(values)
    if n < 3:
        return None
    mean = statistics.mean(values)
    sd = statistics.pstdev(values)
    if sd == 0:
        return 0.0
    return sum((x - mean) ** 3 for x in values) / (n * sd**3)


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    """Quantil por interpolação linear entre os dois vizinhos."""
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


# ── distribuição t sem scipy ──────────────────────────────────────────────────
def _t_two_sided_p(t_statistic: float, df: int) -> float:
    """P-valor bilateral via ``I_x(df/2, 1/2)``, a relação padrão entre t e beta."""
    x = df / (df + t_statistic * t_statistic)
    return _incomplete_beta(df / 2.0, 0.5, x)


def _t_critical(df: int, confidence: float) -> float:
    """
    Valor crítico bilateral, por bisseção sobre a CDF.

    Bisseção em vez de tabela: a tabela cobriria alguns ``df`` e falharia calada nos
    outros, e o custo aqui são ~60 avaliações de uma função que já existe.
    """
    target = 1.0 - confidence
    low, high = 0.0, 1000.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if _t_two_sided_p(mid, df) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Função beta incompleta regularizada, por fração continuada (método de Lentz)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    front = math.exp(log_beta)
    # A fração continuada converge rápido só de um lado; do outro usa-se a simetria
    # I_x(a,b) = 1 - I_{1-x}(b,a). Sem essa troca a convergência trava para x grande.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_continued_fraction(a: float, b: float, x: float, iterations: int = 300) -> float:
    tiny, epsilon = 1e-30, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    result = d

    for m in range(1, iterations + 1):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        c = 1.0 + numerator / c
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = c if abs(c) > tiny else tiny
        result *= d * c

        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        c = 1.0 + numerator / c
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = c if abs(c) > tiny else tiny
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return result
