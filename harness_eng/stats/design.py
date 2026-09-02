"""
Desenho experimental: quantas execuções são necessárias, decidido antes de rodar.

A pergunta que quase ninguém faz em harness engineering, e que decide se a comparação vai
dizer alguma coisa. Rodar 5 tarefas e não achar diferença não é evidência de que não há
diferença — é evidência de que 5 tarefas não bastavam para achar.

Poder é calculado por **simulação**, não por fórmula fechada. Fórmula de poder pressupõe
normalidade, e as distribuições deste domínio (custo por sessão, tokens de contexto,
contagem de retry) são assimétricas com cauda pesada. Simular a partir do dado observado
responde à pergunta certa: *com a variabilidade que eu de fato tenho, quantas tarefas
preciso?*

Sem ``scipy``: ``statistics`` e ``random`` da biblioteca padrão.
"""
from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from .compare import DEFAULT_CONFIDENCE, paired_bootstrap


@dataclass(frozen=True, slots=True)
class PowerEstimate:
    """Quanto poder um desenho tem para detectar um efeito de tamanho dado."""

    n_pairs: int
    effect: float
    power: float
    trials: int

    @property
    def is_adequate(self) -> bool:
        """0,8 é a convenção de campo: 80% de chance de detectar o efeito se ele existe."""
        return self.power >= 0.8

    def as_dict(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "relative_effect": round(self.effect, 4),
            "power": round(self.power, 3),
            "adequate": self.is_adequate,
            "trials": self.trials,
        }


def estimate_power(
    baseline: Sequence[float],
    relative_effect: float,
    n_pairs: int,
    trials: int = 400,
    resamples: int = 400,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int | None = 42,
) -> PowerEstimate:
    """
    Poder de detectar uma melhora de ``relative_effect`` com ``n_pairs`` tarefas.

    ``baseline`` é a distribuição observada da métrica — os valores reais do harness atual.
    Usar o dado real em vez de uma normal suposta é o ponto: a variabilidade que determina
    o poder é a que você tem, não a que a fórmula assume.

    ``relative_effect=0.2`` significa "B é 20% melhor". A simulação reamostra tarefas do
    baseline, aplica a melhora com ruído multiplicativo, e conta em que fração das
    repetições o intervalo de confiança exclui zero.
    """
    if not baseline:
        raise ValueError("estimativa de poder exige uma distribuição de baseline")
    if n_pairs < 2:
        raise ValueError("comparação pareada exige ao menos 2 tarefas")

    rng = random.Random(seed)
    detected = 0

    for _ in range(trials):
        # Reamostra tarefas do baseline real e aplica o efeito com ruído. Sem o ruído a
        # simulação superestima o poder, porque na prática a melhora nunca é uniforme
        # entre tarefas — e um cálculo de poder otimista é pior que nenhum.
        sample_a = [baseline[rng.randrange(len(baseline))] for _ in range(n_pairs)]
        sample_b = [v * (1.0 - relative_effect) * rng.uniform(0.85, 1.15) for v in sample_a]
        # ``strict``: ``sample_b`` é derivado de ``sample_a`` elemento a elemento, então o
        # tamanho é o mesmo por construção. Perder um par aqui enviesaria o poder estimado.
        differences = [b - a for a, b in zip(sample_a, sample_b, strict=True)]
        if paired_bootstrap(differences, resamples, confidence, seed=None).excludes_zero:
            detected += 1

    return PowerEstimate(
        n_pairs=n_pairs,
        effect=relative_effect,
        power=detected / trials,
        trials=trials,
    )


def required_pairs(
    baseline: Sequence[float],
    relative_effect: float,
    target_power: float = 0.8,
    max_pairs: int = 200,
    trials: int = 200,
    seed: int | None = 42,
) -> int | None:
    """
    Menor número de tarefas que atinge ``target_power``, ou ``None`` se ``max_pairs`` não
    basta.

    ``None`` em vez de devolver ``max_pairs`` calado: "preciso de mais de 200 tarefas para
    detectar 5%" é uma resposta útil — quase sempre significa que o efeito procurado é
    pequeno demais para o ruído do sistema, e que a pergunta deve mudar, não o n.

    Busca por passos crescentes: poder é monotônico em n, e testar de um em um até 200
    custa 200 simulações completas sem ganhar precisão que importe na decisão.
    """
    for n in (4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 200):
        if n > max_pairs:
            break
        estimate = estimate_power(
            baseline, relative_effect, n, trials=trials, resamples=300, seed=seed
        )
        if estimate.power >= target_power:
            return n
    return None


def describe_baseline(values: Sequence[float]) -> dict:
    """
    Resumo da distribuição, com a assimetria explícita.

    A razão média/mediana é o número que justifica todas as escolhas metodológicas deste
    pacote: quando ela passa de ~1,2, a distribuição é assimétrica o bastante para que
    teste t e Cohen's d deem resposta errada com aparência de precisão.
    """
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    mean = statistics.mean(ordered)
    median = statistics.median(ordered)
    return {
        "n": len(ordered),
        "mean": round(mean, 4),
        "median": round(median, 4),
        "p90": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))], 4),
        "max": round(ordered[-1], 4),
        "mean_over_median": round(mean / median, 3) if median else None,
        "skewed": bool(median and mean / median > 1.2),
    }
