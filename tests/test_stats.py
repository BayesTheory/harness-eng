"""
Testes da camada estatística.

Código estatístico sutilmente errado é o pior tipo de erro num repositório cujo argumento
é rigor de medição: ele produz número plausível, ninguém desconfia, e a conclusão errada
circula. Por isso estes testes não checam "roda sem levantar" — checam contra
implementação de referência, contra casos com resposta conhecida, e contra a propriedade
que define cada estatística.
"""
from __future__ import annotations

import random

import pytest

from harness_eng.stats.compare import (
    EffectSize,
    cliffs_delta,
    compare_paired,
    compare_unpaired,
    paired_bootstrap,
    paired_dominance,
)
from harness_eng.stats.design import describe_baseline, estimate_power, required_pairs


def naive_cliffs_delta(a, b):
    """Definição direta, O(n·m). A referência contra a qual a versão rápida é conferida."""
    greater = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    return (greater - less) / (len(a) * len(b))


class TestCliffsDelta:
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 0.0),
            ([10, 11, 12], [1, 2, 3], 1.0),
            ([1, 2, 3], [10, 11, 12], -1.0),
            ([1], [1], 0.0),
        ],
    )
    def test_known_cases(self, a, b, expected):
        assert cliffs_delta(a, b) == pytest.approx(expected)

    def test_matches_the_naive_implementation_on_random_input(self):
        """
        A versão rápida usa busca binária em vez do laço duplo.

        A otimização existe porque o bootstrap chama isto 10.000 vezes; se ela divergir da
        definição em algum caso de borda — empate, amostra de tamanho 1, valores repetidos
        — o erro entra em todo relatório e não se manifesta como exceção.
        """
        rng = random.Random(7)
        for _ in range(200):
            a = [rng.choice([rng.gauss(0, 1), rng.randint(0, 3)]) for _ in range(rng.randint(1, 40))]
            b = [rng.choice([rng.gauss(0.3, 1), rng.randint(0, 3)]) for _ in range(rng.randint(1, 40))]
            assert cliffs_delta(a, b) == pytest.approx(naive_cliffs_delta(a, b), abs=1e-12)

    def test_ties_are_counted_as_neither(self):
        # Todos empatam: nem a supera b, nem b supera a.
        assert cliffs_delta([5, 5, 5], [5, 5, 5]) == 0.0

    def test_is_antisymmetric(self):
        a, b = [1, 4, 9], [2, 3, 10]
        assert cliffs_delta(a, b) == pytest.approx(-cliffs_delta(b, a))

    def test_empty_sample_is_an_error_not_zero(self):
        # Zero significaria "sem diferença medida"; o certo é "não dá para medir".
        with pytest.raises(ValueError):
            cliffs_delta([], [1, 2, 3])


class TestPairedDominance:
    def test_all_pairs_improved_is_total_dominance(self):
        assert paired_dominance([-1.0, -2.0, -0.5], lower_is_better=True) == 1.0

    def test_direction_is_respected(self):
        assert paired_dominance([-1.0, -2.0], lower_is_better=False) == 0.0

    def test_ties_are_excluded_from_the_denominator(self):
        # Dois pares decididos, ambos a favor de B, mais dois empates: 100%, não 50%.
        assert paired_dominance([-1.0, -1.0, 0.0, 0.0]) == 1.0

    def test_all_ties_fall_back_to_chance(self):
        assert paired_dominance([0.0, 0.0]) == 0.5


class TestBootstrap:
    def test_interval_brackets_the_observed_median(self):
        differences = [-2.0, -1.5, -1.0, -0.5, -3.0]
        interval = paired_bootstrap(differences, resamples=2000)
        assert interval.low <= -1.5 <= interval.high

    def test_confidence_interval_coverage_is_honest(self):
        """
        Um IC de 95% precisa conter o valor verdadeiro em ~95% das repetições.

        É a propriedade que define um intervalo de confiança, e a única forma de verificar
        que a implementação não é otimista. Um bootstrap com reamostragem errada produz
        intervalo estreito e confiante — que é pior que intervalo nenhum.
        """
        contains = 0
        trials = 200
        for i in range(trials):
            rng = random.Random(1000 + i)
            # Diferença de duas lognormais idênticas: mediana verdadeira é zero.
            differences = [rng.lognormvariate(0, 1) - rng.lognormvariate(0, 1) for _ in range(40)]
            interval = paired_bootstrap(differences, resamples=600, seed=i)
            if interval.low <= 0.0 <= interval.high:
                contains += 1
        assert 0.88 <= contains / trials <= 0.99

    def test_is_reproducible_by_default(self):
        """Relatório que muda de número a cada execução não é auditável."""
        differences = [1.0, -2.0, 3.0, -4.0, 5.0]
        first = paired_bootstrap(differences)
        second = paired_bootstrap(differences)
        assert (first.low, first.high) == (second.low, second.high)

    def test_empty_input_is_an_error(self):
        with pytest.raises(ValueError):
            paired_bootstrap([])


class TestPairedComparison:
    @staticmethod
    def _baseline(n=30, seed=3):
        rng = random.Random(seed)
        return rng, {f"t{i}": rng.lognormvariate(1, 0.8) for i in range(n)}

    def test_no_effect_yields_no_winner(self):
        rng, base = self._baseline()
        same = {t: v * rng.uniform(0.97, 1.03) for t, v in base.items()}
        result = compare_paired(base, same, label_a="atual", label_b="novo")
        assert result.winner is None
        assert not result.interval.excludes_zero

    def test_a_real_improvement_is_detected(self):
        rng, base = self._baseline()
        better = {t: v * rng.uniform(0.55, 0.75) for t, v in base.items()}
        result = compare_paired(base, better, label_a="atual", label_b="novo")
        assert result.winner == "novo"
        assert result.dominance == 1.0
        assert result.relative_change < -0.2

    def test_dominance_beats_cliffs_delta_on_paired_data(self):
        """
        O erro que este módulo cometeu antes de ser medido.

        Toda tarefa melhorou 25-45%, e o delta de Cliff saiu "pequeno" — porque ele
        compara todas as observações contra todas e ignora o pareamento. Com variância
        grande entre tarefas, a tarefa cara melhorada ainda custa mais que a tarefa barata
        original, e as distribuições se sobrepõem. A dominância vê o que o delta não vê.
        """
        rng, base = self._baseline()
        better = {t: v * rng.uniform(0.55, 0.75) for t, v in base.items()}
        result = compare_paired(base, better)
        assert result.dominance == 1.0
        assert result.effect_size in (EffectSize.NEGLIGIBLE, EffectSize.SMALL)
        assert result.winner is not None, "a dominância deve sustentar a conclusão"

    def test_higher_is_better_flips_the_verdict(self):
        rng, base = self._baseline()
        higher = {t: v * 1.5 for t, v in base.items()}
        worse = compare_paired(base, higher, lower_is_better=True)
        better = compare_paired(base, higher, lower_is_better=False)
        assert worse.winner != better.winner

    def test_disjoint_tasks_is_an_error_not_an_empty_result(self):
        """
        Devolver "sem diferença" para dados não comparáveis é pior que falhar.

        O chamador reportaria ausência de efeito onde o certo é ausência de comparação.
        """
        with pytest.raises(ValueError, match="nenhuma tarefa em comum"):
            compare_paired({"a": 1.0}, {"b": 2.0})

    def test_only_shared_tasks_are_compared(self):
        result = compare_paired({"a": 1.0, "b": 2.0, "so_a": 9.0}, {"a": 1.0, "b": 2.0})
        assert result.n_pairs == 2

    def test_summary_is_readable_without_statistics_training(self):
        rng, base = self._baseline()
        better = {t: v * 0.6 for t, v in base.items()}
        summary = compare_paired(base, better, metric="custo", label_b="novo").summary()
        assert "custo" in summary and "novo" in summary and "n=" in summary


class TestUnpairedComparison:
    def test_flags_itself_as_unpaired(self):
        result = compare_unpaired([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert not result.is_paired

    def test_uses_cliffs_delta_for_the_verdict(self):
        # Amostras completamente separadas: delta = -1, efeito grande, veredito claro.
        result = compare_unpaired([1.0, 2.0, 3.0], [10.0, 11.0, 12.0], label_b="novo")
        assert result.effect_size is EffectSize.LARGE
        assert result.winner is not None


class TestDesign:
    def test_skew_is_detected(self):
        """A razão média/mediana é o que justifica bootstrap em vez de teste t."""
        rng = random.Random(11)
        skewed = [rng.lognormvariate(1, 1.2) for _ in range(200)]
        symmetric = [rng.gauss(100, 5) for _ in range(200)]
        assert describe_baseline(skewed)["skewed"] is True
        assert describe_baseline(symmetric)["skewed"] is False

    def test_power_grows_with_sample_size(self):
        rng = random.Random(5)
        baseline = [rng.lognormvariate(1, 0.8) for _ in range(60)]
        small = estimate_power(baseline, 0.2, n_pairs=4, trials=80, resamples=200)
        large = estimate_power(baseline, 0.2, n_pairs=40, trials=80, resamples=200)
        assert large.power >= small.power

    def test_bigger_effects_need_fewer_samples(self):
        rng = random.Random(5)
        baseline = [rng.lognormvariate(1, 0.8) for _ in range(60)]
        few = required_pairs(baseline, 0.30, trials=60)
        many = required_pairs(baseline, 0.05, trials=60)
        assert few is not None
        assert many is None or many >= few

    def test_undetectable_effect_returns_none_not_the_cap(self):
        """
        "Preciso de mais de 200 tarefas" é uma resposta útil.

        Devolver ``max_pairs`` calado faria o usuário rodar 200 execuções achando que
        bastaria, quando o certo é concluir que o efeito procurado é pequeno demais para o
        ruído do sistema.
        """
        rng = random.Random(5)
        baseline = [rng.lognormvariate(1, 1.5) for _ in range(60)]
        assert required_pairs(baseline, 0.001, max_pairs=8, trials=40) is None

    def test_empty_baseline_is_an_error(self):
        with pytest.raises(ValueError):
            estimate_power([], 0.2, n_pairs=10)
