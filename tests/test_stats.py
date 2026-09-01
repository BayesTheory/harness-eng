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
from harness_eng.stats.parametric import classical_skewness, difference_skewness, paired_t_test


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


class TestParametric:
    """
    O teste t, e a razão medida de ele existir aqui.

    A primeira versão deste pacote não tinha teste t, com a justificativa de que as
    distribuições do domínio são assimétricas. A premissa estava certa e a conclusão
    errada: o t pareado supõe simetria das DIFERENÇAS, e o pareamento produz isso.
    """

    def test_critical_values_match_published_tables(self):
        """
        Implementação própria da distribuição t exige verificação independente.

        Valores críticos bilaterais de 95%, tabelados. Uma CDF sutilmente errada produz
        p-valores plausíveis e conclusões erradas — o pior modo de falha possível num
        pacote de estatística.
        """
        from harness_eng.stats.parametric import _t_critical

        published = {1: 12.706, 2: 4.303, 5: 2.571, 10: 2.228, 20: 2.086, 30: 2.042, 120: 1.980}
        for df, expected in published.items():
            assert _t_critical(df, 0.95) == pytest.approx(expected, abs=0.002)

    def test_known_p_values(self):
        from harness_eng.stats.parametric import _t_two_sided_p

        assert _t_two_sided_p(2.228, 10) == pytest.approx(0.05, abs=0.001)
        assert _t_two_sided_p(0.0, 10) == pytest.approx(1.0)
        assert _t_two_sided_p(4.303, 2) == pytest.approx(0.05, abs=0.001)

    def test_undefined_cases_return_none_not_a_p_value(self):
        """"Não dá para testar" e "testei e não achei nada" são conclusões diferentes."""
        assert paired_t_test([1.0]) is None
        assert paired_t_test([2.0, 2.0, 2.0]) is None

    def test_pairing_symmetrises_a_skewed_baseline(self):
        """
        A medição que corrigiu o desenho do pacote.

        O baseline é fortemente assimétrico; as diferenças pareadas, não. É por isso que o
        teste t é aplicável a um dado que "parece" violar sua suposição — e por que a
        justificativa original do bootstrap estava errada.
        """
        rng = random.Random(1)
        baseline = [rng.lognormvariate(1, 1.0) for _ in range(400)]
        differences = [v * rng.uniform(0.85, 1.15) - v for v in baseline]

        assert classical_skewness(baseline) > 1.0, "o baseline precisa ser assimétrico"
        assert abs(difference_skewness(baseline)) > 0.15, "e assimétrico pela medida robusta"
        assert abs(difference_skewness(differences)) < 0.2, "o pareamento deve simetrizar"

    def test_robust_skewness_survives_a_single_outlier(self):
        """
        Por que a medida robusta, e não o momento de terceira ordem.

        Estas diferenças são simétricas por construção (valor × ruído uniforme simétrico)
        sobre um baseline de cauda pesada. A assimetria clássica devolve um número enorme
        porque eleva os desvios ao cubo e um outlier domina tudo; a robusta devolve ~0,
        que é a verdade. Usar a clássica como portão faria o método ser escolhido por
        ruído amostral.
        """
        rng = random.Random(1)
        baseline = [rng.lognormvariate(1, 1.0) for _ in range(400)]
        differences = [v * rng.uniform(0.85, 1.15) - v for v in baseline]

        assert abs(classical_skewness(differences)) > 3.0
        assert abs(difference_skewness(differences)) < 0.2

    def test_collapsed_quartiles_are_undefined_not_symmetric(self):
        """
        Buraco real: 27 de 30 valores iguais colapsam os quartis.

        A medida devolvia 0,0 — "simétrico" — para um dado dominado por três outliers, e
        o portão autorizaria o teste t. ``None`` significa "não dá para verificar", e não
        verificar não é licença para usar o método mais frágil.
        """
        assert difference_skewness([-400.0] * 3 + [-1.0] * 27) is None

    def test_t_test_is_better_calibrated_than_the_bootstrap_here(self):
        """
        Taxa de falso positivo sob a hipótese nula, no regime deste domínio.

        Medido: o bootstrap percentil da mediana é liberal (rejeita mais que os 5%
        nominais) e o teste t não. É a evidência que move o veredito para o t — e que
        contradiz o que a primeira versão deste pacote afirmava.
        """
        rng = random.Random(21)
        baseline = [rng.lognormvariate(1, 1.0) for _ in range(60)]
        t_rejections = bootstrap_rejections = 0
        trials = 400

        for i in range(trials):
            local = random.Random(3000 + i)
            sample = [baseline[local.randrange(len(baseline))] for _ in range(20)]
            differences = [v * local.uniform(0.85, 1.15) - v for v in sample]
            result = paired_t_test(differences)
            if result is not None and result.excludes_zero:
                t_rejections += 1
            if paired_bootstrap(differences, resamples=250, seed=None).excludes_zero:
                bootstrap_rejections += 1

        t_rate = t_rejections / trials
        bootstrap_rate = bootstrap_rejections / trials
        assert t_rate <= 0.075, f"teste t liberal demais: {t_rate:.1%}"
        assert bootstrap_rate >= t_rate, (
            "o achado que motivou este módulo é que o bootstrap rejeita mais; "
            f"aqui deu t={t_rate:.1%} e bootstrap={bootstrap_rate:.1%}"
        )


class TestVerdictMethodSelection:
    def test_symmetric_differences_use_the_t_test(self):
        rng = random.Random(4)
        base = {f"t{i}": rng.uniform(90, 110) for i in range(25)}
        better = {t: v - rng.uniform(4, 6) for t, v in base.items()}
        result = compare_paired(base, better)
        assert result.differences_are_symmetric
        assert result.uses_parametric_verdict
        assert result.as_dict()["verdict_method"] == "t-test"

    def test_skewed_differences_fall_back_to_the_bootstrap(self):
        """
        Quando o pareamento NÃO simetriza, a suposição do t não vale e o bootstrap volta.

        Um teste mal especificado é pior que um liberal: o liberal erra numa direção
        conhecida e reportada; o mal especificado erra numa direção que ninguém sabe.
        """
        rng = random.Random(4)
        base = {f"t{i}": 100.0 for i in range(30)}
        # Poucas tarefas melhoram muitíssimo: diferenças fortemente assimétricas.
        skewed = {t: (v - 400.0 if i < 3 else v - 1.0) for i, (t, v) in enumerate(base.items())}
        result = compare_paired(base, skewed)
        assert not result.differences_are_symmetric
        assert not result.uses_parametric_verdict

    def test_the_summary_names_the_evidence_it_used(self):
        rng = random.Random(4)
        base = {f"t{i}": rng.uniform(90, 110) for i in range(25)}
        better = {t: v - rng.uniform(4, 6) for t, v in base.items()}
        assert "p=" in compare_paired(base, better, label_b="novo").summary()
