"""
Testes das métricas, sobre sessões construídas à mão.

Nenhum transcript real, nenhuma chave de API, nenhuma dependência pesada. É a propriedade
que o formato canônico existe para dar: uma :class:`Session` montada num teste é
indistinguível de uma lida do disco, então a lógica de medição se testa sem o mundo.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from harness_eng.metrics.context import profile_cache, profile_context
from harness_eng.metrics.cost import ModelPricing, estimate_cost
from harness_eng.metrics.loops import detect_loops
from harness_eng.metrics.tools import analyse_tools, error_rate_by_session
from harness_eng.trace.model import (
    Role,
    Session,
    StopReason,
    ToolCall,
    ToolResult,
    TraceSet,
    Turn,
    Usage,
)

BASE = datetime(2026, 9, 1, 12, 0, 0)


def call(index: int, name: str, **arguments) -> ToolCall:
    return ToolCall(id=f"c{index}", name=name, arguments=arguments, turn_index=index)


def result(index: int, *, error: bool = False, content: str = "ok", kinds=("text",)) -> ToolResult:
    return ToolResult(call_id=f"c{index}", is_error=error, content=content, content_kinds=kinds)


def session(
    *pairs: tuple[ToolCall, ToolResult | None],
    session_id: str = "s1",
    usages: list[Usage] | None = None,
) -> Session:
    """Monta uma sessão: cada par vira um turno de assistant seguido de um de user."""
    turns: list[Turn] = []
    for index, (tool_call, tool_result) in enumerate(pairs):
        turns.append(
            Turn(
                index=len(turns),
                role=Role.ASSISTANT,
                timestamp=BASE + timedelta(seconds=index),
                tool_calls=(tool_call,),
                usage=usages[index] if usages and index < len(usages) else None,
                model="claude-opus-5",
                stop_reason=StopReason.TOOL_USE,
            )
        )
        if tool_result is not None:
            turns.append(
                Turn(index=len(turns), role=Role.USER, tool_results=(tool_result,))
            )
    return Session(id=session_id, source="test", turns=tuple(turns))


class TestToolHealth:
    def test_error_rate_pairs_result_to_the_right_tool(self):
        """
        O erro chega num turno de user; o nome da ferramenta está no assistant anterior.

        Sem casar por ``tool_use_id`` a taxa sai atribuída à ferramenta errada — o erro
        que quase toda análise caseira de transcript comete.
        """
        report = analyse_tools([
            session(
                (call(0, "Bash"), result(0, error=True)),
                (call(1, "Read"), result(1)),
                (call(2, "Bash"), result(2)),
            )
        ])
        assert report.tools["Bash"].error_rate == pytest.approx(0.5)
        assert report.tools["Read"].error_rate == 0.0

    def test_a_call_without_a_result_is_neither_success_nor_error(self):
        report = analyse_tools([session((call(0, "Bash"), None))])
        health = report.tools["Bash"]
        assert health.unanswered == 1
        assert health.errors == 0

    def test_non_text_content_is_not_an_empty_result(self):
        """
        O bug que os dados reais pegaram.

        ``ToolSearch`` devolve blocos ``tool_reference`` e ``Read`` de imagem devolve
        blocos ``image``. Um extrator que só lê ``type=='text'`` os conta como resultado
        vazio, e 58 de 60 chamadas viram "falha silenciosa" — um achado alarmante sobre o
        harness que na verdade é defeito da ferramenta de medição.
        """
        report = analyse_tools([
            session((call(0, "ToolSearch"), result(0, content="", kinds=("tool_reference",))))
        ])
        assert report.tools["ToolSearch"].empty_results == 0

    def test_a_genuinely_empty_result_is_counted(self):
        report = analyse_tools([session((call(0, "Bash"), result(0, content="", kinds=())))])
        assert report.tools["Bash"].empty_results == 1

    def test_rate_without_enough_calls_is_flagged_as_noise(self):
        report = analyse_tools([session((call(0, "Rare"), result(0, error=True)))])
        assert report.tools["Rare"].error_rate == 1.0
        assert not report.tools["Rare"].has_signal
        assert report.ranked_by_error_rate() == [], "amostra pequena não deve liderar o ranking"

    def test_outliers_compare_against_the_harness_own_baseline(self):
        turns = [(call(i, "Good"), result(i)) for i in range(100)]
        turns += [(call(100 + i, "Bad"), result(100 + i, error=i < 20)) for i in range(40)]
        report = analyse_tools([session(*turns)])
        assert "Bad" in [t.name for t in report.outliers()]
        assert "Good" not in [t.name for t in report.outliers()]

    def test_per_session_rates_preserve_variance(self):
        """A camada estatística precisa de distribuição, não de um agregado."""
        rates = error_rate_by_session([
            session((call(0, "Bash"), result(0, error=True)), session_id="a"),
            session((call(0, "Bash"), result(0)), session_id="b"),
        ])
        assert rates == {"a": 1.0, "b": 0.0}


class TestLoops:
    def test_identical_repeated_calls_are_detected(self):
        pairs = [(call(i, "Bash", command="pytest"), result(i)) for i in range(6)]
        report = detect_loops([session(*pairs)], min_repeats=4)
        assert len(report.repeats) == 1
        assert report.repeats[0].count == 6
        assert report.wasted_calls == 5, "a primeira chamada era legítima"

    def test_different_arguments_are_not_a_loop(self):
        pairs = [(call(i, "Bash", command=f"echo {i}"), result(i)) for i in range(8)]
        assert detect_loops([session(*pairs)], min_repeats=4).repeats == ()

    def test_argument_order_does_not_hide_a_repeat(self):
        """Dois dicts iguais com ordem de inserção diferente são a mesma chamada."""
        a = ToolCall(id="c0", name="Edit", arguments={"x": 1, "y": 2}, turn_index=0)
        b = ToolCall(id="c1", name="Edit", arguments={"y": 2, "x": 1}, turn_index=1)
        assert a.signature() == b.signature()

    def test_blind_retry_needs_repeated_failure(self):
        pairs = [(call(i, "Bash", command="bad"), result(i, error=True)) for i in range(3)]
        report = detect_loops([session(*pairs)], min_repeats=99)
        assert len(report.blind_retries) == 1
        assert report.blind_retries[0].attempts == 3

    def test_a_corrected_retry_is_not_blind(self):
        pairs = [
            (call(0, "Bash", command="bad"), result(0, error=True)),
            (call(1, "Bash", command="fixed"), result(1)),
        ]
        assert detect_loops([session(*pairs)], min_repeats=99).blind_retries == ()

    def test_spread_out_repetition_is_not_a_loop(self):
        """
        Janela estreita: repetição espalhada pela sessão é trabalho legítimo.

        É o que separa o detector do script ingênuo — no dado real o ingênuo contava 124
        padrões e o detector conta 41, porque só o aglomerado dentro da janela é loop.
        """
        pairs = []
        for i in range(60):
            command = "repetido" if i % 20 == 0 else f"unico {i}"
            pairs.append((call(i, "Bash", command=command), result(i)))
        assert detect_loops([session(*pairs)], min_repeats=3, window=5).repeats == ()


class TestContextAndCost:
    def test_context_deltas_do_not_cross_session_boundaries(self):
        """
        A diferença entre o fim de uma sessão e o começo da seguinte é um salto artificial.

        Incluí-la contamina a distribuição de crescimento com valores negativos enormes e
        faz o p95 descrever um artefato em vez do turno caro real.
        """
        usages = [Usage(cache_read_tokens=n) for n in (1000, 2000, 3000)]
        one = session(*[(call(i, "Bash"), result(i)) for i in range(3)], session_id="a", usages=usages)
        two = session(*[(call(i, "Bash"), result(i)) for i in range(3)], session_id="b", usages=usages)
        profile = profile_context([one, two])
        assert all(delta == 1000 for delta in profile.deltas)
        assert len(profile.deltas) == 4, "2 deltas por sessão, nenhum entre elas"

    def test_turns_without_usage_are_skipped_not_zeroed(self):
        profile = profile_context([session((call(0, "Bash"), result(0)))])
        assert profile.turns_measured == 0
        assert profile.median_growth is None

    def test_truncation_is_counted(self):
        turn = Turn(index=0, role=Role.ASSISTANT, stop_reason=StopReason.MAX_TOKENS)
        profile = profile_context([Session(id="s", source="test", turns=(turn,))])
        assert profile.truncations == 1

    def test_cache_hit_rate(self):
        usages = [Usage(input_tokens=100, cache_read_tokens=900)]
        cache = profile_cache([session((call(0, "Bash"), result(0)), usages=usages)])
        assert cache.hit_rate == pytest.approx(0.9)

    def test_unknown_model_is_a_gap_not_an_estimate(self):
        turn = Turn(index=0, role=Role.ASSISTANT, model="modelo-inventado", usage=Usage(input_tokens=1000))
        report = estimate_cost([Session(id="s", source="test", turns=(turn,))])
        assert not report.is_complete
        assert report.total == 0.0
        assert "modelo-inventado" in report.unpriced_models

    def test_zero_token_model_does_not_break_completeness(self):
        """Um aviso que dispara sempre vira ruído que ninguém lê."""
        turn = Turn(index=0, role=Role.ASSISTANT, model="<synthetic>", usage=Usage())
        assert estimate_cost([Session(id="s", source="test", turns=(turn,))]).is_complete

    def test_dated_model_suffix_matches_by_longest_prefix(self):
        turn = Turn(
            index=0, role=Role.ASSISTANT, model="claude-opus-5-20260101",
            usage=Usage(input_tokens=1_000_000),
        )
        report = estimate_cost([Session(id="s", source="test", turns=(turn,))])
        assert report.is_complete
        assert report.total == pytest.approx(5.0)

    def test_custom_pricing_overrides_the_table(self):
        turn = Turn(index=0, role=Role.ASSISTANT, model="meu-modelo", usage=Usage(output_tokens=1_000_000))
        report = estimate_cost(
            [Session(id="s", source="test", turns=(turn,))],
            pricing={"meu-modelo": ModelPricing(1.0, 7.0)},
        )
        assert report.total == pytest.approx(7.0)


class TestTraceSet:
    def test_filters_by_source(self):
        traces = TraceSet.of([
            Session(id="a", source="claude_code"),
            Session(id="b", source="openai"),
        ])
        assert traces.sources == ("claude_code", "openai")
        assert len(traces.from_source("openai")) == 1

    def test_usage_aggregates_across_sessions(self):
        usages = [Usage(output_tokens=10)]
        traces = TraceSet.of([
            session((call(0, "Bash"), result(0)), session_id="a", usages=usages),
            session((call(0, "Bash"), result(0)), session_id="b", usages=usages),
        ])
        assert traces.total_usage.output_tokens == 20
