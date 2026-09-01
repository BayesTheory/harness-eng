"""
CLI do harness-eng.

    harness-eng analyze [dir]        métricas sobre transcripts
    harness-eng compare A.json B.json  comparação pareada com IC e tamanho de efeito
    harness-eng power [dir]          quantas tarefas para detectar uma melhora
    harness-eng sources              origens de trace disponíveis

``--redact`` substitui caminho e nome de projeto por hash estável. Existe porque o
relatório carrega conteúdo de comando e caminho de arquivo do trabalho real — e um
relatório que vaza nome de cliente por descuido é pior que relatório nenhum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .metrics.context import profile_cache, profile_context
from .metrics.cost import cost_per_session, estimate_cost
from .metrics.loops import detect_loops
from .metrics.tools import analyse_tools
from .stats.compare import compare_paired
from .stats.design import describe_baseline, required_pairs
from .trace.model import TraceSet
from .trace.sources.claude_code import ClaudeCodeSource, default_root

__version__ = "0.1.0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness-eng",
        description="Ferramentas para medir harnesses de agente.",
    )
    parser.add_argument("--version", action="version", version=f"harness-eng {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="métricas sobre um diretório de transcripts")
    analyze.add_argument("root", nargs="?", type=Path, default=None,
                         help="diretório de traces (padrão: ~/.claude/projects)")
    analyze.add_argument("--json", action="store_true", help="saída em JSON")
    analyze.add_argument("--redact", action="store_true",
                         help="troca caminho e comando por hash estável")
    analyze.add_argument("--min-calls", type=int, default=30,
                         help="mínimo de chamadas para uma taxa de erro contar como sinal")

    compare = sub.add_parser("compare", help="comparação pareada entre duas medições")
    compare.add_argument("a", type=Path, help="JSON {tarefa: valor} da configuração A")
    compare.add_argument("b", type=Path, help="JSON {tarefa: valor} da configuração B")
    compare.add_argument("--metric", default="métrica")
    compare.add_argument("--label-a", default="A")
    compare.add_argument("--label-b", default="B")
    compare.add_argument("--higher-is-better", action="store_true")
    compare.add_argument("--json", action="store_true")

    power = sub.add_parser("power", help="quantas tarefas para detectar uma melhora")
    power.add_argument("root", nargs="?", type=Path, default=None)
    power.add_argument("--effects", default="0.05,0.1,0.2,0.3",
                       help="melhoras relativas a testar, separadas por vírgula")

    sub.add_parser("sources", help="origens de trace disponíveis")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "compare":
        return _compare(args)
    if args.command == "power":
        return _power(args)
    if args.command == "sources":
        print("claude_code   ~/.claude/projects/**/*.jsonl")
        print("openai        (planejado)")
        print("native        (planejado — o harness deste repositório)")
        return 0
    return 1


def _load(root: Path | None) -> tuple[TraceSet, ClaudeCodeSource]:
    source = ClaudeCodeSource()
    resolved = root or default_root()
    if not resolved.exists():
        print(f"erro: diretório de traces não encontrado: {resolved}", file=sys.stderr)
        raise SystemExit(2)
    traces = TraceSet.of(list(source.sessions(resolved)))
    if not traces.sessions:
        print(f"erro: nenhum trace legível em {resolved}", file=sys.stderr)
        raise SystemExit(2)
    return traces, source


def _redact(text: str | None) -> str | None:
    """
    Hash estável de 8 caracteres. Estável para o mesmo texto continuar comparável entre
    execuções — redação que randomiza destrói a própria análise que justifica o relatório.
    """
    if not text:
        return text
    return "«" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:8] + "»"


def _analyze(args) -> int:
    traces, source = _load(args.root)

    tools = analyse_tools(traces)
    loops = detect_loops(traces)
    context = profile_context(traces)
    cache = profile_cache(traces)
    cost = estimate_cost(traces)

    if args.json:
        payload = {
            "sessions": len(traces),
            "tools": tools.as_dict(),
            "loops": loops.as_dict(redact=args.redact),
            "context": context.as_dict(),
            "cache": cache.as_dict(),
            "cost": cost.as_dict(),
            "adapter_skipped": dict(source.skipped),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    turns = sum(len(s) for s in traces)
    print(f"\n{len(traces)} sessões · {turns:,} turnos · {tools.total_calls:,} chamadas de ferramenta\n")

    print("FERRAMENTAS")
    print(f"  taxa de erro geral: {tools.overall_error_rate:.2%}")
    ranked = [t for t in tools.ranked_by_error_rate(only_with_signal=False) if t.calls >= 10]
    for health in ranked[:12]:
        flag = "  ← outlier" if health in tools.outliers() else ""
        signal = "" if health.has_signal else "  (amostra pequena)"
        print(f"    {health.name:18} {health.calls:>6} chamadas  {health.error_rate:>6.1%}{flag}{signal}")

    print("\nLOOPS")
    print(f"  {len(loops.repeats)} padrões de repetição, "
          f"{loops.wasted_calls} chamadas desperdiçadas ({loops.waste_rate:.1%})")
    print(f"  {len(loops.blind_retries)} retries cegos · {len(loops.oscillations)} oscilações")
    for repeat in loops.worst(3):
        argument = _redact(repeat.sample_argument) if args.redact else (repeat.sample_argument or "")
        print(f"    {repeat.count:>3}x  {repeat.tool:12} {str(argument)[:56]}")

    print("\nCONTEXTO")
    print(f"  pico            {context.peak:,} tokens")
    print(f"  crescimento/turno  mediana {context.median_growth:,.0f} · "
          f"média {context.mean_growth:,.0f} · p95 {context.p95_growth:,}")
    if context.growth_concentration is not None:
        print(f"  concentração    {context.growth_concentration:.0%} do crescimento vem dos 5% de turnos mais caros")
    print(f"  truncamentos    {context.truncations}")

    print("\nCACHE")
    print(f"  acerto          {cache.hit_rate:.1%}")
    print(f"  rewrite ratio   {cache.rewrite_ratio:.3f} (escrito por token lido)")

    print("\nCUSTO")
    print(f"  total estimado  US$ {cost.total:,.2f}" + ("" if cost.is_complete else "  (INCOMPLETO)"))
    for model, value in cost.ranked()[:5]:
        print(f"    {model:22} US$ {value:>10,.2f}")
    if not cost.is_complete:
        print(f"  sem preço: {', '.join(cost.unpriced_models)} ({cost.unpriced_tokens:,} tokens)")

    skipped = sum(source.skipped.values())
    if skipped:
        print(f"\n  ({skipped:,} registros ignorados pelo adapter — estado de cliente, não turnos)")
    print()
    return 0


def _compare(args) -> int:
    try:
        a = json.loads(args.a.read_text(encoding="utf-8"))
        b = json.loads(args.b.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"erro ao ler as medições: {exc}", file=sys.stderr)
        return 2

    try:
        result = compare_paired(
            a, b,
            metric=args.metric,
            label_a=args.label_a,
            label_b=args.label_b,
            lower_is_better=not args.higher_is_better,
        )
    except ValueError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
        return 0

    print(f"\n{result.summary()}\n")
    print(f"  mediana {args.label_a:>10}  {result.median_a:,.4f}")
    print(f"  mediana {args.label_b:>10}  {result.median_b:,.4f}")
    print(f"  IC95% da diferença   [{result.interval.low:,.4f}, {result.interval.high:,.4f}]")
    print(f"  delta de Cliff       {result.cliffs_delta:+.3f} ({result.effect_size.value})")
    print(f"  dominância pareada   {result.dominance:.0%} das tarefas")
    print(f"  pares comparados     {result.n_pairs}\n")
    return 0


def _power(args) -> int:
    traces, _ = _load(args.root)
    costs = [v for v in cost_per_session(traces).values() if v > 0]
    if len(costs) < 4:
        print("erro: poucas sessões com custo para estimar poder", file=sys.stderr)
        return 1

    baseline = describe_baseline(costs)
    print(f"\nbaseline: custo por sessão (n={baseline['n']})")
    print(f"  mediana {baseline['median']:,.2f} · média {baseline['mean']:,.2f} · "
          f"p90 {baseline['p90']:,.2f}")
    print(f"  média/mediana {baseline['mean_over_median']} → "
          f"{'assimétrico: bootstrap e delta de Cliff são apropriados' if baseline['skewed'] else 'aproximadamente simétrico'}")

    print("\ntarefas pareadas necessárias (poder 0,8):")
    for raw in args.effects.split(","):
        try:
            effect = float(raw)
        except ValueError:
            continue
        n = required_pairs(costs, effect, trials=120)
        print(f"  detectar melhora de {effect:>5.0%}:  {n if n else '> 200'}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
