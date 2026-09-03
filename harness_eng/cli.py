"""
CLI do harness-eng.

    harness-eng analyze [dir]        métricas sobre transcripts
    harness-eng compare A.json B.json  comparação pareada com IC e tamanho de efeito
    harness-eng power [dir]          quantas tarefas para detectar uma melhora
    harness-eng run "tarefa"         roda o harness mínimo e grava o trace nativo
    harness-eng sources              origens de trace disponíveis

``--redact`` substitui caminho e nome de projeto por hash estável. Existe porque o
relatório carrega conteúdo de comando e caminho de arquivo do trabalho real — e um
relatório que vaza nome de cliente por descuido é pior que relatório nenhum.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

# Uma versao so, definida no pacote. Duas copias divergem — e ja tinham divergido.
from . import __version__
from .core.clients import DEFAULT_MODEL, AnthropicClient, ScriptedClient
from .core.loop import AgentLoop
from .core.policy import LEVELS
from .core.policy import level as level_of
from .core.ports import ModelError, ModelResponse
from .core.toolkit import policy_registry
from .core.tools import workspace_registry
from .metrics.context import profile_cache, profile_context
from .metrics.cost import cost_per_session, estimate_cost
from .metrics.loops import detect_loops
from .metrics.policy import analyse_policy
from .metrics.tools import analyse_tools
from .stats.compare import compare_paired
from .stats.design import describe_baseline, required_pairs
from .trace.model import StopReason, ToolCall, TraceSet
from .trace.sources.claude_code import ClaudeCodeSource, default_root
from .trace.sources.native import NativeSink, NativeSource


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
                         default=_env("HARNESS_REDACT", "false").lower() in {"1", "true", "yes"},
                         help="troca caminho e comando por hash estável (HARNESS_REDACT)")
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

    run = sub.add_parser("run", help="roda o harness mínimo e grava o trace nativo")
    run.add_argument("prompt", help="a tarefa")
    run.add_argument("--workspace", type=Path, default=Path("."),
                     help="raiz que as ferramentas de leitura podem enxergar")
    run.add_argument("--level", type=int, default=None, choices=sorted(LEVELS),
                     help="nível de harness: " + " · ".join(
                         f"{n}={p.name}" for n, p in sorted(LEVELS.items())))
    run.add_argument("--allow", action="append", default=[], metavar="DOMINIO",
                     help="libera um domínio (repetível). Só vale a partir do nível 2")
    run.add_argument("--block", action="append", default=[], metavar="DOMINIO",
                     help="bloqueia um domínio (repetível). Bloqueio vence allowlist")
    run.add_argument("--allow-command", action="append", default=[], metavar="EXE",
                     help="libera um executável (repetível). Só vale no nível 4")
    run.add_argument("--out", type=Path, default=None,
                     help="onde gravar o trace (padrão: reports/native/<sessão>.jsonl)")
    run.add_argument("--model", default=_env("HARNESS_MODEL", DEFAULT_MODEL))
    run.add_argument("--effort", default=_env("HARNESS_EFFORT", "high"),
                     choices=["low", "medium", "high", "xhigh", "max"])
    run.add_argument("--max-tokens", type=int, default=int(_env("HARNESS_MAX_TOKENS", "16000")))
    # Sem valor explícito, o teto do nível vale. Com um padrão fixo aqui, ``--level 0``
    # rodaria com 50 iterações em vez das 20 do nível — e o teto do nível seria letra
    # morta sem ninguém notar.
    run.add_argument("--max-iterations", type=int,
                     default=_env_int("HARNESS_MAX_ITERATIONS"))
    run.add_argument("--dry-run", action="store_true",
                     help="roteiro fixo em vez do modelo: exercita o loop inteiro "
                          "sem chave nem custo")
    run.add_argument("--json", action="store_true")

    sub.add_parser("sources", help="origens de trace disponíveis")
    return parser


def _force_utf8_output() -> None:
    """
    Garante que o relatório caiba no console, em vez de matar o processo.

    Bug real, achado rodando ``analyze`` sobre 55 transcripts de verdade: o console do
    Windows usa cp1252 por padrão, e o ``←`` do marcador de outlier levanta
    ``UnicodeEncodeError`` **no meio da impressão** — o relatório morre depois de já ter
    escrito metade.

    Os 170 testes não pegaram porque rodam sobre traces sintéticos, que não têm outlier:
    o caractere nunca chegava a ser impresso. É o modo de falha que este repositório
    documenta em outros — comportamento que só aparece com dado real — acontecendo com
    ele mesmo.

    ``errors="replace"`` em vez de ``"strict"``: um caractere que vira ``?`` é um defeito
    cosmético; um relatório que morre na metade faz perder a análise inteira.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # Stream redirecionado para algo que não aceita reconfiguração: seguir é
            # correto — o relatório ainda sai, só sem a garantia de acento.
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    args = build_parser().parse_args(argv)
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "compare":
        return _compare(args)
    if args.command == "power":
        return _power(args)
    if args.command == "run":
        return _run(args)
    if args.command == "sources":
        print("claude_code   ~/.claude/projects/**/*.jsonl")
        print("native        traces do harness deste repositório (harness-eng run)")
        print("openai        formato chat completions — OpenAI, Ollama, vLLM, Groq...")
        return 0
    return 1


def _env(name: str, fallback: str) -> str:
    """
    Configuração por ambiente, com ``.env`` quando ``python-dotenv`` está instalado.

    Sem dependência obrigatória: quem só analisa transcript não instala o extra do
    harness, e o ``.env`` que só o harness usa não pode virar requisito de import.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()
    return os.environ.get(name) or fallback


def _env_int(name: str) -> int | None:
    """
    Inteiro de uma variável de ambiente, ou ``None`` quando não está definida.

    ``None`` e não um padrão: quem consome precisa distinguir "o usuário escolheu" de
    "ninguém disse nada", porque no segundo caso é o nível de harness que decide.
    """
    raw = _env(name, "")
    try:
        return int(raw) if raw else None
    except ValueError:
        print(f"aviso: {name}={raw!r} não é um inteiro — ignorado", file=sys.stderr)
        return None


def _load(root: Path | None) -> tuple[TraceSet, Counter]:
    """
    Carrega todo trace legível sob ``root``, de qualquer origem conhecida.

    Cada arquivo é oferecido às origens em ordem, e a primeira que o reconhece fica com
    ele — é o que a porta ``TraceSource`` promete ao devolver ``None`` em vez de levantar.
    Na prática as duas origens se excluem sozinhas: o leitor nativo exige a marca de
    formato na primeira linha, e o do Claude Code não acha turno nenhum num arquivo nativo.

    Sem esse laço, ``analyze`` mediria os harnesses dos outros e não o próprio — o que
    seria uma piada num repositório que cobra medição de quem escreve harness.
    """
    # Precedência: argumento > HARNESS_TRACE_ROOT > ~/.claude/projects. O ``.env.example``
    # promete a variável desde o primeiro commit e nada a lia — configuração documentada e
    # não implementada é pior que ausente, porque quem confia nela não descobre que falhou.
    configured = _env("HARNESS_TRACE_ROOT", "")
    resolved = root or (Path(configured) if configured else default_root())
    if not resolved.exists():
        print(f"erro: diretório de traces não encontrado: {resolved}", file=sys.stderr)
        raise SystemExit(2)

    sources = (NativeSource(), ClaudeCodeSource())
    sessions = []
    for path in sorted({p for source in sources for p in source.discover(resolved)}):
        for source in sources:
            session = source.load(path)
            if session is not None and session.turns:
                sessions.append(session)
                break

    traces = TraceSet.of(sessions)
    if not traces.sessions:
        print(f"erro: nenhum trace legível em {resolved}", file=sys.stderr)
        raise SystemExit(2)

    skipped: Counter = Counter()
    for source in sources:
        skipped.update(source.skipped)
    return traces, skipped


def _num(value: float | int | None, *, dash: str = "—") -> str:
    """
    Número para o relatório, com ``None`` virando travessão em vez de exceção.

    Existe porque o pacote inteiro devolve ``None`` para "não deu para medir", e uma
    apresentação que não sabe disso transforma um trace curto — que é dado legítimo — em
    ``TypeError`` no meio do relatório.
    """
    return dash if value is None else f"{value:,.0f}"


def _pct(value: float | None, *, casas: int = 2, dash: str = "—") -> str:
    """Porcentagem, com ``None`` virando travessão. Mesma razão de :func:`_num`."""
    return dash if value is None else f"{value:.{casas}%}"


def _redact(text: str | None) -> str | None:
    """
    Hash estável de 8 caracteres. Estável para o mesmo texto continuar comparável entre
    execuções — redação que randomiza destrói a própria análise que justifica o relatório.
    """
    if not text:
        return text
    return "«" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:8] + "»"


def _analyze(args) -> int:
    traces, skipped_by_reason = _load(args.root)

    tools = analyse_tools(traces)
    loops = detect_loops(traces)
    context = profile_context(traces)
    cache = profile_cache(traces)
    cost = estimate_cost(traces)
    policy_fit = analyse_policy(traces)

    if args.json:
        payload = {
            "sessions": len(traces),
            "tools": tools.as_dict(),
            "loops": loops.as_dict(redact=args.redact),
            "context": context.as_dict(),
            "cache": cache.as_dict(),
            "cost": cost.as_dict(),
            "policy": policy_fit.as_dict(),
            "adapter_skipped": dict(skipped_by_reason),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    turns = sum(len(s) for s in traces)
    print(f"\n{len(traces)} sessões · {turns:,} turnos · "
          f"{tools.total_calls:,} chamadas de ferramenta\n")

    print("FERRAMENTAS")
    print(f"  taxa de erro geral: {_pct(tools.overall_error_rate)}")
    ranked = [t for t in tools.ranked_by_error_rate(only_with_signal=False) if t.calls >= 10]
    for health in ranked[:12]:
        flag = "  ← outlier" if health in tools.outliers() else ""
        signal = "" if health.has_signal else "  (amostra pequena)"
        taxa = _pct(health.error_rate, casas=1)
        print(f"    {health.name:18} {health.calls:>6} chamadas  {taxa:>6}{flag}{signal}")

    print("\nLOOPS")
    print(f"  {len(loops.repeats)} padrões de repetição, "
          f"{loops.wasted_calls} chamadas desperdiçadas ({_pct(loops.waste_rate, casas=1)})")
    print(f"  {len(loops.blind_retries)} retries cegos · {len(loops.oscillations)} oscilações")
    for repeat in loops.worst(3):
        bruto = repeat.sample_argument or ""
        argument = _redact(bruto) if args.redact else bruto
        print(f"    {repeat.count:>3}x  {repeat.tool:12} {str(argument)[:56]}")

    print("\nCONTEXTO")
    print(f"  pico            {context.peak:,} tokens")
    # As métricas devolvem ``None`` quando não houve turno medível, e está certo —
    # "ausência é ausência" atravessa o pacote inteiro. Quem estava errado era este
    # formatador, que estourava ``TypeError`` e derrubava o relatório inteiro num trace
    # curto: a métrica seguia a regra e a apresentação, não.
    print(f"  crescimento/turno  mediana {_num(context.median_growth)} · "
          f"média {_num(context.mean_growth)} · p95 {_num(context.p95_growth)}")
    if context.growth_concentration is not None:
        print(f"  concentração    {context.growth_concentration:.0%} do crescimento "
              f"vem dos 5% de turnos mais caros")
    print(f"  truncamentos    {context.truncations}")

    print("\nCACHE")
    print(f"  acerto          {_pct(cache.hit_rate, casas=1)}")
    escrito = "—" if cache.rewrite_ratio is None else f"{cache.rewrite_ratio:.3f}"
    print(f"  rewrite ratio   {escrito} (escrito por token lido)")

    print("\nCUSTO")
    lacuna = "" if cost.is_complete else "  (INCOMPLETO)"
    print(f"  total estimado  US$ {cost.total:,.2f}{lacuna}")
    for model, value in cost.ranked()[:5]:
        print(f"    {model:22} US$ {value:>10,.2f}")
    if not cost.is_complete:
        print(f"  sem preço: {', '.join(cost.unpriced_models)} ({cost.unpriced_tokens:,} tokens)")

    if policy_fit.with_policy:
        print("\nPOLÍTICA")
        print(f"  {policy_fit.verdict()}")
        for alvo, vezes in list(policy_fit.blocked_targets.items())[:5]:
            print(f"    {vezes:>3}x  tentou {alvo}")

    skipped = sum(skipped_by_reason.values())
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
    leitura = (
        "assimétrico: bootstrap e delta de Cliff são apropriados"
        if baseline["skewed"]
        else "aproximadamente simétrico"
    )
    print(f"  média/mediana {baseline['mean_over_median']} → {leitura}")

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


def _run(args) -> int:
    """
    Roda o harness mínimo, grava o trace nativo e diz como a execução terminou.

    O código de saída é ``0`` só quando o modelo terminou por vontade própria. Bater no
    teto de iterações, ser cortado por ``max_tokens`` ou morrer de erro de provedor
    devolvem ``1`` — porque num script de CI as três coisas são "não terminou", e um
    harness que devolve zero ao ser desligado no meio mente para a automação que o chama.
    """
    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        print(f"erro: workspace não é um diretório: {workspace}", file=sys.stderr)
        return 2

    # Sem --level, o comportamento antigo: leitura do workspace, sem politica. Com
    # --level, o nivel decide o conjunto inteiro e o trace grava o que foi concedido.
    policy = None
    if args.level is not None:
        policy = level_of(args.level)
        if args.allow:
            policy = policy.allowing(*args.allow)
        if args.block:
            policy = policy.blocking(*args.block)
        if args.allow_command:
            policy = policy.with_(allowed_commands=frozenset(args.allow_command))
        registry = policy_registry(policy, workspace)
    else:
        registry = workspace_registry(workspace)

    if args.dry_run:
        # Roteiro fixo: lista o workspace e encerra. Exercita loop, executor, trace e
        # métricas de ponta a ponta sem chave e sem custo — o caminho que prova que a
        # instalação funciona antes de alguém gastar a primeira requisição.
        client = ScriptedClient(
            [
                ModelResponse(
                    text="vou olhar o diretório",
                    tool_calls=(ToolCall(id="dry-1", name="list_dir", arguments={"path": "."}),),
                    stop_reason=StopReason.TOOL_USE,
                ),
                ModelResponse(text="(dry-run: sem modelo)", stop_reason=StopReason.END_TURN),
            ],
            model=f"{args.model} (dry-run)",
        )
    else:
        try:
            client = AnthropicClient(
                model=args.model,
                max_tokens=args.max_tokens,
                effort=args.effort,
                system=(
                    "Você é um agente de leitura. Use as ferramentas para responder sobre "
                    "o workspace e pare quando tiver a resposta."
                ),
            )
        except ModelError as exc:
            print(f"erro: {exc}", file=sys.stderr)
            return 2

    loop = AgentLoop(client, registry, policy=policy,
                     max_iterations=args.max_iterations, cwd=str(workspace))
    # O teto que de fato valeu: o do argumento, o do nível, ou o padrão. Reportar
    # ``args.max_iterations`` mostraria ``None`` quando o nível decidiu.
    loop_ceiling = loop._max_iterations
    try:
        outcome = loop.run(args.prompt)
    except KeyboardInterrupt:
        print("\ninterrompido", file=sys.stderr)
        return 130

    target = args.out or Path("reports") / "native" / f"{outcome.session.id}.jsonl"
    NativeSink().write(outcome.session, target)

    if args.json:
        print(json.dumps({**outcome.as_dict(), "trace": str(target)}, indent=2, ensure_ascii=False))
        return 0 if outcome.status.finished_on_its_own else 1

    usage = outcome.session.total_usage
    print(f"\n{outcome.status.value.upper()} — {outcome.detail}")
    if policy is not None:
        print(f"  política        {policy}")
        if registry.denials.total:
            motivos = ", ".join(
                f"{motivo.value} ({vezes}x)"
                for motivo, vezes in registry.denials.by_reason.items()
            )
            print(f"  paredes         {registry.denials.total} negativas — {motivos}")
    print(f"  iterações       {outcome.iterations}/{loop_ceiling}")
    print(f"  turnos          {outcome.turns}")
    print(f"  ferramentas     {outcome.tool_calls} chamadas")
    errors = sum(1 for _, result in outcome.session.paired_calls() if result and result.is_error)
    if outcome.tool_calls:
        print(f"  erros           {errors} ({errors / outcome.tool_calls:.0%})")
    print(f"  tokens          {usage.total_tokens:,} "
          f"(cache lido {usage.cache_read_tokens:,}, escrito {usage.cache_write_tokens:,})")
    if usage.cache_hit_rate is not None:
        print(f"  acerto de cache {usage.cache_hit_rate:.1%}")
    print(f"  trace           {target}")
    print(f"\n  medir: harness-eng analyze {target.parent}\n")
    return 0 if outcome.status.finished_on_its_own else 1


if __name__ == "__main__":
    sys.exit(main())
