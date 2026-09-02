"""
A porta da frente: ``Harness``.

O resto do pacote é feito de peças pequenas e explícitas — cliente, registro, loop, sink,
métricas — porque é assim que dá para trocar cada uma e testar todas. Só que "explícito"
vira "chato" quando alguém só quer rodar um agente com duas ferramentas: eram quatro
imports e um JSON Schema escrito à mão antes deste módulo existir.

Esta classe é o **composition root**: o lugar onde as peças se conectam. Ela não implementa
nada — monta. Tudo que ela faz continua possível na mão, e é de propósito: a fachada
existe para o caminho comum ser curto, não para ser o único caminho.

::

    from harness_eng import Harness

    h = Harness(model="claude-opus-5")

    @h.tool
    def soma(a: int, b: int) -> int:
        "Soma dois números."
        return a + b

    resposta = h.run("quanto é 17 + 25?")
    print(resposta.final_text)
    print(resposta.report())
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .core.clients import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, AnthropicClient
from .core.loop import AgentLoop, RunOutcome
from .core.policy import Policy
from .core.policy import level as level_of
from .core.ports import ModelClient
from .core.toolkit import policy_registry
from .core.tools import ToolRegistry, workspace_registry
from .metrics.context import profile_cache
from .metrics.tools import analyse_tools
from .trace.model import TraceSet
from .trace.sources.native import NativeSink

#: Sistema padrão. Curto de propósito: instrução longa é escolha do usuário, e um prompt
#: de sistema gordo embutido na biblioteca apareceria como custo em toda execução de todo
#: mundo — sem que ninguém soubesse de onde veio.
DEFAULT_SYSTEM = (
    "Você é um agente. Use as ferramentas disponíveis para resolver a tarefa "
    "e pare quando tiver a resposta."
)


class Harness:
    """
    Um agente pronto para rodar: modelo, ferramentas, loop e trace.

    ``model`` monta um cliente da Anthropic. ``client`` aceita **qualquer** objeto com
    ``.model`` e ``.complete(conversa, tools)`` — é a porta
    :class:`~harness_eng.core.ports.ModelClient`, e é o que torna o pacote utilizável com
    qualquer provedor sem tocar no loop, nas métricas ou no formato de trace.

    ``level`` escolhe um nível de harness (0 a 4) e com ele o conjunto de ferramentas
    concedidas; ``policy`` aceita uma :class:`~harness_eng.core.policy.Policy` montada à
    mão, para quando os níveis nomeados não servirem. Os dois juntos são erro, e não a
    silenciosa precedência de um sobre o outro — que só apareceria como "por que este
    agente tem internet?" muito depois.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        client: ModelClient | None = None,
        system: str | None = DEFAULT_SYSTEM,
        level: int | None = None,
        policy: Policy | None = None,
        max_iterations: int | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str = "high",
        workspace: Path | str | None = None,
        file_tools: bool = False,
    ) -> None:
        if level is not None and policy is not None:
            raise ValueError("passe 'level' ou 'policy', não os dois")
        if policy is not None:
            self.policy: Policy | None = policy
        elif level is not None:
            self.policy = level_of(level)
        else:
            self.policy = None
        self.workspace = Path(workspace).resolve() if workspace else Path.cwd()

        # Três caminhos, em ordem de precedência. O nível decide o conjunto inteiro de
        # ferramentas; ``file_tools`` é o atalho antigo, sem política; e sem nenhum dos
        # dois o registro nasce vazio — uma biblioteca que dá leitura de disco ao modelo
        # por padrão decide sozinha uma questão que é do usuário.
        if self.policy is not None:
            self.registry = policy_registry(self.policy, self.workspace)
        elif file_tools:
            self.registry = workspace_registry(self.workspace)
        else:
            self.registry = ToolRegistry()
        self.client: ModelClient = client or AnthropicClient(
            model=model, system=system, max_tokens=max_tokens, effort=effort
        )
        self._max_iterations = max_iterations

    # ── ferramentas ──────────────────────────────────────────────────────────────────
    def tool(self, fn: Callable[..., Any] | None = None, **options: Any) -> Any:
        """
        Decorador que registra a função como ferramenta deste harness.

        ``@h.tool`` e ``h.add(fn)`` fazem a mesma coisa; o decorador existe porque
        declarar a ferramenta junto da função é o que as pessoas escrevem naturalmente.
        """
        from .core.tools import tool as make_tool

        def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
            decorated = make_tool(target, **options)
            self.registry.add(decorated)
            return decorated

        return decorate(fn) if fn is not None else decorate

    def add(self, *functions: Callable[..., Any]) -> Harness:
        """Registra funções já existentes. Devolve ``self`` para encadear."""
        self.registry.add(*functions)
        return self

    @property
    def tools(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.registry.specs)

    # ── execução ─────────────────────────────────────────────────────────────────────
    def run(self, prompt: str, *, save_to: Path | str | None = None) -> Run:
        """
        Roda a tarefa. ``save_to`` grava o trace nativo no caminho dado.

        O trace é opcional porque medir é opcional — mas é uma linha, e é o que separa
        "rodei um agente" de "sei o que o agente fez".
        """
        loop = AgentLoop(
            self.client,
            self.registry,
            policy=self.policy,
            max_iterations=self._max_iterations,
            cwd=str(self.workspace),
        )
        outcome = loop.run(prompt)
        path = None
        if save_to is not None:
            path = NativeSink().write(outcome.session, Path(save_to))
        return Run(outcome, path)


class Run:
    """
    O resultado de :meth:`Harness.run`: a resposta, o trace e as métricas.

    Embrulha :class:`~harness_eng.core.loop.RunOutcome` em vez de substituí-lo — quem quer
    a sessão canônica pega ``run.session`` e usa o pacote inteiro normalmente.
    """

    def __init__(self, outcome: RunOutcome, trace_path: Path | None = None) -> None:
        self.outcome = outcome
        self.trace_path = trace_path

    def __getattr__(self, name: str) -> Any:
        # Delegação para o RunOutcome: final_text, status, iterations, session, tool_calls.
        return getattr(self.outcome, name)

    def __repr__(self) -> str:
        return (
            f"<Run {self.outcome.status.value} "
            f"{self.outcome.iterations} iterações, {self.outcome.tool_calls} ferramentas>"
        )

    @property
    def ok(self) -> bool:
        """Se o modelo terminou por vontade própria. Teto de iteração não conta."""
        return self.outcome.status.finished_on_its_own

    def save(self, path: Path | str) -> Path:
        """Grava o trace nativo, legível por ``harness-eng analyze``."""
        self.trace_path = NativeSink().write(self.outcome.session, Path(path))
        return self.trace_path

    def report(self) -> str:
        """
        As métricas desta execução, em texto.

        É o círculo fechado em uma chamada: o trace que o loop acabou de escrever passa
        pelas mesmas métricas que medem harness de terceiro, sem adapter no meio.
        """
        traces = TraceSet.of([self.outcome.session])
        health = analyse_tools(traces)
        cache = profile_cache(traces)
        usage = self.outcome.session.total_usage

        lines = [
            f"{self.outcome.status.value} — {self.outcome.detail}",
            f"  iterações   {self.outcome.iterations}",
            f"  ferramentas {health.total_calls} chamadas, {health.total_errors} erros",
        ]
        for tool_health in health.ranked_by_error_rate(only_with_signal=False):
            if tool_health.errors:
                lines.append(
                    f"    {tool_health.name}: {tool_health.errors}/{tool_health.calls} falharam"
                )
        negadas = self.outcome.session.metadata.get("denials") or {}
        if negadas.get("total"):
            motivos = ", ".join(f"{r} ({c}x)" for r, c in (negadas.get("by_reason") or {}).items())
            lines.append(f"  paredes     {negadas['total']} negativas — {motivos}")
        lines.append(f"  tokens      {usage.total_tokens:,}")
        if cache.hit_rate:
            lines.append(f"  cache       {cache.hit_rate:.1%} de acerto")
        if self.trace_path:
            lines.append(f"  trace       {self.trace_path}")
        return "\n".join(lines)


def quick(prompt: str, *functions: Callable[..., Any], model: str = DEFAULT_MODEL) -> str:
    """
    Uma linha, do prompt à resposta::

        from harness_eng import quick
        print(quick("quanto é 17+25?", soma))

    Descarta o trace de propósito: é o atalho para experimentar, e um atalho que grava
    arquivo escondido no diretório de quem chamou seria uma surpresa desagradável. Para
    medir, use :class:`Harness`.
    """
    harness = Harness(model=model)
    harness.add(*functions)
    return harness.run(prompt).final_text


__all__ = ["Harness", "Run", "quick", "DEFAULT_SYSTEM"]
