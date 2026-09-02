"""
harness-eng — rodar agentes e **medir** o que eles fizeram.

O caminho curto::

    from harness_eng import Harness

    h = Harness(model="claude-opus-5")

    @h.tool
    def soma(a: int, b: int) -> int:
        "Soma dois números."
        return a + b

    run = h.run("quanto é 17 + 25?")
    print(run.final_text)
    print(run.report())

Níveis de harness: ``Harness(level=2)`` concede um conjunto de ferramentas e registra
no trace o que foi concedido — e o que foi negado. Ver :mod:`harness_eng.core.policy`.

Qualquer provedor serve. ``Harness(client=...)`` aceita qualquer objeto com ``.model`` e
``.complete(conversa, tools)`` — ver :mod:`harness_eng.core.ports`.

As peças continuam acessíveis para quem quiser montar na mão:
:mod:`harness_eng.trace` (formato canônico), :mod:`harness_eng.metrics`,
:mod:`harness_eng.stats` e :mod:`harness_eng.core`.
"""
from .core.loop import AgentLoop, LoopStatus, RunOutcome
from .core.policy import (
    BUILDER,
    LEVELS,
    OPERATOR,
    READER,
    RESEARCHER,
    SEALED,
    Policy,
    level,
)
from .core.ports import ModelClient, ModelError, ModelResponse, ToolSpec
from .core.tools import ToolError, ToolRegistry, describe, tool
from .harness import DEFAULT_SYSTEM, Harness, Run, quick

__version__ = "0.2.0"

__all__ = [
    "Harness",
    "Run",
    "quick",
    "tool",
    "describe",
    "AgentLoop",
    "LoopStatus",
    "RunOutcome",
    "ToolRegistry",
    "ToolError",
    "ModelClient",
    "ModelError",
    "ModelResponse",
    "ToolSpec",
    "Policy",
    "level",
    "LEVELS",
    "SEALED",
    "READER",
    "RESEARCHER",
    "BUILDER",
    "OPERATOR",
    "DEFAULT_SYSTEM",
    "__version__",
]
