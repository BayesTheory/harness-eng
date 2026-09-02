"""
Registro e execução de ferramentas.

Duas responsabilidades, e a segunda é a que interessa ao projeto: executar a chamada e
**registrar honestamente o que aconteceu**. Um executor que transforma toda falha em
string de erro no contexto do modelo produz um trace onde nada nunca falha — e aí a taxa
de erro por ferramenta, a métrica mais reveladora deste repositório, mede zero em toda
ferramenta e não serve para nada.

As ferramentas embutidas são de leitura só, presas a um diretório raiz. O harness existe
para ser medido, não para ser útil: um executor de comando arbitrário mudaria o assunto
do repositório, e a superfície de risco junto.
"""
from __future__ import annotations

import fnmatch
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..trace.model import ToolCall, ToolResult
from .ports import ToolSpec

#: Assinatura de um executor: recebe os argumentos, devolve texto.
#:
#: Falha se sinaliza levantando exceção, não devolvendo uma string que começa com "erro".
#: A distinção é o que permite ``ToolResult.is_error`` significar alguma coisa — e é
#: exatamente o que um harness descuidado apaga ao capturar tudo e concatenar em texto.
Handler = Callable[[Mapping[str, object]], str]

#: Teto de caracteres devolvidos por uma ferramenta de leitura.
#:
#: Existe porque leitura grande é o que estoura contexto: nos 54 transcripts analisados,
#: 27% do crescimento de contexto veio de 5% dos turnos. Truncar é uma escolha de harness,
#: e — sendo escolha — o resultado diz que truncou, em vez de deixar o modelo supor que
#: leu o arquivo inteiro.
MAX_OUTPUT_CHARS = 20_000


class ToolError(Exception):
    """
    Falha esperada de ferramenta: arquivo ausente, caminho fora da raiz, argumento faltando.

    Separada de exceção inesperada de propósito. As duas viram ``is_error=True`` no trace,
    mas só a inesperada é bug do harness — e o relatório precisa poder distinguir "a
    ferramenta recusou corretamente" de "o executor quebrou".
    """


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    spec: ToolSpec
    handler: Handler


class ToolRegistry:
    """
    O conjunto de ferramentas que o modelo pode chamar, e quem as executa.

    Guarda ordem de registro. Não é estética: a lista de ferramentas entra no prefixo da
    requisição, antes do sistema e das mensagens, e qualquer variação nela invalida o
    cache de tudo que vem depois. Um registro que iterasse um ``set`` reordenaria a lista
    entre execuções e derrubaria o acerto de cache sem nenhum sintoma visível além da
    conta no fim do mês.
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, handler: Handler) -> "ToolRegistry":
        if spec.name in self._tools:
            raise ValueError(f"ferramenta duplicada: {spec.name}")
        self._tools[spec.name] = RegisteredTool(spec=spec, handler=handler)
        return self

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        """As declarações, em ordem estável de registro. Ver docstring da classe."""
        return tuple(tool.spec for tool in self._tools.values())

    def execute(self, call: ToolCall) -> ToolResult:
        """
        Roda uma chamada e devolve o resultado canônico, com duração medida.

        Nunca levanta: o loop precisa devolver um resultado ao modelo mesmo quando a
        ferramenta explodiu, e uma exceção que suba daqui mata a sessão inteira por causa
        de uma chamada — perdendo, junto, o trace que explicaria o que houve.
        """
        started = time.perf_counter()

        tool = self._tools.get(call.name)
        if tool is None:
            # Modelo inventou uma ferramenta. É um modo de falha real e vale contar
            # separado: quase sempre significa descrição ambígua, ou ferramenta removida
            # sem tirar a menção do prompt.
            return self._failure(call, f"ferramenta desconhecida: {call.name}", started)

        try:
            output = tool.handler(call.arguments)
        except ToolError as exc:
            return self._failure(call, str(exc), started)
        except Exception as exc:  # noqa: BLE001 — ver docstring: nunca levanta
            return self._failure(call, f"{type(exc).__name__}: {exc}", started)

        text = output if isinstance(output, str) else str(output)
        if len(text) > MAX_OUTPUT_CHARS:
            text = text[:MAX_OUTPUT_CHARS] + f"\n[truncado em {MAX_OUTPUT_CHARS} caracteres]"

        return ToolResult(
            call_id=call.id,
            is_error=False,
            content=text,
            # Sem bloco quando não veio nada. Marcar ``("text",)`` aqui faria
            # ``ToolResult.is_empty`` devolver False para um retorno vazio e esconderia
            # justamente a falha silenciosa que a métrica existe para achar: sucesso sem
            # saída nenhuma, que deixa o modelo sem sinal para o passo seguinte.
            content_kinds=("text",) if text else (),
            duration=timedelta(seconds=time.perf_counter() - started),
        )

    @staticmethod
    def _failure(call: ToolCall, message: str, started: float) -> ToolResult:
        return ToolResult(
            call_id=call.id,
            is_error=True,
            content=message,
            content_kinds=("text",),
            duration=timedelta(seconds=time.perf_counter() - started),
        )


# ── ferramentas embutidas ────────────────────────────────────────────────────────────

def _require_str(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"argumento obrigatório ausente ou vazio: {key}")
    return value


def _resolve_inside(root: Path, candidate: str) -> Path:
    """
    Resolve ``candidate`` sob ``root`` e recusa qualquer coisa que escape.

    ``resolve()`` antes de comparar, não depois: sem isso, ``..`` e link simbólico passam
    por uma checagem textual e leem fora da raiz.
    """
    base = root.resolve()
    raw = Path(candidate)
    target = raw.resolve() if raw.is_absolute() else (base / raw).resolve()
    if target != base and base not in target.parents:
        raise ToolError(f"caminho fora da raiz do workspace: {candidate}")
    return target


def read_file_tool(root: Path) -> tuple[ToolSpec, Handler]:
    spec = ToolSpec(
        name="read_file",
        description=(
            "Lê um arquivo de texto do workspace. O caminho é relativo à raiz. "
            "Saída truncada acima de 20.000 caracteres, e o corte é anunciado."
        ),
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "caminho relativo à raiz"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )

    def handler(arguments: Mapping[str, object]) -> str:
        target = _resolve_inside(root, _require_str(arguments, "path"))
        if not target.is_file():
            raise ToolError(f"não é um arquivo legível: {target.name}")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"arquivo não é texto UTF-8: {target.name}") from exc

    return spec, handler


def list_dir_tool(root: Path) -> tuple[ToolSpec, Handler]:
    spec = ToolSpec(
        name="list_dir",
        description=(
            "Lista o conteúdo de um diretório do workspace. "
            "Diretórios saem com barra final."
        ),
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "diretório, relativo à raiz"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )

    def handler(arguments: Mapping[str, object]) -> str:
        target = _resolve_inside(root, _require_str(arguments, "path"))
        if not target.is_dir():
            raise ToolError(f"não é um diretório: {target.name}")
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        return "\n".join(f"{p.name}/" if p.is_dir() else p.name for p in entries)

    return spec, handler


def find_files_tool(root: Path) -> tuple[ToolSpec, Handler]:
    spec = ToolSpec(
        name="find_files",
        description="Procura arquivos por padrão glob a partir da raiz do workspace.",
        input_schema={
            "type": "object",
            "properties": {"pattern": {"type": "string", "description": "padrão glob"}},
            "required": ["pattern"],
            "additionalProperties": False,
        },
    )

    def handler(arguments: Mapping[str, object]) -> str:
        pattern = _require_str(arguments, "pattern")
        base = root.resolve()
        found = []
        for candidate in base.rglob("*"):
            if not candidate.is_file():
                continue
            # Separador normalizado para o padrão do modelo casar igual nos dois sistemas.
            relative = str(candidate.relative_to(base)).replace("\\", "/")
            if fnmatch.fnmatch(relative, pattern):
                found.append(relative)
        return "\n".join(sorted(found))

    return spec, handler


def workspace_registry(root: Path, tools: Sequence[str] | None = None) -> ToolRegistry:
    """
    O conjunto padrão: leitura só, preso a ``root``.

    ``tools`` restringe o conjunto — usado nos testes para provar que o loop reage a uma
    ferramenta ausente do jeito que se espera, em vez de a ausência virar exceção.
    """
    builders = {
        "read_file": read_file_tool,
        "list_dir": list_dir_tool,
        "find_files": find_files_tool,
    }
    chosen = list(builders) if tools is None else list(tools)
    registry = ToolRegistry()
    for name in chosen:
        if name not in builders:
            raise ValueError(f"ferramenta embutida desconhecida: {name}")
        spec, handler = builders[name](root)
        registry.register(spec, handler)
    return registry
