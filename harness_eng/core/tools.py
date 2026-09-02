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
import inspect
import json
import re
import time
import types
import typing
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, get_args, get_origin

from ..trace.model import ToolCall, ToolResult
from .policy import DenialLog
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
        #: O que a política negou, por motivo e alvo. Mesmo padrão de
        #: ``ClaudeCodeSource.skipped``: negar em silêncio esconde a informação mais útil
        #: que um nível de harness produz — se ele serve ou não para a tarefa.
        self.denials = DenialLog()

    def register(self, spec: ToolSpec, handler: Handler) -> "ToolRegistry":
        """Registro explícito: você traz o schema. Prefira :meth:`add` quando for função."""
        if spec.name in self._tools:
            raise ValueError(f"ferramenta duplicada: {spec.name}")
        self._tools[spec.name] = RegisteredTool(spec=spec, handler=handler)
        return self

    def add(self, *functions: Callable[..., Any]) -> "ToolRegistry":
        """
        Registra funções direto, derivando o schema da assinatura.

        Aceita função decorada com :func:`tool` e função crua — na crua, deriva na hora.
        Existe para o caminho comum ser uma linha: ``registry.add(soma, subtrai)``.
        """
        for fn in functions:
            spec = getattr(fn, "spec", None) or describe(fn)
            handler = getattr(fn, "handler", None) or as_handler(fn)
            self.register(spec, handler)
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
            # Negativa de política é contada à parte. Para o modelo as duas são erro — ele
            # precisa ver as duas —, mas para a medição são opostas: falha significa que
            # algo quebrou, negativa significa que a política funcionou. Somá-las produz
            # uma taxa de erro que sobe quando você aperta a segurança.
            decision = getattr(exc, "decision", None)
            if decision is not None:
                self.denials.record(decision)
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


# ── @tool: a assinatura vira o schema ────────────────────────────────────────────────

#: As duas formas de escrever união. ``Optional[X]`` produz ``typing.Union``; ``X | None``
#: produz ``types.UnionType``, que é outra classe — as duas precisam ser reconhecidas.
_UNION_ORIGINS = {typing.Union, types.UnionType}

#: Tipos Python que viram tipo JSON Schema direto.
_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _json_type(annotation: Any, parameter: str, function: str) -> dict:
    """
    Traduz uma anotação Python para um pedaço de JSON Schema.

    Sem anotação, não há schema — e por isso o erro é levantado na **decoração**, não na
    execução. Um schema adivinhado passa no teste, chega ao modelo com o tipo errado e
    reaparece como "o modelo passou string onde eu queria número", que é caro de
    diagnosticar e barato de prevenir aqui.
    """
    if annotation is inspect.Parameter.empty:
        raise ValueError(
            f"{function}(): o parâmetro '{parameter}' não tem anotação de tipo. "
            f"O modelo lê o schema para saber o que mandar — anote (ex.: {parameter}: str)."
        )

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[X] / X | None: o tipo é o de X, e a ausência vira "não obrigatório".
    #
    # As duas sintaxes precisam ser checadas. ``Optional[float]`` tem origem
    # ``typing.Union``; ``float | None`` tem origem ``types.UnionType``, que é uma classe
    # diferente — e a primeira versão disto comparava o *texto* da origem, então a forma
    # moderna, que é a que as pessoas escrevem hoje, caía direto no erro de "tipo sem
    # tradução". Achado pelo teste que cobria as duas.
    if origin in _UNION_ORIGINS:
        real = [a for a in args if a is not type(None)]
        if len(real) == 1:
            return _json_type(real[0], parameter, function)
        raise ValueError(
            f"{function}(): união de vários tipos em '{parameter}' não vira schema claro. "
            f"Escolha um tipo, ou receba str e converta dentro da função."
        )

    if annotation in _JSON_TYPES:
        return {"type": _JSON_TYPES[annotation]}

    if origin in (list, tuple) or annotation in (list, tuple):
        items = _json_type(args[0], parameter, function) if args else {"type": "string"}
        return {"type": "array", "items": items}

    if origin is dict or annotation is dict:
        return {"type": "object"}

    raise ValueError(
        f"{function}(): não sei virar schema o tipo {annotation!r} de '{parameter}'. "
        f"Tipos suportados: str, int, float, bool, list[...], dict e Optional deles."
    )


def _split_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """
    Separa a descrição da ferramenta das descrições de parâmetro.

    Aceita as duas convenções que as pessoas de fato escrevem — ``Args:``/``Params:`` do
    Google e ``:param x:`` do reST. Não é firula: a descrição do parâmetro é o que o
    modelo lê para decidir o que mandar, e obrigar um formato só faria a maioria dos
    docstrings existentes não render nada.
    """
    if not doc:
        return "", {}

    lines = inspect.cleandoc(doc).splitlines()
    description: list[str] = []
    params: dict[str, str] = {}
    in_args = False

    for line in lines:
        stripped = line.strip()
        if re.match(r"^(Args|Arguments|Params|Parameters|Argumentos|Parâmetros)\s*:$", stripped):
            in_args = True
            continue
        rest = re.match(r"^:param\s+(\w+)\s*:\s*(.+)$", stripped)
        if rest:
            params[rest.group(1)] = rest.group(2).strip()
            continue
        if in_args:
            entry = re.match(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$", stripped)
            if entry:
                params[entry.group(1)] = entry.group(2).strip()
                continue
            if not stripped:
                continue
            in_args = False
        if not stripped.startswith(":"):
            description.append(line)

    return "\n".join(description).strip(), params


def describe(fn: Callable[..., Any], *, name: str | None = None,
             description: str | None = None) -> ToolSpec:
    """
    Deriva o :class:`ToolSpec` de uma função: nome, descrição e schema.

    Esta é a peça que tira o JSON Schema escrito à mão do caminho. Escrever schema à mão
    não é só chato — é uma segunda fonte de verdade que sai de sincronia com a função na
    primeira vez que alguém renomeia um parâmetro, e o sintoma é o modelo mandando um
    argumento que a função não aceita.
    """
    signature = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn)
    except NameError as exc:
        # Com ``from __future__ import annotations`` as anotações são strings, e resolvê-las
        # exige que o nome exista no módulo. Tipo declarado dentro de outra função, ou
        # importado só sob ``TYPE_CHECKING``, estoura um ``NameError`` cru que não diz o
        # que fazer. A mensagem abaixo diz.
        raise ValueError(
            f"{fn.__name__}(): não consegui resolver uma anotação de tipo ({exc}). "
            f"Tipo definido dentro de função, ou importado só sob TYPE_CHECKING, não é "
            f"visível aqui — use um tipo de módulo, ou receba str e converta na função."
        ) from exc
    doc, param_docs = _split_docstring(fn.__doc__)

    properties: dict[str, dict] = {}
    required: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise ValueError(
                f"{fn.__name__}(): *args/**kwargs não viram schema. "
                f"Declare os parâmetros que o modelo pode mandar."
            )
        annotation = hints.get(parameter.name, parameter.annotation)
        schema = _json_type(annotation, parameter.name, fn.__name__)
        if parameter.name in param_docs:
            schema["description"] = param_docs[parameter.name]
        properties[parameter.name] = schema
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)

    return ToolSpec(
        name=name or fn.__name__,
        description=description or doc or fn.__name__,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    )


def _coerce(value: Any) -> str:
    """
    O retorno da função vira o texto que o modelo lê.

    ``dict`` e ``list`` saem como JSON em vez de ``repr`` do Python: aspas simples e
    ``None`` no lugar de ``null`` são coisas que o modelo lê pior, e é gratuito acertar.
    ``None`` vira string vazia de propósito — e string vazia é contada como falha
    silenciosa pelo executor, que é a leitura correta de uma ferramenta que não devolveu nada.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def as_handler(fn: Callable[..., Any]) -> Handler:
    """
    Embrulha a função para o executor: recebe o mapa de argumentos, chama com kwargs.

    Argumento faltando ou sobrando vira :class:`ToolError` com mensagem legível em vez de
    ``TypeError`` cru. A diferença importa porque essa mensagem vai **para o modelo**, e
    "faltou o argumento 'a'" o faz corrigir na próxima chamada; um traceback, não.
    """
    signature = inspect.signature(fn)

    def handler(arguments: Mapping[str, object]) -> str:
        try:
            bound = signature.bind(**arguments)
        except TypeError as exc:
            raise ToolError(f"argumentos inválidos para {fn.__name__}(): {exc}") from exc
        bound.apply_defaults()
        return _coerce(fn(*bound.args, **bound.kwargs))

    return handler


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None,
         description: str | None = None) -> Any:
    """
    Decorador: transforma uma função Python numa ferramenta, sem schema à mão.

    ::

        @tool
        def soma(a: int, b: int) -> int:
            \"\"\"Soma dois números.

            Args:
                a: primeira parcela
                b: segunda parcela
            \"\"\"
            return a + b

    A função continua sendo uma função normal — dá para chamar, testar e importar como
    sempre. O decorador só pendura ``.spec`` e ``.handler`` nela, que é o que o registro
    consome. Nada de classe base, nada de registro global: uma ferramenta é uma função.
    """

    def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        target.spec = describe(target, name=name, description=description)  # type: ignore[attr-defined]
        target.handler = as_handler(target)  # type: ignore[attr-defined]
        return target

    return decorate(fn) if fn is not None else decorate


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
