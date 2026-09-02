"""
A regra de dependência, verificada automaticamente.

O formato canônico só é agnóstico enquanto ninguém encosta um import de provedor nele.
E ninguém mantém regra de import na revisão de código por muito tempo — este teste falha
quando a seta inverte, e é o que impede o repositório de voltar ao estado anterior em
alguns meses com a mesma aparência de camadas.

A regra: ``trace/model.py``, ``metrics/`` e ``stats/`` são puros. Recebem dado, devolvem
número. Não conhecem provedor, não tocam disco, não abrem rede. Isso é o que torna
possível testar toda a medição sem transcript, sem chave de API e sem instalar nada.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "harness_eng"

#: Pacotes que a camada pura não pode importar. ``statistics`` e ``random`` são da
#: biblioteca padrão e estão liberados: matemática não é infraestrutura.
FORBIDDEN = {
    "anthropic", "openai", "httpx", "requests", "aiohttp", "urllib3",
    "scipy", "numpy", "pandas", "sklearn", "matplotlib",
    "fastapi", "flask", "django", "sqlalchemy", "boto3",
}

#: Módulos que precisam ser puros. O adapter de origem lê disco por definição; a CLI e o
#: harness falam com o mundo. A pureza é exigida onde ela compra alguma coisa.
PURE_MODULES = ["trace/model.py", "trace/ports.py", "metrics", "stats"]


def _python_files(relative: str) -> list[Path]:
    target = PACKAGE / relative
    if target.is_file():
        return [target]
    return [p for p in target.rglob("*.py") if "__pycache__" not in p.parts]


def _imported_roots(path: Path) -> set[str]:
    """Módulos de topo importados, incluindo imports dentro de função."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _imported_modules(path: Path) -> set[str]:
    """
    Módulos importados pelo nome completo, incluindo os relativos.

    Devolve ``core.policy`` tanto para ``from harness_eng.core.policy import X`` quanto
    para ``from ..core.policy import X``, normalizando as duas formas de escrever a mesma
    dependência. Ler o AST em vez de grepar o texto não é preciosismo: a primeira versão
    desta regra grepava o arquivo, e o primeiro módulo a citar ``harness_eng.core`` **num
    docstring** — para dizer que NÃO o importa — reprovou. Um teste de arquitetura que dá
    falso positivo é desligado numa semana, e a lição já estava escrita mais abaixo neste
    mesmo arquivo, para outra regra.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _is_core(module: str) -> bool:
    """Se o módulo importado é o harness — na forma absoluta ou na relativa."""
    return module == "core" or module.startswith(("core.", "harness_eng.core"))


ALL_PURE_FILES = [p for relative in PURE_MODULES for p in _python_files(relative)]


@pytest.mark.parametrize("path", ALL_PURE_FILES, ids=lambda p: p.name)
def test_pure_layer_has_no_heavy_dependency(path: Path) -> None:
    offenders = _imported_roots(path) & FORBIDDEN
    assert not offenders, (
        f"{path.relative_to(PACKAGE)} importa {sorted(offenders)}. Esta camada recebe dado e "
        f"devolve número — mova o acesso ao mundo para trace/sources/ ou core/."
    )


@pytest.mark.parametrize("path", ALL_PURE_FILES, ids=lambda p: p.name)
def test_pure_layer_does_not_touch_the_filesystem(path: Path) -> None:
    """
    Nem ``open``, nem ``Path.read_*``, nem ``os.environ``.

    Uma métrica que lê arquivo por conta própria não é testável sem arquivo, e é assim
    que uma suíte rápida vira uma suíte que precisa de fixture no disco.
    """
    source = path.read_text(encoding="utf-8")
    for marker in ("open(", ".read_text(", ".read_bytes(", "os.environ", "os.getenv"):
        assert marker not in source, (
            f"{path.relative_to(PACKAGE)} usa {marker!r}: acesso a disco/ambiente pertence "
            f"aos adapters, não à camada de medição."
        )


def _code_identifiers_and_literals(path: Path) -> set[str]:
    """
    Nomes e literais que o CÓDIGO usa, sem docstring nem comentário.

    A distinção importa: um docstring que cita ``claude_code`` como exemplo de valor de
    ``source`` é documentação legítima; uma comparação ``if source == "claude_code"`` é
    acoplamento. A primeira versão deste teste grepava o arquivo inteiro e reprovou o
    próprio docstring — um teste de arquitetura que dá falso positivo é abandonado em uma
    semana, e aí não protege nada.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings = {
        ast.get_docstring(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            found.add(node.attr.lower())
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in docstrings:
                found.add(node.value.lower())
    return found


def test_the_canonical_model_knows_no_specific_harness() -> None:
    """
    ``trace/model.py`` não pode ter caso especial de Claude Code, OpenAI ou qualquer origem.

    É o vocabulário comum; no momento em que ele carrega um caso especial de um harness,
    deixa de ser comum e o segundo adapter passa a lutar contra ele.
    """
    used = _code_identifiers_and_literals(PACKAGE / "trace" / "model.py")
    for name in ("claude_code", "tooluseresult", "cache_creation_input_tokens", "jsonl"):
        assert name not in used, (
            f"trace/model.py usa {name!r} no código — detalhe de origem vazou para o "
            f"formato canônico (menção em docstring é permitida; uso, não)"
        )


def test_metrics_do_not_import_sources() -> None:
    """Métrica fala com o formato canônico, nunca com um leitor concreto."""
    for path in _python_files("metrics"):
        source = path.read_text(encoding="utf-8")
        assert "trace.sources" not in source and "claude_code" not in source, (
            f"{path.relative_to(PACKAGE)} conhece um adapter concreto"
        )


def test_stats_do_not_import_trace() -> None:
    """
    A camada estatística é independente do domínio de trace.

    Ela recebe ``Sequence[float]`` e ``Mapping[str, float]``. Isso não é purismo: torna
    o pacote de estatística reutilizável para qualquer comparação pareada, e o mantém
    testável contra distribuições sintéticas com resposta conhecida.
    """
    for path in _python_files("stats"):
        assert "trace" not in _imported_roots(path), (
            f"{path.relative_to(PACKAGE)} importa a camada de trace"
        )


def test_nothing_measurable_depends_on_the_harness() -> None:
    """
    ``core/`` é consumidor do formato canônico, nunca dependência dele.

    A seta aponta num sentido só: o harness importa ``trace``, e ``trace``, ``metrics`` e
    ``stats`` não sabem que ele existe. Inverter é tentador — o harness tem o
    ``ToolSpec``, e uma métrica sobre descrição de ferramenta ficaria "natural" importando
    dele. No dia em que isso acontecer, medir um harness de terceiro passa a arrastar o
    loop, o cliente de modelo e o SDK junto, e a promessa de rodar a suíte sem chave de
    API morre sem que nenhum teste reclame.
    """
    for relative in ("trace/model.py", "trace/ports.py", "trace/sources", "metrics", "stats"):
        for path in _python_files(relative):
            offenders = sorted(m for m in _imported_modules(path) if _is_core(m))
            assert not offenders, (
                f"{path.relative_to(PACKAGE)} importa o harness ({offenders}). A camada de "
                f"medição existe para medir harness de terceiro também — ela não pode "
                f"depender do nosso."
            )


def test_the_harness_talks_to_one_provider_in_one_place() -> None:
    """
    Só ``core/clients.py`` pode importar SDK de provedor.

    O loop recebe um ``ModelClient`` pronto; no dia em que ele importar ``anthropic``
    direto, a porta vira decoração e o adapter da OpenAI passa a ser reescrita em vez de
    acréscimo.
    """
    for path in _python_files("core"):
        if path.name == "clients.py":
            continue
        offenders = _imported_roots(path) & FORBIDDEN
        assert not offenders, (
            f"core/{path.name} importa {sorted(offenders)}: o acesso a provedor mora em "
            f"core/clients.py, atrás da porta ModelClient."
        )
