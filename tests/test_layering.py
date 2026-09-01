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

SRC = Path(__file__).resolve().parents[1] / "src" / "harness_eng"

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
    target = SRC / relative
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


ALL_PURE_FILES = [p for relative in PURE_MODULES for p in _python_files(relative)]


@pytest.mark.parametrize("path", ALL_PURE_FILES, ids=lambda p: p.name)
def test_pure_layer_has_no_heavy_dependency(path: Path) -> None:
    offenders = _imported_roots(path) & FORBIDDEN
    assert not offenders, (
        f"{path.relative_to(SRC)} importa {sorted(offenders)}. Esta camada recebe dado e "
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
            f"{path.relative_to(SRC)} usa {marker!r}: acesso a disco/ambiente pertence "
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
    used = _code_identifiers_and_literals(SRC / "trace" / "model.py")
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
            f"{path.relative_to(SRC)} conhece um adapter concreto"
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
            f"{path.relative_to(SRC)} importa a camada de trace"
        )
