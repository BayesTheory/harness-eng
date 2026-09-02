"""
As ferramentas que cada nível concede, e a fronteira onde a política é aplicada.

:mod:`harness_eng.core.tools` cuida da mecânica — registro, ``@tool``, schema. Este módulo
cuida do **conteúdo**: quais ferramentas existem em cada nível e onde exatamente a
negativa acontece.

A separação importa porque a fronteira é o lugar que precisa estar certo. Toda checagem
mora dentro do handler, imediatamente antes do efeito — não na montagem do registro, não
no loop. Um agente que recebe a ferramenta ``fetch_url`` e só descobre o bloqueio ao
chamá-la é o comportamento correto: ele aprende *qual* parede bateu e se adapta, e a
parede fica contada. Filtrar a ferramenta da lista esconde a informação dos dois lados.
"""
from __future__ import annotations

import shlex
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from .policy import (
    Decision,
    FileAccess,
    NetworkAccess,
    Policy,
    ShellAccess,
)
from .ports import ToolSpec
from .tools import (
    Handler,
    ToolError,
    ToolRegistry,
    _require_str,
    _resolve_inside,
    find_files_tool,
    list_dir_tool,
    read_file_tool,
)

#: Teto de bytes que uma busca na rede devolve. Leitura grande é o que estoura contexto —
#: nos 54 transcripts, 27% do crescimento veio de 5% dos turnos, e página web é a fonte
#: mais fácil de trazer 300 KB sem querer.
MAX_FETCH_BYTES = 200_000

#: Segundos até desistir de uma requisição. Sem timeout, uma URL que não responde trava a
#: execução inteira — e trava sem virar erro, que é o pior formato de falha.
FETCH_TIMEOUT = 15

#: Segundos até matar um comando. Mesma razão.
COMMAND_TIMEOUT = 60


class PolicyDenied(ToolError):
    """
    Negativa de política, distinta de falha de ferramenta.

    As duas viram ``is_error=True`` no trace — o modelo precisa ver as duas como erro. Mas
    para a **medição** são coisas opostas: falha significa que algo quebrou, negativa
    significa que a política funcionou. Somá-las produziria uma taxa de erro que sobe
    quando você aperta a segurança, sugerindo que apertar a segurança quebra o agente.
    """

    def __init__(self, decision: Decision) -> None:
        super().__init__(decision.message)
        self.decision = decision


def _deny(decision: Decision) -> None:
    if not decision.allowed:
        raise PolicyDenied(decision)


# ── escrita ──────────────────────────────────────────────────────────────────────────

def write_file_tool(root: Path, policy: Policy) -> tuple[ToolSpec, Handler]:
    spec = ToolSpec(
        name="write_file",
        description=(
            "Escreve texto num arquivo do workspace, criando diretórios se preciso. "
            "Sobrescreve o conteúdo anterior."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "caminho relativo à raiz"},
                "content": {"type": "string", "description": "o conteúdo a gravar"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    )

    def handler(arguments):
        _deny(policy.check_write())
        target = _resolve_inside(root, _require_str(arguments, "path"))
        content = arguments.get("content")
        if not isinstance(content, str):
            raise ToolError("o argumento 'content' precisa ser texto")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"gravado: {target.name} ({len(content)} caracteres)"

    return spec, handler


# ── rede ─────────────────────────────────────────────────────────────────────────────

class _PolicyRedirect(urllib.request.HTTPRedirectHandler):
    """
    Revalida a política **a cada redirecionamento**.

    Sem isto a allowlist tem um buraco grande: um domínio liberado responde 302 para um
    domínio bloqueado, o ``urllib`` segue sozinho e o conteúdo chega como se tivesse vindo
    do lugar permitido. É a forma clássica de furar allowlist, e não aparece em teste
    nenhum que só busque URLs bem-comportadas.
    """

    def __init__(self, policy: Policy) -> None:
        self._policy = policy

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        decision = self._policy.check_url(newurl)
        if not decision.allowed:
            raise PolicyDenied(decision)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_url_tool(policy: Policy) -> tuple[ToolSpec, Handler]:
    spec = ToolSpec(
        name="fetch_url",
        description=(
            "Busca o conteúdo de uma URL http(s). Sujeito à política de domínios do "
            "harness — se o domínio não estiver liberado, a chamada é negada e o motivo "
            "é informado."
        ),
        input_schema={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "a URL a buscar"}},
            "required": ["url"],
            "additionalProperties": False,
        },
    )

    def handler(arguments):
        url = _require_str(arguments, "url")

        # Esquema **antes** da política, e a ordem é o ponto. ``file://`` transformaria
        # uma ferramenta de rede em leitura de disco irrestrita, contornando o eixo de
        # arquivo por inteiro — e não é uma questão de política: esta ferramenta não faz
        # isso, ponto. Com a ordem invertida a chamada até era barrada, mas pelo motivo
        # errado ("domínio fora da allowlist"), o que polui a métrica de negativas com um
        # caso que não é de domínio e esconde o problema real de quem lê o relatório.
        if not url.lower().startswith(("http://", "https://")):
            if "//" in url:
                raise ToolError(f"esquema não suportado: só http e https ({url})")
            url = "https://" + url

        _deny(policy.check_url(url))

        opener = urllib.request.build_opener(_PolicyRedirect(policy))
        request = urllib.request.Request(url, headers={"User-Agent": "harness-eng"})
        try:
            with opener.open(request, timeout=FETCH_TIMEOUT) as response:
                raw = response.read(MAX_FETCH_BYTES + 1)
                final = response.geturl()
        except PolicyDenied:
            raise
        except urllib.error.HTTPError as exc:
            raise ToolError(f"HTTP {exc.code} em {url}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise ToolError(f"falha ao buscar {url}: {exc}") from exc

        # Cinto e suspensório: mesmo com o handler acima, a URL final é conferida.
        _deny(policy.check_url(final))

        text = raw.decode("utf-8", errors="replace")
        if len(raw) > MAX_FETCH_BYTES:
            text = text[:MAX_FETCH_BYTES] + f"\n[truncado em {MAX_FETCH_BYTES} bytes]"
        return text

    return spec, handler


# ── comando ──────────────────────────────────────────────────────────────────────────

def run_command_tool(policy: Policy, cwd: Path) -> tuple[ToolSpec, Handler]:
    spec = ToolSpec(
        name="run_command",
        description=(
            "Roda um comando no workspace e devolve a saída. Sem shell: pipes, "
            "redirecionamentos e encadeamento com && não são interpretados. Sujeito à "
            "allowlist de executáveis do harness."
        ),
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string", "description": "o comando"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    )

    def handler(arguments):
        command = _require_str(arguments, "command")
        _deny(policy.check_command(command))

        try:
            argv = shlex.split(command)
        except ValueError as exc:
            raise ToolError(f"não consegui separar o comando: {exc}") from exc
        if not argv:
            raise ToolError("comando vazio")

        # ``shell=False``, sempre. A allowlist casa o primeiro token, e isso só significa
        # alguma coisa se o primeiro token for de fato o executável — com shell, um
        # ``pytest && curl ...`` passaria pela checagem e rodaria as duas coisas. Perder
        # pipe e redirecionamento é o preço de a checagem não ser teatro.
        try:
            completed = subprocess.run(  # noqa: S603 - argv sem shell, filtrado por allowlist
                argv,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ToolError(f"executável não encontrado: {argv[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ToolError(f"o comando passou de {COMMAND_TIMEOUT}s e foi encerrado") from exc

        partes = []
        if completed.stdout:
            partes.append(completed.stdout.rstrip())
        if completed.stderr:
            partes.append(f"[stderr]\n{completed.stderr.rstrip()}")
        if completed.returncode != 0:
            # Código de saída diferente de zero é informação para o modelo, não erro do
            # harness: o comando rodou, e o que ele disse é o resultado.
            partes.append(f"[código de saída {completed.returncode}]")
        return "\n".join(partes)

    return spec, handler


# ── o conjunto de cada nível ─────────────────────────────────────────────────────────

def policy_registry(policy: Policy, root: Path) -> ToolRegistry:
    """
    O conjunto de ferramentas que ``policy`` concede.

    Ordem estável, do mais inócuo ao mais poderoso. Não é estética: a lista entra no
    prefixo da requisição e qualquer variação nela invalida o cache de tudo que vem
    depois — e um nível que reordena ferramentas entre execuções derruba o acerto de
    cache sem nenhum sintoma além da conta no fim do mês.
    """
    registry = ToolRegistry()

    if policy.files is not FileAccess.NONE:
        for build in (read_file_tool, list_dir_tool, find_files_tool):
            spec, handler = build(root)
            registry.register(spec, handler)

    if policy.files is FileAccess.WRITE:
        registry.register(*write_file_tool(root, policy))

    if policy.network is not NetworkAccess.NONE:
        registry.register(*fetch_url_tool(policy))

    if policy.shell is not ShellAccess.NONE:
        registry.register(*run_command_tool(policy, root))

    return registry


__all__ = [
    "PolicyDenied",
    "policy_registry",
    "write_file_tool",
    "fetch_url_tool",
    "run_command_tool",
    "MAX_FETCH_BYTES",
    "FETCH_TIMEOUT",
    "COMMAND_TIMEOUT",
]
