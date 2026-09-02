"""
Níveis de harness: o que o agente **pode** fazer, e o registro do que foi negado.

Um harness não é uma coisa só. É um conjunto de capacidades concedidas — ler arquivo,
sair para a internet, rodar comando — e cada uma tem grau. "Rodei o agente" não descreve
um experimento; "rodei o agente no nível 2" descreve.

E é isto que fecha com o resto do pacote: **o nível é a variável independente**. A camada
estatística compara duas configurações pareando por tarefa, e a pergunta que mais aparece
na prática — *"preciso mesmo dar internet para este agente?"* — é literalmente uma
comparação de níveis.

A parte que muda a conversa é a segunda metade do módulo. Uma política que bloqueia em
silêncio não ensina nada: você fica sabendo que o agente falhou, não que ele **bateu numa
parede**. Aqui toda negativa é contada, com motivo e alvo, e vira métrica:

* **zero negativas no nível 4** significa que você concedeu risco por nada — o agente
  nunca precisou do que você deu;
* **muitas negativas no nível 1** significa que o nível não serve para a tarefa, e o
  relatório diz *qual* parede foi batida, então o próximo nível é escolha informada em
  vez de palpite.

Nenhum dos dois aparece num harness que só devolve "deu certo" ou "deu errado".

Deny by default em todos os eixos: uma política que erra para o lado permissivo erra
silenciosamente, e o custo do erro é assimétrico.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Mapping
from urllib.parse import urlparse


class FileAccess(str, Enum):
    """Acesso ao disco. ``READ`` e ``WRITE`` continuam presos à raiz do workspace."""

    NONE = "none"
    READ = "read"
    WRITE = "write"


class NetworkAccess(str, Enum):
    """
    Acesso à rede.

    ``ALLOWLIST`` é o grau que quase todo agente de verdade deveria usar e quase nenhum
    usa: a lista de domínios é o que separa "pesquisa na documentação" de "exfiltra o que
    leu para qualquer lugar", e a diferença não aparece em nenhuma métrica de qualidade.
    """

    NONE = "none"
    ALLOWLIST = "allowlist"
    FULL = "full"


class ShellAccess(str, Enum):
    """Execução de comando. ``ALLOWLIST`` casa pelo primeiro token (o executável)."""

    NONE = "none"
    ALLOWLIST = "allowlist"
    FULL = "full"


class Denial(str, Enum):
    """
    Por que uma ação foi negada.

    Enum e não string livre porque estes valores viram chave de contagem no relatório, e
    contador com chave digitada à mão vira três grafias do mesmo motivo em duas semanas.
    """

    NO_FILE_ACCESS = "sem acesso a arquivo"
    NO_WRITE_ACCESS = "sem permissão de escrita"
    OUTSIDE_WORKSPACE = "caminho fora do workspace"
    NO_NETWORK = "sem acesso à rede"
    DOMAIN_NOT_ALLOWED = "domínio fora da allowlist"
    DOMAIN_BLOCKED = "domínio bloqueado"
    NO_SHELL = "sem execução de comando"
    COMMAND_NOT_ALLOWED = "comando fora da allowlist"
    BUDGET_EXHAUSTED = "orçamento de tokens esgotado"


@dataclass(frozen=True, slots=True)
class Decision:
    """
    Resultado de uma checagem. ``reason`` só existe quando negou.

    Tipo próprio em vez de ``bool``: a métrica precisa do motivo, e uma função que devolve
    só ``False`` obriga quem chama a reconstruir o porquê — que é como o motivo acaba
    virando string improvisada no ponto de uso.
    """

    allowed: bool
    reason: Denial | None = None
    target: str = ""

    def __bool__(self) -> bool:
        return self.allowed

    @property
    def message(self) -> str:
        """O texto que o **modelo** lê. Diz o que foi negado e por quê, para ele se adaptar."""
        if self.allowed:
            return ""
        alvo = f" ({self.target})" if self.target else ""
        return f"negado pela política do harness: {self.reason.value}{alvo}"


ALLOWED = Decision(allowed=True)


def _host_of(url: str) -> str:
    """
    Extrai o host de uma URL, sem porta e em minúsculas.

    Aceita ``exemplo.com/x`` sem esquema porque é o que o modelo escreve metade das vezes,
    e recusar por falta de ``https://`` transformaria um detalhe de digitação numa negativa
    de política — que aí conta errado na métrica.
    """
    candidate = url.strip()
    if "//" not in candidate:
        candidate = "https://" + candidate
    return (urlparse(candidate).hostname or "").lower()


def _matches(host: str, domain: str) -> bool:
    """
    Casa host contra domínio, incluindo subdomínio.

    ``exemplo.com`` casa ``api.exemplo.com``, mas **não** casa ``naoexemplo.com`` — o
    ponto na comparação de sufixo é o que separa as duas, e esquecê-lo é o jeito clássico
    de uma allowlist deixar passar um domínio parecido.
    """
    domain = domain.lower().lstrip(".")
    return host == domain or host.endswith("." + domain)


@dataclass(frozen=True, slots=True)
class Policy:
    """
    Um nível de harness: os eixos, com grau em cada um.

    Imutável de propósito. Uma política que muda no meio da execução torna o trace
    impossível de interpretar — "o agente leu este arquivo" passa a depender de *quando*
    ele leu. :meth:`with_` devolve outra política em vez de alterar esta.
    """

    name: str = "custom"
    level: int = -1
    files: FileAccess = FileAccess.NONE
    network: NetworkAccess = NetworkAccess.NONE
    shell: ShellAccess = ShellAccess.NONE
    allowed_domains: frozenset[str] = field(default_factory=frozenset)
    blocked_domains: frozenset[str] = field(default_factory=frozenset)
    allowed_commands: frozenset[str] = field(default_factory=frozenset)
    max_iterations: int = 50
    #: Teto de tokens da execução inteira. ``None`` é sem teto — e sem teto é uma escolha,
    #: não um padrão distraído: um loop que não termina gasta até alguém perceber.
    token_budget: int | None = None

    # ── checagens ────────────────────────────────────────────────────────────────────
    def check_read(self) -> Decision:
        if self.files is FileAccess.NONE:
            return Decision(False, Denial.NO_FILE_ACCESS)
        return ALLOWED

    def check_write(self) -> Decision:
        if self.files is FileAccess.NONE:
            return Decision(False, Denial.NO_FILE_ACCESS)
        if self.files is not FileAccess.WRITE:
            return Decision(False, Denial.NO_WRITE_ACCESS)
        return ALLOWED

    def check_url(self, url: str) -> Decision:
        """
        Se esta URL pode ser buscada.

        Ordem deliberada: **bloqueio vence allowlist**. Quem escreve as duas listas
        costuma querer "tudo em ``exemplo.com``, menos ``interno.exemplo.com``", e a ordem
        inversa faria a exceção não valer nada.
        """
        if self.network is NetworkAccess.NONE:
            return Decision(False, Denial.NO_NETWORK, url)

        host = _host_of(url)
        if not host:
            return Decision(False, Denial.DOMAIN_NOT_ALLOWED, url)

        if any(_matches(host, blocked) for blocked in self.blocked_domains):
            return Decision(False, Denial.DOMAIN_BLOCKED, host)

        if self.network is NetworkAccess.FULL:
            return ALLOWED

        if any(_matches(host, allowed) for allowed in self.allowed_domains):
            return ALLOWED
        return Decision(False, Denial.DOMAIN_NOT_ALLOWED, host)

    def check_command(self, command: str) -> Decision:
        """
        Se este comando pode rodar. A allowlist casa o **executável**, o primeiro token.

        Não tenta interpretar a linha inteira: analisar shell direito é difícil, e uma
        checagem que *parece* entender ``&&``, aspas e substituição, mas não entende, é
        pior que uma que assume o próprio limite. Para superfície pequena, use
        ``ALLOWLIST`` com nomes de executável; ``FULL`` é opt-in explícito.
        """
        if self.shell is ShellAccess.NONE:
            return Decision(False, Denial.NO_SHELL, command)
        if self.shell is ShellAccess.FULL:
            return ALLOWED

        executable = command.strip().split()[0] if command.strip() else ""
        if executable in self.allowed_commands:
            return ALLOWED
        return Decision(False, Denial.COMMAND_NOT_ALLOWED, executable or command)

    def check_budget(self, spent: int) -> Decision:
        if self.token_budget is not None and spent >= self.token_budget:
            return Decision(False, Denial.BUDGET_EXHAUSTED, f"{spent:,}/{self.token_budget:,}")
        return ALLOWED

    # ── composição ───────────────────────────────────────────────────────────────────
    def with_(self, **changes: object) -> "Policy":
        """
        Uma variante desta política. ``name`` vira ``custom`` a menos que você dê outro.

        O renome é de propósito: um trace que diz ``researcher`` mas roda com a allowlist
        trocada mente sobre o experimento, e é justamente o tipo de mentira que só aparece
        quando alguém tenta reproduzir o resultado três meses depois.
        """
        if "name" not in changes:
            changes["name"] = "custom"
        if "level" not in changes:
            changes["level"] = -1
        for key in ("allowed_domains", "blocked_domains", "allowed_commands"):
            if key in changes and not isinstance(changes[key], frozenset):
                changes[key] = frozenset(changes[key])  # type: ignore[arg-type]
        return replace(self, **changes)  # type: ignore[arg-type]

    def allowing(self, *domains: str) -> "Policy":
        """Atalho para o caso comum: mesma política, mais domínios liberados."""
        return self.with_(allowed_domains=self.allowed_domains | frozenset(domains))

    def blocking(self, *domains: str) -> "Policy":
        return self.with_(blocked_domains=self.blocked_domains | frozenset(domains))

    def as_dict(self) -> dict:
        """
        A política como ela vai para ``Session.metadata``.

        Grava a política **inteira**, não só o número do nível. Nível sozinho é um rótulo
        que depende de uma tabela que pode mudar entre versões; o trace precisa dizer o
        que de fato estava concedido quando aquela execução aconteceu.
        """
        return {
            "name": self.name,
            "level": self.level,
            "files": self.files.value,
            "network": self.network.value,
            "shell": self.shell.value,
            "allowed_domains": sorted(self.allowed_domains),
            "blocked_domains": sorted(self.blocked_domains),
            "allowed_commands": sorted(self.allowed_commands),
            "max_iterations": self.max_iterations,
            "token_budget": self.token_budget,
        }

    def __str__(self) -> str:
        label = f"nível {self.level} ({self.name})" if self.level >= 0 else self.name
        return (
            f"{label}: arquivo={self.files.value} · rede={self.network.value} "
            f"· shell={self.shell.value}"
        )


# ── os níveis nomeados ───────────────────────────────────────────────────────────────

#: Nada. Só o modelo e as ferramentas que você registrar na mão.
#:
#: Não é um nível inútil: é a linha de base contra a qual todo o resto se compara. "Dar
#: acesso a arquivo melhorou?" só tem resposta se existir a execução sem acesso a arquivo.
SEALED = Policy(name="sealed", level=0, max_iterations=20)

#: Lê o workspace. O nível da maioria dos agentes de análise.
READER = Policy(
    name="reader",
    level=1,
    files=FileAccess.READ,
    max_iterations=30,
)

#: Lê o workspace e sai para a internet — com allowlist vazia por padrão.
#:
#: Allowlist vazia nega tudo, e é o padrão certo: quem sobe para este nível tem de dizer
#: **onde** o agente pode ir. ``RESEARCHER.allowing("docs.python.org")`` é a forma de uso.
RESEARCHER = Policy(
    name="researcher",
    level=2,
    files=FileAccess.READ,
    network=NetworkAccess.ALLOWLIST,
    max_iterations=40,
)

#: Escreve no workspace, além de tudo do nível anterior.
BUILDER = Policy(
    name="builder",
    level=3,
    files=FileAccess.WRITE,
    network=NetworkAccess.ALLOWLIST,
    max_iterations=50,
)

#: Roda comando, com allowlist. O topo, e o único que exige lista explícita de executáveis.
OPERATOR = Policy(
    name="operator",
    level=4,
    files=FileAccess.WRITE,
    network=NetworkAccess.ALLOWLIST,
    shell=ShellAccess.ALLOWLIST,
    max_iterations=60,
)

#: Os níveis por número. ``Policy`` continua construível na mão para o que não couber aqui.
LEVELS: Mapping[int, Policy] = {
    0: SEALED,
    1: READER,
    2: RESEARCHER,
    3: BUILDER,
    4: OPERATOR,
}


def level(number: int) -> Policy:
    """A política de um nível. Levanta com a lista de opções quando o número não existe."""
    if number not in LEVELS:
        disponiveis = ", ".join(f"{n} ({p.name})" for n, p in sorted(LEVELS.items()))
        raise ValueError(f"nível {number} não existe. Disponíveis: {disponiveis}")
    return LEVELS[number]


class DenialLog:
    """
    Conta o que a política negou, por motivo e por alvo.

    Segue o mesmo padrão de ``ClaudeCodeSource.skipped``, e pelo mesmo motivo: descartar
    em silêncio esconde o próprio defeito atrás de um número que parece bom. Aqui o número
    que pareceria bom é "o agente não fez nada de errado" — quando a verdade pode ser "o
    agente tentou doze vezes e a parede segurou".
    """

    def __init__(self) -> None:
        self.by_reason: dict[Denial, int] = {}
        self.by_target: dict[str, int] = {}
        self.total = 0

    def record(self, decision: Decision) -> None:
        if decision.allowed or decision.reason is None:
            return
        self.total += 1
        self.by_reason[decision.reason] = self.by_reason.get(decision.reason, 0) + 1
        if decision.target:
            self.by_target[decision.target] = self.by_target.get(decision.target, 0) + 1

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "by_reason": {reason.value: count for reason, count in self.by_reason.items()},
            "by_target": dict(sorted(self.by_target.items(), key=lambda kv: -kv[1])),
        }

    def __len__(self) -> int:
        return self.total


__all__ = [
    "Policy",
    "Decision",
    "Denial",
    "DenialLog",
    "FileAccess",
    "NetworkAccess",
    "ShellAccess",
    "LEVELS",
    "SEALED",
    "READER",
    "RESEARCHER",
    "BUILDER",
    "OPERATOR",
    "level",
]
