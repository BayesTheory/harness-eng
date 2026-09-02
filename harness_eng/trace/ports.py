"""
Portas da camada de trace.

``TraceSource`` é o contrato que torna o projeto agnóstico: as métricas falam com esta
porta e nunca com um formato concreto. Trocar Claude Code por OpenAI é uma linha no
composition root, não uma edição nas métricas — o mesmo padrão que sustentou o TennisIA.

``Protocol`` e não classe base: um adapter não precisa herdar nada, o que mantém possível
embrulhar leitor de terceiro sem envolvê-lo numa hierarquia.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from .model import Session


@runtime_checkable
class TraceSource(Protocol):
    """
    Lê traces de algum lugar e devolve :class:`Session` no formato canônico.

    :meth:`sessions` é iterador, não lista, por uma razão medida: os 54 transcripts
    analisados durante o planejamento somam 44.152 linhas, e a maior sessão sozinha tem
    1.669 turnos. Materializar tudo antes de contar uma média é desperdício que cresce
    com o histórico do usuário — e histórico de agente só cresce.
    """

    @property
    def name(self) -> str:
        """Identificador da origem, gravado em ``Session.source``."""
        ...

    def discover(self, root: Path) -> list[Path]:
        """Arquivos de trace sob ``root``. Vazio quando não há — nunca levanta."""
        ...

    def load(self, path: Path) -> Session | None:
        """
        Uma sessão de um arquivo. ``None`` quando o arquivo não é desta origem.

        ``None`` em vez de exceção porque varrer um diretório heterogêneo é o caso
        normal: o usuário aponta para uma pasta e o carregador tenta cada origem.
        """
        ...

    def sessions(self, root: Path) -> Iterator[Session]:
        """Todas as sessões sob ``root``, pulando o que não for legível."""
        ...


@runtime_checkable
class TraceSink(Protocol):
    """
    Escreve trace no formato canônico.

    Usada pelo harness deste repositório para emitir o próprio trace nativamente — é o
    que fecha o círculo: o harness é medido pelas mesmas ferramentas que medem os outros,
    sem adapter no meio.
    """

    def write(self, session: Session, target: Path) -> Path: ...
