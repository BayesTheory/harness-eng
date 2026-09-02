"""
Adapter do Claude Code: JSONL de ``~/.claude/projects/`` → formato canônico.

O schema abaixo foi extraído **empiricamente** de 54 transcripts reais (44.152 linhas),
não das docs nem por suposição. Isso importa: a lição da sessão que originou este
repositório foi que assinatura adivinhada passa em teste e quebra em execução.

Layout do disco::

    ~/.claude/projects/
      c--Users-Rian-Desktop-projeto/     nome do cwd com separadores trocados
        <uuid-da-sessao>.jsonl

Um registro por linha. Tipos observados, com contagem nos 54 transcripts::

    assistant 16708   user 9352      attachment 5651   ai-title 2584
    last-prompt 2561  queue-operation 2313   atis-latch 1280   mode 1234
    bridge-session 781   file-history-snapshot 715   file-history-delta 711
    frame-link 124    history-suppression 64   system 39   artifact-* 35

Só ``assistant``, ``user`` e ``system`` viram turno. O resto é estado de UI, histórico de
arquivo e telemetria do cliente — descartar é correto, e descartar *em silêncio* seria
errado, então :attr:`ClaudeCodeSource.skipped` conta o que foi ignorado e por quê.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..model import Role, Session, StopReason, ToolCall, ToolResult, Turn, Usage

#: Tipos de registro que viram turno. Os demais são estado de cliente.
_TURN_TYPES = frozenset({"assistant", "user", "system"})

#: Blocos de conteúdo que carregam informação medível.
_TEXT_BLOCKS = frozenset({"text"})
_THINKING_BLOCKS = frozenset({"thinking", "redacted_thinking"})


class ClaudeCodeSource:
    """
    Lê transcripts do Claude Code. Implementa :class:`~harness_eng.trace.ports.TraceSource`.

    Tolerante por projeto: linha corrompida, JSON inválido e registro de tipo desconhecido
    são contados e pulados, nunca fatais. Um transcript de 1.669 turnos com uma linha
    truncada no fim — que acontece quando a sessão é morta — continua sendo 1.668 turnos
    de dado bom, e recusá-lo inteiro seria perder medição por preciosismo.
    """

    name = "claude_code"

    def __init__(self, include_sidechains: bool = True) -> None:
        self._include_sidechains = include_sidechains
        #: Contagem do que foi pulado, por motivo. Sai no relatório: um adapter que
        #: descarta 30% das linhas está errado, e sem este contador ninguém descobre.
        self.skipped: Counter[str] = Counter()

    # ── descoberta ────────────────────────────────────────────────────────────
    def discover(self, root: Path) -> list[Path]:
        root = Path(root)
        if root.is_file():
            return [root] if root.suffix == ".jsonl" else []
        if not root.is_dir():
            return []
        return sorted(root.rglob("*.jsonl"))

    def sessions(self, root: Path) -> Iterator[Session]:
        for path in self.discover(root):
            session = self.load(path)
            if session is not None and session.turns:
                yield session

    # ── carga ─────────────────────────────────────────────────────────────────
    def load(self, path: Path) -> Session | None:
        path = Path(path)
        try:
            raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            self.skipped["arquivo ilegível"] += 1
            return None

        turns: list[Turn] = []
        session_id = path.stem
        cwd: str | None = None
        metadata: dict[str, Any] = {}
        timestamps: list[datetime] = []
        index = 0

        for line in raw_lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Linha truncada: acontece quando a sessão é morta no meio da escrita.
                self.skipped["json inválido"] += 1
                continue
            if not isinstance(record, dict):
                self.skipped["registro não-objeto"] += 1
                continue

            record_type = record.get("type", "")
            if record_type not in _TURN_TYPES:
                self.skipped[f"tipo {record_type or '?'}"] += 1
                continue

            if record.get("isSidechain") and not self._include_sidechains:
                self.skipped["sidechain"] += 1
                continue

            turn = self._to_turn(record, index)
            if turn is None:
                self.skipped["sem mensagem"] += 1
                continue

            turns.append(turn)
            index += 1
            if turn.timestamp:
                timestamps.append(turn.timestamp)

            session_id = record.get("sessionId") or session_id
            cwd = cwd or record.get("cwd")
            for key in ("version", "gitBranch", "entrypoint"):
                if key not in metadata and record.get(key):
                    metadata[key] = record[key]

        if not turns:
            return None

        metadata["path"] = str(path)
        metadata["skipped"] = dict(self.skipped)
        return Session(
            id=session_id,
            source=self.name,
            turns=tuple(turns),
            cwd=cwd,
            started_at=min(timestamps) if timestamps else None,
            ended_at=max(timestamps) if timestamps else None,
            metadata=metadata,
        )

    # ── conversão ─────────────────────────────────────────────────────────────
    def _to_turn(self, record: Mapping[str, Any], index: int) -> Turn | None:
        message = record.get("message")
        if not isinstance(message, dict):
            return None

        role = _parse_role(message.get("role") or record.get("type"))
        blocks = message.get("content")
        text, thinking, calls, results = self._parse_content(blocks, record, index)

        return Turn(
            index=index,
            role=role,
            timestamp=_parse_timestamp(record.get("timestamp")),
            text=text,
            thinking=thinking,
            tool_calls=tuple(calls),
            tool_results=tuple(results),
            usage=_parse_usage(message.get("usage")),
            model=message.get("model"),
            stop_reason=StopReason.parse(message.get("stop_reason")),
            is_sidechain=bool(record.get("isSidechain")),
            agent_id=record.get("agentId"),
            raw={},  # o registro cru não é guardado: 44 mil deles não cabem em memória
        )

    def _parse_content(
        self, blocks: Any, record: Mapping[str, Any], index: int
    ) -> tuple[str, str, list[ToolCall], list[ToolResult]]:
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        calls: list[ToolCall] = []
        results: list[ToolResult] = []

        # Conteúdo pode ser string simples (230 ocorrências nos transcripts) ou lista de
        # blocos. Tratar só a lista perderia esses turnos inteiros.
        if isinstance(blocks, str):
            return blocks, "", [], []
        if not isinstance(blocks, list):
            return "", "", [], []

        timestamp = _parse_timestamp(record.get("timestamp"))
        tool_use_result = record.get("toolUseResult")

        for block in blocks:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")

            if kind in _TEXT_BLOCKS:
                text_parts.append(str(block.get("text", "")))

            elif kind in _THINKING_BLOCKS:
                thinking_parts.append(str(block.get("thinking", "")))

            elif kind == "tool_use":
                calls.append(
                    ToolCall(
                        id=str(block.get("id", "")),
                        name=str(block.get("name", "?")),
                        arguments=block.get("input") or {},
                        turn_index=index,
                        timestamp=timestamp,
                    )
                )

            elif kind == "tool_result":
                results.append(_parse_tool_result(block, tool_use_result))

        return "\n".join(text_parts), "\n".join(thinking_parts), calls, results


def _parse_tool_result(block: Mapping[str, Any], extra: Any) -> ToolResult:
    """
    Constrói o resultado a partir do bloco e do ``toolUseResult`` de topo.

    O Claude Code guarda o resultado em dois lugares: o bloco ``tool_result`` dentro da
    mensagem (com ``is_error``) e um ``toolUseResult`` no topo do registro (com
    ``stdout``, ``stderr``, ``interrupted``). Os dois descrevem a mesma chamada e nenhum
    é completo sozinho — ler só um perde ou o sinal de erro ou a saída.
    """
    stdout = stderr = None
    interrupted = False
    if isinstance(extra, dict):
        stdout = _as_text(extra.get("stdout"))
        stderr = _as_text(extra.get("stderr"))
        interrupted = bool(extra.get("interrupted"))

    content = block.get("content")
    return ToolResult(
        call_id=str(block.get("tool_use_id", "")),
        is_error=bool(block.get("is_error", False)),
        content=_as_text(content) or "",
        stdout=stdout,
        stderr=stderr,
        interrupted=interrupted,
        content_kinds=_content_kinds(content),
    )


def _content_kinds(value: Any) -> tuple[str, ...]:
    """
    Tipos de bloco presentes no resultado, distintos e ordenados.

    Sem isto, um resultado composto só de blocos ``image`` ou ``tool_reference`` vira
    string vazia e a métrica o conta como falha silenciosa. Guardar os tipos custa nada
    e é a diferença entre "a ferramenta não devolveu nada" e "eu não sei ler o que ela
    devolveu" — que são conclusões opostas.
    """
    if isinstance(value, str):
        return ("text",) if value else ()
    if isinstance(value, list):
        return tuple(sorted({
            str(item.get("type", "?")) for item in value if isinstance(item, dict)
        }))
    return ("unknown",) if value is not None else ()


def _as_text(value: Any) -> str | None:
    """
    Achata conteúdo que pode vir como string, lista de blocos ou objeto.

    O campo ``content`` de um ``tool_result`` é string na maioria dos casos e lista de
    blocos ``{"type": "text", "text": ...}`` quando a ferramenta devolve imagem junto.
    Assumir string dá ``TypeError`` justamente nos turnos mais interessantes.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [
            str(item.get("text", ""))
            for item in value
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return str(value)


def _parse_role(raw: Any) -> Role:
    try:
        return Role(str(raw))
    except ValueError:
        return Role.USER


def _parse_timestamp(raw: Any) -> datetime | None:
    """
    ISO 8601, com o ``Z`` que o ``fromisoformat`` do Python < 3.11 não aceita.

    Devolve ``None`` em vez de levantar: um turno sem timestamp legível ainda conta para
    taxa de erro e uso de token, e recusá-lo perderia medição por causa de um campo que
    nem toda métrica usa.
    """
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_usage(raw: Any) -> Usage | None:
    """
    ``None`` quando não há uso registrado — turno de user não consome token.

    Devolver ``Usage()`` zerado aqui poluiria toda média de contexto com zeros de turnos
    que nunca chamaram o modelo. Ausente é ausente.
    """
    if not isinstance(raw, dict):
        return None
    return Usage(
        input_tokens=int(raw.get("input_tokens") or 0),
        output_tokens=int(raw.get("output_tokens") or 0),
        cache_write_tokens=int(raw.get("cache_creation_input_tokens") or 0),
        cache_read_tokens=int(raw.get("cache_read_input_tokens") or 0),
        service_tier=raw.get("service_tier"),
    )


def default_root() -> Path:
    """Onde o Claude Code guarda transcripts nesta máquina."""
    return Path.home() / ".claude" / "projects"
