"""
Trace nativo: o formato canônico gravado direto, sem tradução.

Todo outro adapter do pacote é uma tradução — lê o formato de um harness de terceiro e o
aproxima do canônico. Sempre resta a dúvida de quanto do número medido é o harness e
quanto é a tradução. Este par não tem essa dúvida: escreve os campos canônicos como eles
são e os lê de volta. Serve para duas coisas:

1. o harness deste repositório é medido pelas mesmas ferramentas que medem os outros;
2. o *round-trip* vira teste. Se um campo se perde ao ir para o disco e voltar, o formato
   canônico tem um buraco — e é melhor descobrir com um teste de 40 linhas do que com um
   relatório que reporta zero onde havia dado.

Layout: um JSONL por sessão. A primeira linha é o cabeçalho da sessão, cada linha
seguinte é um turno. Streaming de verdade — dá para escrever turno a turno enquanto a
execução acontece e ler sem carregar o arquivo inteiro, que é o que a porta
:class:`~harness_eng.trace.ports.TraceSource` promete.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..model import Role, Session, StopReason, ToolCall, ToolResult, Turn, Usage

#: Marca de formato na primeira linha. É o que permite ``load`` recusar um JSONL que não é
#: nosso sem adivinhar por extensão — ``~/.claude/projects`` também é cheio de ``.jsonl``.
FORMAT = "harness-eng/native/v1"


# ── escrita ──────────────────────────────────────────────────────────────────────────

def _usage_to_json(usage: Usage) -> dict:
    """
    Só os campos armazenados.

    Deliberadamente diferente de :meth:`Usage.as_dict`, que é a visão de *relatório* e
    inclui derivados (``context_size``, ``cache_hit_rate``) para quem lê. Persistir
    derivado é convidar o arquivo a discordar do código que o calcula — e ``as_dict`` não
    carrega ``service_tier``, então usá-lo aqui perderia um campo em silêncio.
    """
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "service_tier": usage.service_tier,
    }


def _call_to_json(call: ToolCall) -> dict:
    return {
        "id": call.id,
        "name": call.name,
        "arguments": dict(call.arguments),
        "turn_index": call.turn_index,
        "timestamp": call.timestamp.isoformat() if call.timestamp else None,
    }


def _result_to_json(result: ToolResult) -> dict:
    return {
        "call_id": result.call_id,
        "is_error": result.is_error,
        "content": result.content,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "interrupted": result.interrupted,
        "duration_s": result.duration.total_seconds() if result.duration else None,
        "content_kinds": list(result.content_kinds),
    }


def _turn_to_json(turn: Turn) -> dict:
    """
    Um turno como linha de JSONL.

    ``Turn.raw`` fica de fora, e é escolha, não esquecimento: nos outros adapters ele
    guarda o registro do provedor, e aqui guardaria os blocos que o loop usa para replay.
    Um trace é registro de **medição**, não checkpoint retomável — gravar o payload
    inteiro multiplicaria o arquivo por conteúdo que nenhuma métrica lê.
    """
    return {
        "type": "turn",
        "index": turn.index,
        "role": turn.role.value,
        "timestamp": turn.timestamp.isoformat() if turn.timestamp else None,
        "text": turn.text,
        "thinking": turn.thinking,
        "tool_calls": [_call_to_json(c) for c in turn.tool_calls],
        "tool_results": [_result_to_json(r) for r in turn.tool_results],
        "usage": _usage_to_json(turn.usage) if turn.usage else None,
        "model": turn.model,
        "stop_reason": turn.stop_reason.value,
        "is_sidechain": turn.is_sidechain,
        "agent_id": turn.agent_id,
    }


class NativeSink:
    """
    Grava uma :class:`Session` canônica. Implementa :class:`~harness_eng.trace.ports.TraceSink`.
    """

    def write(self, session: Session, target: Path) -> Path:
        """
        Escreve a sessão em ``target``, criando o diretório se preciso.

        ``newline="\\n"`` explícito: no Windows o padrão do Python vira ``\\r\\n``, e um
        JSONL com terminador de linha diferente por sistema operacional é uma diferença
        que aparece em ``git diff`` de todo mundo e em hash de nenhum arquivo igual.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "type": "session",
            "format": FORMAT,
            "id": session.id,
            "source": session.source,
            "cwd": session.cwd,
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
            "metadata": dict(session.metadata),
        }
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(header, ensure_ascii=False) + "\n")
            for turn in session.turns:
                line = json.dumps(_turn_to_json(turn), ensure_ascii=False, default=str)
                handle.write(line + "\n")
        return target


# ── leitura ──────────────────────────────────────────────────────────────────────────

def _parse_timestamp(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _usage_from_json(raw: Any) -> Usage | None:
    """``None`` continua ``None``: ausência de medição não vira ``Usage(0, 0, 0, 0)``."""
    if not isinstance(raw, Mapping):
        return None
    return Usage(
        input_tokens=int(raw.get("input_tokens", 0)),
        output_tokens=int(raw.get("output_tokens", 0)),
        cache_write_tokens=int(raw.get("cache_write_tokens", 0)),
        cache_read_tokens=int(raw.get("cache_read_tokens", 0)),
        service_tier=raw.get("service_tier"),
    )


def _duration_from_json(raw: Any) -> timedelta | None:
    return timedelta(seconds=raw) if isinstance(raw, (int, float)) else None


def _turn_from_json(raw: Mapping[str, Any]) -> Turn:
    return Turn(
        index=int(raw.get("index", -1)),
        role=Role(raw.get("role", "user")),
        timestamp=_parse_timestamp(raw.get("timestamp")),
        text=raw.get("text") or "",
        thinking=raw.get("thinking") or "",
        tool_calls=tuple(
            ToolCall(
                id=c.get("id", ""),
                name=c.get("name", ""),
                arguments=c.get("arguments") or {},
                turn_index=int(c.get("turn_index", -1)),
                timestamp=_parse_timestamp(c.get("timestamp")),
            )
            for c in raw.get("tool_calls") or []
        ),
        tool_results=tuple(
            ToolResult(
                call_id=r.get("call_id", ""),
                is_error=bool(r.get("is_error", False)),
                content=r.get("content") or "",
                stdout=r.get("stdout"),
                stderr=r.get("stderr"),
                interrupted=bool(r.get("interrupted", False)),
                duration=_duration_from_json(r.get("duration_s")),
                content_kinds=tuple(r.get("content_kinds") or ()),
            )
            for r in raw.get("tool_results") or []
        ),
        usage=_usage_from_json(raw.get("usage")),
        model=raw.get("model"),
        stop_reason=StopReason.parse(raw.get("stop_reason")),
        is_sidechain=bool(raw.get("is_sidechain", False)),
        agent_id=raw.get("agent_id"),
    )


class NativeSource:
    """
    Lê trace nativo. Implementa :class:`~harness_eng.trace.ports.TraceSource`.

    Tolerante do mesmo jeito que o adapter do Claude Code, e pelo mesmo motivo: uma
    sessão morta no meio deixa a última linha truncada, e recusar o arquivo inteiro por
    causa dela joga fora todo o resto do dado — que é bom.
    """

    name = "native"

    def __init__(self) -> None:
        #: O que foi pulado, por motivo. Um leitor que descarta em silêncio esconde o
        #: próprio defeito atrás de um número que parece bom.
        self.skipped: Counter[str] = Counter()

    def discover(self, root: Path) -> list[Path]:
        if not root.exists():
            return []
        if root.is_file():
            return [root]
        return sorted(root.rglob("*.jsonl"))

    def load(self, path: Path) -> Session | None:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            self.skipped["arquivo ilegível"] += 1
            return None
        if not lines:
            self.skipped["arquivo vazio"] += 1
            return None

        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError:
            self.skipped["cabeçalho inválido"] += 1
            return None
        # Recusa por marca de formato, não por extensão: ``~/.claude/projects`` está cheio
        # de ``.jsonl`` que não são nossos, e adivinhar pela extensão faria este leitor
        # engolir transcript alheio e devolver sessão vazia em vez de ``None``.
        if not isinstance(header, dict) or header.get("format") != FORMAT:
            self.skipped["formato de outra origem"] += 1
            return None

        turns: list[Turn] = []
        for line in lines[1:]:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                self.skipped["linha corrompida"] += 1
                continue
            if not isinstance(record, dict) or record.get("type") != "turn":
                self.skipped["registro que não é turno"] += 1
                continue
            try:
                turns.append(_turn_from_json(record))
            except (ValueError, TypeError):
                self.skipped["turno ilegível"] += 1

        return Session(
            id=header.get("id") or path.stem,
            source=header.get("source") or self.name,
            turns=tuple(turns),
            cwd=header.get("cwd"),
            started_at=_parse_timestamp(header.get("started_at")),
            ended_at=_parse_timestamp(header.get("ended_at")),
            metadata=header.get("metadata") or {},
        )

    def sessions(self, root: Path) -> Iterator[Session]:
        for path in self.discover(root):
            session = self.load(path)
            if session is not None:
                yield session
