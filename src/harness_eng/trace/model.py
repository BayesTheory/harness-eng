"""
O formato canônico de trace: o vocabulário comum a todo harness.

Este é o núcleo do projeto. Cada harness registra execução no seu próprio formato — o
Claude Code escreve JSONL com uma árvore de `parentUuid`, o Agents SDK da OpenAI escreve
outra coisa, e um harness caseiro escreve o que o autor inventou. Medir os três exige um
vocabulário comum, e é este arquivo.

Nada aqui importa ``anthropic``, ``openai``, ``httpx`` ou toca disco. Um :class:`Session`
construído à mão num teste é indistinguível de um lido de transcript real, e é isso que
permite testar toda a camada de métricas sem transcript, sem chave de API e sem rede.

Princípio que atravessa o módulo: **ausência é ausência**. Um turno sem uso de token tem
``usage=None``, não ``Usage(0,0,0,0)``. Zero é uma medição; ausente não é. Confundir os
dois é como uma média de velocidade de tacada afunda ao contar bolas que ninguém mediu.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Iterator, Mapping, Sequence


class Role(str, Enum):
    """Quem produziu o turno."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class StopReason(str, Enum):
    """
    Por que o modelo parou de gerar.

    Importa para medição: ``TOOL_USE`` é o caso normal de um agente em loop,
    ``MAX_TOKENS`` é truncamento — resposta cortada no meio, quase sempre um sintoma de
    contexto mal administrado, e o tipo de coisa que se quer contar.
    """

    TOOL_USE = "tool_use"
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    REFUSAL = "refusal"
    ERROR = "error"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, raw: str | None) -> "StopReason":
        if raw is None:
            return cls.UNKNOWN
        try:
            return cls(str(raw))
        except ValueError:
            return cls.UNKNOWN


@dataclass(frozen=True, slots=True)
class Usage:
    """
    Consumo de token de uma chamada ao modelo.

    A separação entre escrita e leitura de cache não é detalhe de cobrança: é a métrica
    de eficiência mais reveladora de um harness. Escrever cache custa mais que input
    normal, ler custa uma fração. Um harness que reescreve o prefixo a cada turno paga
    caro e nem sabe; um que mantém o prefixo estável lê barato.

    Nos 54 transcripts analisados a taxa de acerto ficou em 96,5% — 5,13 bilhões de
    tokens lidos de cache contra 184 milhões escritos.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    service_tier: str | None = None

    @property
    def context_size(self) -> int:
        """
        Quanto o modelo realmente leu neste turno.

        É a soma de input novo, cache lido e cache escrito — porque cache escrito também
        foi lido, só que pela primeira vez. Usar só ``input_tokens`` para estimar tamanho
        de contexto subestima em ordens de grandeza quando há cache, que é o caso normal.
        """
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def cache_hit_rate(self) -> float | None:
        """
        Fração do contexto que veio de cache. ``None`` quando não houve contexto algum.

        ``None`` e não 0.0: um turno sem leitura nenhuma não teve 0% de acerto, não teve
        acerto definível.
        """
        total = self.context_size
        return self.cache_read_tokens / total if total else None

    @property
    def total_tokens(self) -> int:
        return self.context_size + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        """Soma para agregar sessão. ``service_tier`` some: não é somável."""
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            service_tier=self.service_tier if self.service_tier == other.service_tier else None,
        )

    def as_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "context_size": self.context_size,
            "cache_hit_rate": round(self.cache_hit_rate, 4) if self.cache_hit_rate is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    """
    Uma invocação de ferramenta pedida pelo modelo.

    ``arguments`` fica como mapa cru, sem interpretação: o formato varia por ferramenta e
    por harness, e normalizar aqui perderia justamente o que as métricas precisam
    comparar. :meth:`signature` existe para o detector de loop, que precisa responder
    "esta chamada é a mesma daquela?" sem depender de qual ferramenta é.
    """

    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    turn_index: int = -1
    timestamp: datetime | None = None

    def signature(self, max_length: int = 200) -> str:
        """
        Assinatura estável para detecção de repetição.

        Argumentos ordenados por chave — dois dicts iguais com ordem de inserção
        diferente têm de dar a mesma assinatura, senão o detector de loop perde
        repetições reais por um detalhe de serialização.
        """
        import json

        try:
            payload = json.dumps(self.arguments, sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            payload = repr(sorted(self.arguments.items(), key=lambda kv: kv[0]))
        return f"{self.name}:{payload[:max_length]}"

    @property
    def primary_argument(self) -> str | None:
        """
        O argumento que identifica a chamada para um humano lendo o relatório.

        Heurística por convenção de nome, na ordem em que as ferramentas reais usam.
        Não é semântica — é para o relatório dizer ``Bash: pytest tests/`` em vez de
        despejar o dicionário inteiro numa tabela.
        """
        for key in ("command", "file_path", "path", "pattern", "query", "url", "prompt"):
            value = self.arguments.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None


@dataclass(frozen=True, slots=True)
class ToolResult:
    """
    O que voltou de uma :class:`ToolCall`.

    ``is_error`` é a métrica mais barata e mais reveladora do repositório: nos transcripts
    analisados o PowerShell errou 14,3% das chamadas contra 3,0% do Bash — cinco vezes
    mais, num harness onde as duas ferramentas fazem o mesmo trabalho.
    """

    call_id: str
    is_error: bool = False
    content: str = ""
    stdout: str | None = None
    stderr: str | None = None
    interrupted: bool = False
    duration: timedelta | None = None
    #: Tipos de bloco que o resultado carregava (``text``, ``image``, ``tool_reference``...).
    #:
    #: Existe porque nem todo conteúdo é texto, e achatar tudo para string faz um
    #: resultado rico parecer vazio. Foi um bug real deste adapter: ``ToolSearch``
    #: devolve blocos ``tool_reference`` e o extrator de texto os ignorava, produzindo
    #: "58 de 60 resultados vazios" — que seria um achado alarmante sobre o harness se
    #: não fosse um defeito da ferramenta de medição.
    content_kinds: tuple[str, ...] = ()

    @property
    def has_content(self) -> bool:
        """Se voltou alguma coisa — texto, saída de processo ou bloco não-textual."""
        return bool(self.content or self.stdout or self.stderr or self.content_kinds)

    @property
    def is_empty(self) -> bool:
        """
        Sucesso sem saída **nenhuma**, de tipo nenhum.

        Vale contar separado de erro: uma ferramenta que "funciona" e não devolve nada
        deixa o modelo sem sinal para o próximo passo, e é um jeito silencioso de o loop
        travar sem nunca registrar erro.

        Conteúdo não-textual NÃO conta como vazio — ver :attr:`content_kinds`.
        """
        return not self.is_error and not self.has_content


@dataclass(frozen=True, slots=True)
class Turn:
    """
    Um turno: uma mensagem, com o que ela pediu e o que consumiu.

    Não é "uma requisição HTTP" nem "uma mensagem da API" — é a unidade que as métricas
    contam. Um turno de assistant pode pedir várias ferramentas de uma vez, e um turno de
    user pode carregar vários resultados; a estrutura reflete isso em vez de achatar.
    """

    index: int
    role: Role
    timestamp: datetime | None = None
    text: str = ""
    thinking: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    usage: Usage | None = None
    model: str | None = None
    stop_reason: StopReason = StopReason.UNKNOWN
    is_sidechain: bool = False
    agent_id: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_assistant(self) -> bool:
        return self.role is Role.ASSISTANT

    @property
    def calls_tools(self) -> bool:
        return bool(self.tool_calls)

    @property
    def was_truncated(self) -> bool:
        """Resposta cortada por limite de token — sintoma de contexto mal administrado."""
        return self.stop_reason is StopReason.MAX_TOKENS

    @property
    def context_size(self) -> int:
        return self.usage.context_size if self.usage else 0


@dataclass(frozen=True, slots=True)
class Session:
    """
    Uma execução completa de agente. Raiz de agregado do formato canônico.

    ``source`` diz de qual harness veio (``claude_code``, ``openai``, ``native``) porque
    a comparação entre harnesses é o ponto do projeto, e um número sem procedência não
    serve para comparar nada.
    """

    id: str
    source: str
    turns: tuple[Turn, ...] = ()
    cwd: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Turn]:
        return iter(self.turns)

    def __len__(self) -> int:
        return len(self.turns)

    # ── recortes ──────────────────────────────────────────────────────────────
    @property
    def assistant_turns(self) -> tuple[Turn, ...]:
        return tuple(t for t in self.turns if t.is_assistant)

    def tool_calls(self, include_sidechains: bool = True) -> list[ToolCall]:
        """
        Todas as chamadas de ferramenta, em ordem.

        ``include_sidechains=False`` exclui o trabalho de subagente. Importa para
        comparação justa: um harness que delega a subagentes tem contagem de chamada
        inflada em relação a um que faz tudo na linha principal, e comparar os dois sem
        separar isso mede a arquitetura, não a eficiência.
        """
        return [
            call
            for turn in self.turns
            if include_sidechains or not turn.is_sidechain
            for call in turn.tool_calls
        ]

    def results_by_call_id(self) -> dict[str, ToolResult]:
        """
        Índice de resultado por id de chamada.

        É o casamento que torna a taxa de erro por ferramenta calculável: o erro chega
        num turno de user, e o nome da ferramenta está no turno de assistant anterior.
        Sem este índice, todo consumidor refaz o pareamento — e um deles erra.
        """
        return {
            result.call_id: result
            for turn in self.turns
            for result in turn.tool_results
        }

    def paired_calls(self) -> list[tuple[ToolCall, ToolResult | None]]:
        """
        Cada chamada com seu resultado, ou ``None`` quando não houve.

        Chamada sem resultado é um estado real: sessão interrompida no meio, ou o
        harness abandonando a chamada. Não pode virar sucesso silencioso.
        """
        results = self.results_by_call_id()
        return [(call, results.get(call.id)) for call in self.tool_calls()]

    # ── agregados ─────────────────────────────────────────────────────────────
    @property
    def total_usage(self) -> Usage:
        total = Usage()
        for turn in self.turns:
            if turn.usage is not None:
                total = total + turn.usage
        return total

    @property
    def duration(self) -> timedelta | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return self.ended_at - self.started_at

    @property
    def models_used(self) -> tuple[str, ...]:
        """Modelos distintos usados. Uma sessão pode trocar de modelo no meio."""
        return tuple(sorted({t.model for t in self.turns if t.model}))

    def as_dict(self) -> dict:
        usage = self.total_usage
        return {
            "id": self.id,
            "source": self.source,
            "turns": len(self.turns),
            "assistant_turns": len(self.assistant_turns),
            "tool_calls": len(self.tool_calls()),
            "models": list(self.models_used),
            "duration_s": round(self.duration.total_seconds(), 1) if self.duration else None,
            "usage": usage.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class TraceSet:
    """
    Um conjunto de sessões — o que as métricas de fato consomem.

    Existe como tipo próprio em vez de ``list[Session]`` porque quase toda pergunta útil
    é sobre o conjunto ("qual ferramenta erra mais", "quanto custou o mês"), e porque
    filtrar por origem é a operação que separa a comparação entre harnesses de uma
    média sem sentido sobre tudo junto.
    """

    sessions: tuple[Session, ...] = ()

    @classmethod
    def of(cls, sessions: Sequence[Session]) -> "TraceSet":
        return cls(tuple(sessions))

    def __iter__(self) -> Iterator[Session]:
        return iter(self.sessions)

    def __len__(self) -> int:
        return len(self.sessions)

    def from_source(self, source: str) -> "TraceSet":
        return TraceSet(tuple(s for s in self.sessions if s.source == source))

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted({s.source for s in self.sessions}))

    def all_calls(self) -> list[ToolCall]:
        return [call for session in self.sessions for call in session.tool_calls()]

    def all_paired_calls(self) -> list[tuple[ToolCall, ToolResult | None]]:
        return [pair for session in self.sessions for pair in session.paired_calls()]

    @property
    def total_usage(self) -> Usage:
        total = Usage()
        for session in self.sessions:
            total = total + session.total_usage
        return total
