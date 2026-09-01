"""
Custo: quanto o harness gastou, e onde.

Custo é a métrica que fecha o argumento. "Este harness usa menos contexto" convence um
engenheiro; "este harness custa 40% menos por tarefa concluída" convence quem paga.

Duas decisões de honestidade que atravessam o módulo:

* **Modelo desconhecido não vira estimativa.** Se o preço de um modelo não está na tabela,
  o custo dele é ``None`` e o relatório diz quantos tokens ficaram sem preço. Inventar um
  preço "próximo" produz um total que parece exato e está errado — exatamente o padrão que
  este repositório existe para detectar.
* **Custo por tarefa concluída, não por requisição.** Uma requisição mais barata que
  precisa de três tentativas não é mais barata. :func:`cost_per_session` é a unidade que
  a camada estatística compara.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..trace.model import Session, TraceSet, Usage


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """
    Preço de um modelo, em dólares por milhão de tokens.

    ``cache_write_multiplier`` e ``cache_read_multiplier`` são razões sobre o preço de
    input em vez de valores absolutos porque é assim que a cobrança de cache é definida —
    e assim um ajuste de preço de input propaga sozinho, sem deixar as três linhas
    inconsistentes entre si.
    """

    input_per_mtok: float
    output_per_mtok: float
    cache_write_multiplier: float = 1.25
    cache_read_multiplier: float = 0.1
    context_window: int | None = None
    verified: bool = True

    @property
    def cache_write_per_mtok(self) -> float:
        return self.input_per_mtok * self.cache_write_multiplier

    @property
    def cache_read_per_mtok(self) -> float:
        return self.input_per_mtok * self.cache_read_multiplier

    def cost_of(self, usage: Usage) -> float:
        """Custo em dólares de um :class:`Usage`."""
        million = 1_000_000
        return (
            usage.input_tokens / million * self.input_per_mtok
            + usage.output_tokens / million * self.output_per_mtok
            + usage.cache_write_tokens / million * self.cache_write_per_mtok
            + usage.cache_read_tokens / million * self.cache_read_per_mtok
        )


#: Tabela de preços da API Anthropic de primeira parte, em USD por milhão de tokens.
#:
#: Preços de input e output conferidos contra a referência oficial do projeto (cache de
#: 2026-06-24). Bedrock e Vertex são operados por parceiro e têm preço próprio — não
#: estão aqui, e um trace vindo de lá sai sem custo em vez de sair com o preço errado.
#:
#: Os multiplicadores de cache (1,25x escrita, 0,1x leitura) são as razões publicadas
#: padrão e NÃO foram confirmadas na mesma referência; ``verified=False`` marca isso onde
#: o modelo depende fortemente deles. Corrija com ``--pricing`` se a sua fatura discordar.
DEFAULT_PRICING: dict[str, ModelPricing] = {
    "claude-fable-5": ModelPricing(10.00, 50.00, context_window=1_000_000),
    "claude-mythos-5": ModelPricing(10.00, 50.00, context_window=1_000_000),
    "claude-opus-5": ModelPricing(5.00, 25.00, context_window=1_000_000),
    "claude-opus-4-8": ModelPricing(5.00, 25.00, context_window=1_000_000),
    "claude-opus-4-7": ModelPricing(5.00, 25.00, context_window=1_000_000),
    "claude-opus-4-6": ModelPricing(5.00, 25.00, context_window=1_000_000),
    "claude-sonnet-5": ModelPricing(2.00, 10.00, context_window=1_000_000),
    "claude-sonnet-4-6": ModelPricing(3.00, 15.00, context_window=1_000_000),
    "claude-haiku-4-5": ModelPricing(1.00, 5.00, context_window=200_000),
}


@dataclass(frozen=True, slots=True)
class CostReport:
    """Custo estimado, com o que ficou de fora explicitado."""

    by_model: Mapping[str, float] = field(default_factory=dict)
    tokens_by_model: Mapping[str, Usage] = field(default_factory=dict)
    unpriced_models: tuple[str, ...] = ()
    unpriced_tokens: int = 0

    @property
    def total(self) -> float:
        return sum(self.by_model.values())

    @property
    def is_complete(self) -> bool:
        """
        Se todo token teve preço.

        Um total incompleto continua útil como piso, mas quem lê precisa saber que é
        piso. É a mesma regra da auditoria de rally do TennisIA: limite inferior é uma
        resposta válida, limite inferior disfarçado de total não é.
        """
        return not self.unpriced_models

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.by_model.items(), key=lambda kv: -kv[1])

    def as_dict(self) -> dict:
        return {
            "total_usd": round(self.total, 2),
            "complete": self.is_complete,
            "by_model": {m: round(c, 4) for m, c in self.ranked()},
            "unpriced_models": list(self.unpriced_models),
            "unpriced_tokens": self.unpriced_tokens,
        }


def estimate_cost(
    traces: TraceSet | Sequence[Session],
    pricing: Mapping[str, ModelPricing] | None = None,
) -> CostReport:
    """
    Custo estimado de um conjunto de traces, agrupado por modelo.

    Agrupa por modelo antes de multiplicar porque uma sessão pode trocar de modelo no
    meio — subagente barato, fallback, mudança manual — e aplicar um preço só à sessão
    inteira erraria nos dois sentidos.
    """
    table = dict(DEFAULT_PRICING if pricing is None else pricing)
    sessions = traces.sessions if isinstance(traces, TraceSet) else tuple(traces)

    usage_by_model: dict[str, Usage] = {}
    for session in sessions:
        for turn in session:
            if turn.usage is None:
                continue
            model = turn.model or "<desconhecido>"
            usage_by_model[model] = usage_by_model.get(model, Usage()) + turn.usage

    costs: dict[str, float] = {}
    unpriced: list[str] = []
    unpriced_tokens = 0

    for model, usage in usage_by_model.items():
        price = table.get(model) or _match_prefix(model, table)
        if price is None:
            # Modelo sem preço E sem token não torna o relatório incompleto: o
            # `<synthetic>` que o Claude Code usa para mensagens internas aparece nos
            # traces com uso zerado, e marcá-lo como lacuna faria todo relatório sair
            # com a ressalva de incompletude — um aviso que dispara sempre logo vira
            # ruído que ninguém lê.
            if usage.total_tokens > 0:
                unpriced.append(model)
                unpriced_tokens += usage.total_tokens
            continue
        costs[model] = price.cost_of(usage)

    return CostReport(
        by_model=costs,
        tokens_by_model=usage_by_model,
        unpriced_models=tuple(sorted(unpriced)),
        unpriced_tokens=unpriced_tokens,
    )


def _match_prefix(model: str, table: Mapping[str, ModelPricing]) -> ModelPricing | None:
    """
    Casa ``claude-opus-5-20260101`` com ``claude-opus-5``.

    Só prefixo, e só o mais longo: um sufixo de data ou de plataforma não muda o preço,
    mas ``claude-opus-5`` casando com ``claude-opus-4-8`` mudaria. Casar pelo mais longo
    evita que uma entrada curta capture modelos que pertencem a outra faixa.
    """
    candidates = [key for key in table if model.startswith(key)]
    return table[max(candidates, key=len)] if candidates else None


def cost_per_session(
    traces: TraceSet | Sequence[Session],
    pricing: Mapping[str, ModelPricing] | None = None,
) -> dict[str, float]:
    """
    Custo por sessão — uma observação por unidade experimental.

    A entrada da comparação estatística. Custo total de um harness contra outro é um par
    de números; custo por sessão é uma distribuição, e é dela que saem intervalo de
    confiança e tamanho de efeito.
    """
    sessions = traces.sessions if isinstance(traces, TraceSet) else tuple(traces)
    return {
        session.id: estimate_cost([session], pricing).total
        for session in sessions
    }


def cost_per_tool_call(
    traces: TraceSet | Sequence[Session],
    pricing: Mapping[str, ModelPricing] | None = None,
) -> float | None:
    """
    Custo médio por chamada de ferramenta.

    Aproximação de "custo por unidade de trabalho" quando não há rótulo de tarefa. Sujeita
    à crítica óbvia — nem toda chamada vale o mesmo — e por isso serve para comparar dois
    harnesses na MESMA carga, nunca como número absoluto.
    """
    sessions = traces.sessions if isinstance(traces, TraceSet) else tuple(traces)
    calls = sum(len(s.tool_calls()) for s in sessions)
    return estimate_cost(sessions, pricing).total / calls if calls else None
