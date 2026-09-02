"""
O nível serviu para a tarefa?

Esta é a métrica que não existe em lugar nenhum. Um harness com política reporta, no
máximo, que bloqueou alguma coisa. Aqui a pergunta é outra e é a que decide configuração:
**o nível concedido bate com o nível necessário?**

Duas direções, e as duas custam caro sem aparecer:

* **concedeu demais.** Ferramenta disponível que o agente nunca chamou é risco carregado
  de graça. Ninguém percebe, porque nada dá errado — é exatamente por nada dar errado que
  o excesso sobrevive por anos.
* **concedeu de menos.** O agente bateu numa parede e teve de contornar. Isso aparece como
  execução mais longa, mais cara ou pior — e é atribuído ao modelo, ao prompt, à tarefa,
  a qualquer coisa menos à política, porque a parede não estava no relatório.

Puro sobre o formato canônico. Lê ``Session.metadata`` como dicionário comum — não importa
:mod:`harness_eng.core.policy` nem sabe que ele existe. Não é purismo: é o que permite
esta métrica rodar sobre trace de **qualquer** harness que grave essas chaves, inclusive um
que não seja este.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..trace.model import Session, TraceSet

#: Abaixo disto o padrão é anedota. Uma sessão que não usou ``fetch_url`` não prova
#: excesso de permissão — prova que aquela tarefa não precisou de rede. O mesmo raciocínio
#: de ``MIN_CALLS_FOR_SIGNAL`` em ``metrics/tools.py``, e pelo mesmo motivo: número sem
#: amostra vira conclusão errada com aparência de dado.
MIN_SESSIONS_FOR_SIGNAL = 5


@dataclass(frozen=True, slots=True)
class PolicyFit:
    """Como o nível se comportou numa sessão."""

    session_id: str
    level: int | None = None
    name: str = "sem política"
    granted: tuple[str, ...] = ()
    used: tuple[str, ...] = ()
    denials: int = 0
    denials_by_reason: Mapping[str, int] = field(default_factory=dict)
    blocked_targets: Mapping[str, int] = field(default_factory=dict)

    @property
    def has_policy(self) -> bool:
        return self.level is not None or self.name != "sem política"

    @property
    def unused(self) -> tuple[str, ...]:
        """Ferramentas concedidas e nunca chamadas — o excesso, nesta sessão."""
        return tuple(sorted(set(self.granted) - set(self.used)))

    @property
    def hit_a_wall(self) -> bool:
        return self.denials > 0

    def as_dict(self) -> dict:
        return {
            "session": self.session_id,
            "level": self.level,
            "policy": self.name,
            "granted": list(self.granted),
            "used": list(self.used),
            "unused": list(self.unused),
            "denials": self.denials,
            "denials_by_reason": dict(self.denials_by_reason),
            "blocked_targets": dict(self.blocked_targets),
        }


def _as_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def fit_of(session: Session) -> PolicyFit:
    """
    Lê o ajuste de política de uma sessão.

    Tolerante a metadata ausente ou de outro formato: um trace gravado por um harness que
    não registra política não é erro, é uma sessão sem política — e devolver
    ``PolicyFit`` vazio deixa o agregado somar as duas coisas sem caso especial.
    """
    metadata = session.metadata or {}
    policy = metadata.get("policy")
    denials = metadata.get("denials") or {}

    granted = metadata.get("tools")
    granted_names = tuple(str(t) for t in granted) if isinstance(granted, (list, tuple)) else ()
    used = tuple(sorted({call.name for call in session.tool_calls()}))

    by_reason = denials.get("by_reason") if isinstance(denials, Mapping) else None
    by_target = denials.get("by_target") if isinstance(denials, Mapping) else None

    return PolicyFit(
        session_id=session.id,
        level=policy.get("level") if isinstance(policy, Mapping) else None,
        name=(policy.get("name") if isinstance(policy, Mapping) else None) or "sem política",
        granted=granted_names,
        used=used,
        denials=_as_int(denials.get("total")) if isinstance(denials, Mapping) else 0,
        denials_by_reason=dict(by_reason) if isinstance(by_reason, Mapping) else {},
        blocked_targets=dict(by_target) if isinstance(by_target, Mapping) else {},
    )


@dataclass(frozen=True, slots=True)
class PolicyReport:
    """O ajuste de política agregado sobre um conjunto de sessões."""

    fits: tuple[PolicyFit, ...] = ()

    @property
    def with_policy(self) -> tuple[PolicyFit, ...]:
        return tuple(f for f in self.fits if f.has_policy)

    @property
    def has_signal(self) -> bool:
        """Se há sessões suficientes para o padrão significar alguma coisa."""
        return len(self.with_policy) >= MIN_SESSIONS_FOR_SIGNAL

    @property
    def sessions_that_hit_a_wall(self) -> tuple[PolicyFit, ...]:
        return tuple(f for f in self.with_policy if f.hit_a_wall)

    @property
    def total_denials(self) -> int:
        return sum(f.denials for f in self.with_policy)

    @property
    def denials_by_reason(self) -> dict[str, int]:
        counter: Counter[str] = Counter()
        for fit in self.with_policy:
            counter.update(fit.denials_by_reason)
        return dict(counter.most_common())

    @property
    def blocked_targets(self) -> dict[str, int]:
        """Os alvos negados, do mais tentado ao menos. É o que diz *qual* parede incomoda."""
        counter: Counter[str] = Counter()
        for fit in self.with_policy:
            counter.update(fit.blocked_targets)
        return dict(counter.most_common())

    def never_used(self) -> dict[str, int]:
        """
        Ferramentas concedidas que **nenhuma** sessão chamou, com quantas vezes foram
        concedidas.

        Só entram as que ninguém usou em execução nenhuma. Uma ferramenta usada em uma
        sessão de vinte não é excesso — é uma ferramenta que serve às vezes, que é o caso
        normal.
        """
        granted: Counter[str] = Counter()
        used: set[str] = set()
        for fit in self.with_policy:
            granted.update(fit.granted)
            used.update(fit.used)
        return {name: count for name, count in granted.most_common() if name not in used}

    def verdict(self) -> str:
        """
        A leitura em uma frase, com a incerteza dita em voz alta.

        Nunca afirma além da amostra: abaixo de :data:`MIN_SESSIONS_FOR_SIGNAL` sessões o
        veredito diz que é indício, não conclusão. Um relatório que chama n=2 de evidência
        é o mesmo erro que este repositório documenta na própria camada estatística.
        """
        medidas = self.with_policy
        if not medidas:
            return "nenhuma sessão registrou política — nada a comparar."

        ressalva = "" if self.has_signal else (
            f" (só {len(medidas)} sessões: indício, não conclusão — "
            f"{MIN_SESSIONS_FOR_SIGNAL} é o mínimo para o padrão contar)"
        )

        if self.total_denials:
            paredes = len(self.sessions_that_hit_a_wall)
            alvo = next(iter(self.blocked_targets), None)
            onde = f", mais em '{alvo}'" if alvo else ""
            return (
                f"o nível apertou: {self.total_denials} negativas em {paredes} de "
                f"{len(medidas)} sessões{onde}{ressalva}."
            )

        ociosas = self.never_used()
        if ociosas:
            nomes = ", ".join(ociosas)
            return (
                f"o nível sobrou: {nomes} concedida(s) e nunca usada(s) em "
                f"{len(medidas)} sessões — risco carregado de graça{ressalva}."
            )

        return f"o nível serviu: nada foi negado e tudo que foi concedido foi usado{ressalva}."

    def as_dict(self) -> dict:
        return {
            "sessions_with_policy": len(self.with_policy),
            "has_signal": self.has_signal,
            "total_denials": self.total_denials,
            "sessions_that_hit_a_wall": len(self.sessions_that_hit_a_wall),
            "denials_by_reason": self.denials_by_reason,
            "blocked_targets": self.blocked_targets,
            "never_used": self.never_used(),
            "verdict": self.verdict(),
        }


def analyse_policy(traces: TraceSet) -> PolicyReport:
    """O ajuste de política sobre um conjunto de traces."""
    return PolicyReport(fits=tuple(fit_of(session) for session in traces))
