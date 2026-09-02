"""
Testes do harness mínimo: o loop, o executor de ferramenta e o trace nativo.

Nenhuma chave de API, nenhuma rede, nenhum modelo. É o motivo de :class:`ScriptedClient`
existir: os modos de falha que interessam — estourar o teto de iterações, receber
``pause_turn``, ser cortado por ``max_tokens`` — são justamente os que não dá para
provocar sob demanda pagando por eles, e um comportamento de borda que só aparece em
produção é um comportamento que ninguém verifica.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from harness_eng.core.clients import (
    AnthropicClient,
    ScriptedClient,
    from_message,
    to_messages,
)
from harness_eng.core.loop import AgentLoop, LoopStatus
from harness_eng.core.ports import ModelClient, ModelError, ModelResponse, ToolSpec
from harness_eng.core.tools import ToolError, ToolRegistry, workspace_registry
from harness_eng.metrics.tools import analyse_tools
from harness_eng.trace.model import Role, StopReason, ToolCall, TraceSet, Usage
from harness_eng.trace.ports import TraceSink, TraceSource
from harness_eng.trace.sources.native import FORMAT, NativeSink, NativeSource

BASE = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)


def ticking_clock():
    """Relógio determinístico: um segundo por leitura. Duração sem dormir."""
    state = {"n": 0}

    def now() -> datetime:
        state["n"] += 1
        return BASE + timedelta(seconds=state["n"])

    return now


def asks(name: str, /, **arguments) -> ModelResponse:
    """Resposta que pede uma ferramenta."""
    return ModelResponse(
        tool_calls=(ToolCall(id=f"call-{name}-{len(arguments)}", name=name, arguments=arguments),),
        stop_reason=StopReason.TOOL_USE,
        usage=Usage(input_tokens=10, output_tokens=5, cache_read_tokens=100),
    )


def finishes(text: str = "pronto") -> ModelResponse:
    return ModelResponse(
        text=text,
        stop_reason=StopReason.END_TURN,
        usage=Usage(input_tokens=10, output_tokens=5, cache_read_tokens=100),
    )


def echo_registry(**handlers) -> ToolRegistry:
    registry = ToolRegistry()
    for name, handler in handlers.items():
        spec = ToolSpec(name=name, description=name, input_schema={"type": "object"})
        registry.register(spec, handler)
    return registry


# ── o loop ───────────────────────────────────────────────────────────────────────────

def test_a_conversa_e_o_trace() -> None:
    """
    Um ciclo completo: pede ferramenta, recebe resultado, termina.

    A sessão resultante tem de ser indistinguível de uma lida de transcript — é a
    propriedade que permite as métricas rodarem sobre o harness próprio sem caso especial.
    """
    loop = AgentLoop(
        ScriptedClient([asks("echo", value="oi"), finishes()]),
        echo_registry(echo=lambda args: str(args.get("value", ""))),
        clock=ticking_clock(),
    )
    outcome = loop.run("faça algo")

    assert outcome.status is LoopStatus.COMPLETED
    assert outcome.status.finished_on_its_own
    assert outcome.iterations == 2
    assert [t.role for t in outcome.session] == [
        Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT,
    ]
    assert outcome.session.source == "native"
    assert outcome.tool_calls == 1

    result = outcome.session.results_by_call_id()["call-echo-1"]
    assert result.content == "oi"
    assert not result.is_error


def test_resultados_paralelos_voltam_num_turno_so() -> None:
    """
    Três ferramentas pedidas juntas, três resultados numa mensagem só.

    Espalhá-los em mensagens separadas é aceito pela API e ensina o modelo a parar de
    pedir chamadas em paralelo — regressão de desempenho que não levanta erro nenhum e
    que nenhum teste de conteúdo pega. Por isso o teste é sobre a *forma* da conversa.
    """
    parallel = ModelResponse(
        tool_calls=tuple(
            ToolCall(id=f"c{i}", name="echo", arguments={"value": str(i)}) for i in range(3)
        ),
        stop_reason=StopReason.TOOL_USE,
    )
    client = ScriptedClient([parallel, finishes()])
    loop = AgentLoop(client, echo_registry(echo=lambda a: str(a["value"])), clock=ticking_clock())
    outcome = loop.run("três de uma vez")

    user_turns = [t for t in outcome.session if t.role is Role.USER and t.tool_results]
    assert len(user_turns) == 1, "os resultados foram partidos em mensagens separadas"
    assert len(user_turns[0].tool_results) == 3

    # E a mesma unicidade tem de sobreviver à tradução para o formato da API.
    messages = to_messages(client.seen[-1])
    tool_result_messages = [
        m for m in messages
        if isinstance(m["content"], list) and m["content"][0].get("type") == "tool_result"
    ]
    assert len(tool_result_messages) == 1
    assert len(tool_result_messages[0]["content"]) == 3


def test_teto_de_iteracoes_nao_e_sucesso() -> None:
    """
    Um modelo que nunca emite ``end_turn`` roda até a conta acabar. O teto para isso — e
    bater nele é um desfecho próprio, não um final feliz com menos turnos.
    """
    loop = AgentLoop(
        ScriptedClient([asks("echo", value="x") for _ in range(10)]),
        echo_registry(echo=lambda a: "ok"),
        max_iterations=3,
        clock=ticking_clock(),
    )
    outcome = loop.run("nunca termina")

    assert outcome.status is LoopStatus.MAX_ITERATIONS
    assert not outcome.status.finished_on_its_own
    assert outcome.iterations == 3
    assert outcome.session.metadata["status"] == "max_iterations"


def test_pause_turn_nao_e_fim_de_turno() -> None:
    """
    O caso que este pacote não sabia representar até o harness precisar dele.

    ``pause_turn`` significa "pausei, me retome" — e um loop que trata tudo que não é
    ``tool_use`` como fim devolve trabalho pela metade **sem erro nenhum**. O teste checa
    as duas metades: o loop continua, e continua sem inventar um turno de user.
    """
    paused = ModelResponse(text="metade", stop_reason=StopReason.PAUSE_TURN)
    client = ScriptedClient([paused, finishes("resto")])
    outcome = AgentLoop(client, ToolRegistry(), clock=ticking_clock()).run("tarefa longa")

    assert outcome.status is LoopStatus.COMPLETED
    assert outcome.iterations == 2
    # A segunda requisição foi enviada com a conversa terminando no assistant pausado.
    assert client.seen[-1][-1].role is Role.ASSISTANT
    assert client.seen[-1][-1].stop_reason is StopReason.PAUSE_TURN


def test_resposta_truncada_nao_executa_a_ferramenta() -> None:
    """
    Cortada por ``max_tokens``, a última chamada pode estar escrita pela metade.
    Executá-la é rodar um argumento que o modelo não terminou de escrever.
    """
    executed: list[str] = []
    truncated = ModelResponse(
        tool_calls=(ToolCall(id="c1", name="echo", arguments={"value": "incompl"}),),
        stop_reason=StopReason.MAX_TOKENS,
    )
    loop = AgentLoop(
        ScriptedClient([truncated]),
        echo_registry(echo=lambda a: executed.append(str(a)) or "ok"),
        clock=ticking_clock(),
    )
    outcome = loop.run("resposta longa")

    assert outcome.status is LoopStatus.TRUNCATED
    assert executed == [], "executou uma chamada de ferramenta truncada"


def test_recusa_sai_com_categoria() -> None:
    refusal = ModelResponse(stop_reason=StopReason.REFUSAL, stop_details={"category": "cyber"})
    outcome = AgentLoop(ScriptedClient([refusal]), ToolRegistry(), clock=ticking_clock()).run("x")

    assert outcome.status is LoopStatus.REFUSED
    assert outcome.detail == "cyber"


def test_falha_de_modelo_vira_turno_no_trace() -> None:
    """
    Uma execução que morreu na segunda chamada é dado. Apagá-la do trace deixaria o custo
    medido sem a explicação de por que a sessão é curta.
    """

    class Broken:
        model = "quebrado"

        def complete(self, conversation, tools):
            raise ModelError("APIConnectionError: rede caiu", retryable=True)

    outcome = AgentLoop(Broken(), ToolRegistry(), clock=ticking_clock()).run("x")

    assert outcome.status is LoopStatus.MODEL_ERROR
    assert outcome.session.turns[-1].stop_reason is StopReason.ERROR
    assert "rede caiu" in outcome.session.turns[-1].text


def test_max_iterations_zero_e_recusado() -> None:
    with pytest.raises(ValueError):
        AgentLoop(ScriptedClient([]), ToolRegistry(), max_iterations=0)


# ── o executor de ferramenta ─────────────────────────────────────────────────────────

def test_ferramenta_desconhecida_e_erro_contavel() -> None:
    """Modelo inventando nome de ferramenta é modo de falha real, não exceção."""
    registry = echo_registry(echo=lambda a: "ok")
    result = registry.execute(ToolCall(id="c1", name="inexistente", arguments={}))

    assert result.is_error
    assert "desconhecida" in result.content


def test_excecao_do_handler_nao_derruba_a_sessao() -> None:
    registry = echo_registry(
        recusa=lambda a: (_ for _ in ()).throw(ToolError("caminho fora da raiz")),
        explode=lambda a: 1 / 0,
    )

    recusado = registry.execute(ToolCall(id="c1", name="recusa", arguments={}))
    explodido = registry.execute(ToolCall(id="c2", name="explode", arguments={}))

    assert recusado.is_error and "fora da raiz" in recusado.content
    # Bug do executor também vira resultado de erro, com o tipo preservado para o
    # relatório poder distinguir "recusou certo" de "quebrou".
    assert explodido.is_error and "ZeroDivisionError" in explodido.content


def test_retorno_vazio_conta_como_falha_silenciosa() -> None:
    """
    Sucesso sem saída nenhuma deixa o modelo sem sinal para o passo seguinte. Se o
    executor marcasse ``content_kinds=("text",)`` num retorno vazio, ``is_empty`` daria
    False e a métrica de falha silenciosa mediria zero para sempre.
    """
    registry = echo_registry(vazio=lambda a: "", cheio=lambda a: "conteúdo")

    assert registry.execute(ToolCall(id="c1", name="vazio", arguments={})).is_empty
    assert not registry.execute(ToolCall(id="c2", name="cheio", arguments={})).is_empty


def test_a_ordem_das_ferramentas_e_estavel() -> None:
    """
    A lista de ferramentas entra no prefixo da requisição. Reordená-la entre execuções
    invalida o cache de tudo que vem depois, sem nenhum sintoma além da conta no fim do mês.
    """
    names = ["read_file", "list_dir", "find_files"]
    for _ in range(5):
        registry = workspace_registry(__import__("pathlib").Path("."), tools=names)
        assert [spec.name for spec in registry.specs] == names


def test_ferramenta_de_leitura_nao_escapa_da_raiz(tmp_path) -> None:
    (tmp_path / "dentro.txt").write_text("visível", encoding="utf-8")
    (tmp_path.parent / "fora.txt").write_text("segredo", encoding="utf-8")
    registry = workspace_registry(tmp_path)

    dentro = registry.execute(ToolCall(id="c1", name="read_file", arguments={"path": "dentro.txt"}))
    fora = registry.execute(
        ToolCall(id="c2", name="read_file", arguments={"path": "../fora.txt"})
    )

    assert dentro.content == "visível"
    assert fora.is_error and "fora da raiz" in fora.content


def test_argumento_ausente_e_erro_e_nao_excecao(tmp_path) -> None:
    registry = workspace_registry(tmp_path, tools=["read_file"])
    result = registry.execute(ToolCall(id="c1", name="read_file", arguments={}))

    assert result.is_error and "obrigatório" in result.content


# ── tradução de/para a API ───────────────────────────────────────────────────────────

class FakeBlock:
    def __init__(self, **fields) -> None:
        self.__dict__.update(fields)


class FakeMessage:
    def __init__(self, content, stop_reason, usage=None, model="claude-opus-5") -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage
        self.model = model
        self.stop_details = None


def test_traducao_le_todos_os_tipos_de_bloco() -> None:
    """
    Percorrer os blocos, não assumir que o primeiro é texto.

    É a lição registrada no README: a primeira versão do adapter do Claude Code lia só
    ``type == "text"`` e reportou "58 de 60 chamadas voltaram vazias" — defeito da
    ferramenta de medição travestido de achado sobre o harness.
    """
    message = FakeMessage(
        content=[
            FakeBlock(type="thinking", thinking="raciocínio"),
            FakeBlock(type="text", text="resposta"),
            FakeBlock(type="tool_use", id="c1", name="read_file", input={"path": "a.py"}),
        ],
        stop_reason="tool_use",
        usage=FakeBlock(
            input_tokens=100,
            output_tokens=20,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=4000,
            service_tier="standard",
        ),
    )
    response = from_message(message)

    assert response.text == "resposta"
    assert response.thinking == "raciocínio"
    assert response.tool_calls[0].name == "read_file"
    assert response.stop_reason is StopReason.TOOL_USE
    # ``None`` de campo de cache vira 0 — senão a soma da sessão quebra.
    assert response.usage.cache_write_tokens == 0
    assert response.usage.cache_read_tokens == 4000
    assert response.usage.cache_hit_rate == pytest.approx(4000 / 4100)
    assert response.replay_content is message.content


def test_pause_turn_sobrevive_a_traducao() -> None:
    """Se ``StopReason.parse`` devolvesse UNKNOWN aqui, o loop encerraria no meio."""
    assert from_message(FakeMessage([], "pause_turn")).stop_reason is StopReason.PAUSE_TURN


def test_resposta_sem_uso_fica_sem_uso() -> None:
    """Ausência é ausência: ``None``, não ``Usage(0, 0, 0, 0)``."""
    assert from_message(FakeMessage([], "end_turn", usage=None)).usage is None


def test_replay_usa_os_blocos_originais() -> None:
    """
    O bloco de pensamento carrega assinatura do provedor. Reconstruí-lo a partir do texto
    canônico entrega um bloco sem assinatura, que a API descarta — daí ``replay_content``.
    """
    original = [FakeBlock(type="thinking", thinking="...", signature="abc123")]
    loop_turn = replace(
        AgentLoop(ScriptedClient([finishes()]), ToolRegistry(), clock=ticking_clock())
        .run("x")
        .session.turns[1],
        raw={"replay_content": original},
    )
    messages = to_messages([loop_turn])

    assert messages[0]["content"] is original


# ── trace nativo ─────────────────────────────────────────────────────────────────────

def test_o_round_trip_nao_perde_campo(tmp_path) -> None:
    """
    Sessão → disco → sessão. Se um campo se perde no caminho, o formato canônico tem um
    buraco — e é melhor descobrir aqui do que num relatório que reporta zero onde havia dado.
    """
    loop = AgentLoop(
        ScriptedClient([asks("echo", value="oi"), finishes()]),
        echo_registry(echo=lambda a: str(a["value"])),
        clock=ticking_clock(),
    )
    original = loop.run("faça algo", session_id="s-round-trip").session

    path = NativeSink().write(original, tmp_path / "sessao.jsonl")
    recovered = NativeSource().load(path)

    assert recovered is not None
    assert recovered.id == original.id
    assert recovered.source == original.source
    assert recovered.started_at == original.started_at
    assert recovered.ended_at == original.ended_at
    assert recovered.metadata == original.metadata
    # ``raw`` é a única exclusão, e é deliberada: trace é registro de medição, não
    # checkpoint retomável.
    assert [replace(t, raw={}) for t in original.turns] == list(recovered.turns)


def test_o_leitor_nativo_recusa_trace_de_outra_origem(tmp_path) -> None:
    """
    ``~/.claude/projects`` também é cheio de ``.jsonl``. Adivinhar pela extensão faria
    este leitor engolir transcript alheio e devolver sessão vazia em vez de ``None``.
    """
    alheio = tmp_path / "outro.jsonl"
    alheio.write_text(json.dumps({"type": "assistant", "uuid": "x"}) + "\n", encoding="utf-8")
    source = NativeSource()

    assert source.load(alheio) is None
    assert source.skipped["formato de outra origem"] == 1


def test_linha_corrompida_nao_custa_o_arquivo_inteiro(tmp_path) -> None:
    """Sessão morta no meio deixa a última linha truncada. O resto do dado continua bom."""
    path = tmp_path / "parcial.jsonl"
    path.write_text(
        json.dumps({"type": "session", "format": FORMAT, "id": "s1", "source": "native"})
        + "\n"
        + json.dumps({"type": "turn", "index": 0, "role": "user", "text": "oi"})
        + '\n{"type": "turn", "index": 1, "rol',
        encoding="utf-8",
    )
    source = NativeSource()
    session = source.load(path)

    assert session is not None
    assert len(session) == 1
    assert source.skipped["linha corrompida"] == 1


def test_o_par_nativo_satisfaz_as_portas() -> None:
    assert isinstance(NativeSource(), TraceSource)
    assert isinstance(NativeSink(), TraceSink)


def test_os_dois_clientes_satisfazem_a_porta() -> None:
    """
    ``runtime_checkable`` só confere nome de método — e é exatamente o erro que acontece:
    renomear ``complete`` num cliente e descobrir na primeira execução paga.
    """
    assert isinstance(ScriptedClient([]), ModelClient)
    assert isinstance(AnthropicClient(client=object()), ModelClient)


def test_as_metricas_rodam_sobre_o_proprio_harness(tmp_path) -> None:
    """
    O círculo fechado, que é a razão de o harness existir neste repositório: o trace que
    ele emite entra nas mesmas métricas que medem os harnesses de terceiro, sem adapter.
    """
    loop = AgentLoop(
        ScriptedClient([asks("falha"), asks("echo", value="ok"), finishes()]),
        echo_registry(
            echo=lambda a: str(a["value"]),
            falha=lambda a: (_ for _ in ()).throw(ToolError("não deu")),
        ),
        clock=ticking_clock(),
    )
    session = loop.run("duas ferramentas").session
    path = NativeSink().write(session, tmp_path / "s.jsonl")

    traces = TraceSet.of(list(NativeSource().sessions(tmp_path)))
    health = analyse_tools(traces)

    assert path.exists()
    assert health.total_calls == 2
    assert health.overall_error_rate == 0.5
    ranked = health.ranked_by_error_rate(only_with_signal=False)
    assert {t.name for t in ranked} == {"echo", "falha"}
