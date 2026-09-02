"""
Testes da porta da frente: ``@tool``, ``Harness`` e o cliente de formato OpenAI.

A pergunta que estes testes respondem não é "funciona?" — é **"é fácil?"**. Por isso
vários deles são curtos de propósito: se registrar uma ferramenta ou plugar um provedor
novo precisar de mais linhas do que cabe confortavelmente num teste, a API está errada e
o teste é o primeiro lugar onde isso aparece.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from harness_eng import Harness, quick, tool
from harness_eng.core.clients import (
    OpenAIClient,
    ScriptedClient,
    from_completion,
    to_openai_messages,
)
from harness_eng.core.ports import ModelClient, ModelResponse
from harness_eng.core.tools import ToolRegistry, describe
from harness_eng.trace.model import Role, StopReason, ToolCall, ToolResult, Turn, Usage


class TipoQualquer:
    """Um tipo que o gerador de schema não sabe traduzir. Existe só para o teste."""


def answering(*responses: ModelResponse) -> ScriptedClient:
    return ScriptedClient(responses)


def calls(name: str, **arguments) -> ModelResponse:
    return ModelResponse(
        tool_calls=(ToolCall(id="c1", name=name, arguments=arguments),),
        stop_reason=StopReason.TOOL_USE,
        usage=Usage(input_tokens=50, output_tokens=10, cache_read_tokens=900),
    )


def says(text: str) -> ModelResponse:
    return ModelResponse(
        text=text,
        stop_reason=StopReason.END_TURN,
        usage=Usage(input_tokens=60, output_tokens=8, cache_read_tokens=950),
    )


# ── @tool: a assinatura vira o schema ────────────────────────────────────────────────

def test_a_assinatura_vira_schema() -> None:
    """
    O ponto do decorador: ninguém escreve JSON Schema à mão.

    Schema à mão é uma segunda fonte de verdade, e ela sai de sincronia com a função na
    primeira vez que alguém renomeia um parâmetro.
    """

    @tool
    def buscar(termo: str, limite: int = 10, exatos: bool = False) -> str:
        """Busca no índice.

        Args:
            termo: o que procurar
            limite: máximo de resultados
        """
        return termo

    schema = buscar.spec.input_schema
    assert buscar.spec.name == "buscar"
    assert buscar.spec.description == "Busca no índice."
    assert schema["properties"]["termo"] == {"type": "string", "description": "o que procurar"}
    assert schema["properties"]["limite"]["type"] == "integer"
    assert schema["properties"]["exatos"]["type"] == "boolean"
    # Só o que não tem default é obrigatório — é o que o default significa.
    assert schema["required"] == ["termo"]
    assert schema["additionalProperties"] is False


def test_tipos_compostos_e_opcionais() -> None:
    @tool
    def filtrar(tags: list[str], peso: float | None = None) -> str:
        """:param tags: rótulos a considerar"""
        return ""

    properties = filtrar.spec.input_schema["properties"]
    assert properties["tags"] == {"type": "array", "items": {"type": "string"},
                                  "description": "rótulos a considerar"}
    # Optional[float] é um número que pode faltar, não um tipo novo.
    assert properties["peso"]["type"] == "number"
    assert filtrar.spec.input_schema["required"] == ["tags"]


def test_parametro_sem_anotacao_falha_na_decoracao() -> None:
    """
    Cedo e com mensagem, não tarde e em silêncio.

    Um schema adivinhado passa no teste, chega ao modelo com o tipo errado e reaparece
    como "o modelo mandou string onde eu queria número" — caro de diagnosticar, barato de
    prevenir aqui.
    """
    with pytest.raises(ValueError, match="não tem anotação de tipo"):

        @tool
        def quebrado(a) -> str:  # noqa: ANN001
            return str(a)


def test_tipo_sem_traducao_diz_quais_servem() -> None:
    def com_tipo_estranho(c: TipoQualquer) -> str:
        return ""

    # A mensagem diz o que aconteceu E o que serve, para o conserto não exigir ler o
    # código-fonte da biblioteca.
    with pytest.raises(ValueError, match="não sei virar schema"):
        describe(com_tipo_estranho)
    with pytest.raises(ValueError, match="Tipos suportados"):
        describe(com_tipo_estranho)


def test_anotacao_que_nao_resolve_ganha_mensagem_util() -> None:
    """
    Tipo declarado dentro de outra função vira string sob ``from __future__ import
    annotations`` e ``get_type_hints`` estoura um ``NameError`` que não diz o que fazer.
    Encontrado escrevendo o teste acima — a primeira versão dele quebrava assim.
    """

    class SoAquiDentro:
        pass

    def usa(c: SoAquiDentro) -> str:
        return ""

    with pytest.raises(ValueError, match="não consegui resolver uma anotação"):
        describe(usa)


def test_a_funcao_decorada_continua_uma_funcao() -> None:
    """Sem classe base, sem registro global: uma ferramenta é uma função."""

    @tool
    def dobro(n: int) -> int:
        "Dobra."
        return n * 2

    assert dobro(21) == 42


def test_retorno_vira_texto_que_o_modelo_le_bem() -> None:
    registry = ToolRegistry()

    @tool
    def mapa() -> dict:
        "Devolve um mapa."
        return {"a": 1, "b": None}

    @tool
    def nada() -> None:
        "Não devolve nada."
        return None

    registry.add(mapa, nada)

    saida = registry.execute(ToolCall(id="c1", name="mapa", arguments={})).content
    # JSON, não repr do Python: 'null' e aspas duplas, que o modelo lê melhor.
    assert '"a": 1' in saida and "null" in saida

    vazio = registry.execute(ToolCall(id="c2", name="nada", arguments={}))
    # Retorno None conta como falha silenciosa — é a leitura correta de uma ferramenta
    # que "funcionou" e não deu sinal nenhum ao modelo.
    assert vazio.is_empty


def test_argumento_errado_vira_mensagem_para_o_modelo() -> None:
    """
    A mensagem de erro vai **para o modelo**. "faltou o argumento 'b'" o faz corrigir na
    chamada seguinte; um ``TypeError`` cru, não.
    """
    registry = ToolRegistry()

    @tool
    def soma(a: int, b: int) -> int:
        "Soma."
        return a + b

    registry.add(soma)
    result = registry.execute(ToolCall(id="c1", name="soma", arguments={"a": 1}))

    assert result.is_error
    assert "b" in result.content and "soma" in result.content


def test_funcao_crua_tambem_serve() -> None:
    """``add`` deriva na hora: o decorador é conveniência, não requisito."""

    def triplo(n: int) -> int:
        "Triplica."
        return n * 3

    registry = ToolRegistry().add(triplo)
    assert registry.specs[0].name == "triplo"
    assert registry.execute(ToolCall(id="c1", name="triplo", arguments={"n": 5})).content == "15"


# ── a fachada ────────────────────────────────────────────────────────────────────────

def test_o_caminho_curto_e_curto(tmp_path) -> None:
    """Do prompt à resposta medida, sem montar peça nenhuma na mão."""
    h = Harness(client=answering(calls("soma", a=17, b=25), says("17 + 25 = 42")))

    @h.tool
    def soma(a: int, b: int) -> int:
        "Soma dois números."
        return a + b

    run = h.run("quanto é 17 + 25?", save_to=tmp_path / "run.jsonl")

    assert h.tools == ("soma",)
    assert run.final_text == "17 + 25 = 42"
    assert run.ok
    assert run.trace_path.exists()
    assert "completed" in run.report()
    assert "1 chamadas" in run.report()


def test_final_text_ignora_o_turno_de_resultado() -> None:
    """
    ``session.turns[-1].text`` é o que todo mundo escreveria e está errado quando o último
    turno carrega resultados de ferramenta: devolve vazio e parece "o modelo não respondeu".
    """
    h = Harness(client=answering(calls("eco", texto="oi"), says("terminei")))

    @h.tool
    def eco(texto: str) -> str:
        "Repete."
        return texto

    assert h.run("...").final_text == "terminei"


def test_run_delega_para_o_outcome() -> None:
    h = Harness(client=answering(says("pronto")))
    run = h.run("oi")

    assert run.status.value == "completed"
    assert run.iterations == 1
    assert len(run.session) == 2


def test_ferramentas_de_arquivo_sao_opt_in(tmp_path) -> None:
    """
    Uma biblioteca que dá leitura de disco ao modelo por padrão decide sozinha uma questão
    que é do usuário.
    """
    assert Harness(client=answering(says("x"))).tools == ()
    com_arquivo = Harness(client=answering(says("x")), workspace=tmp_path, file_tools=True)
    assert "read_file" in com_arquivo.tools


def test_quick_e_uma_linha() -> None:
    def soma(a: int, b: int) -> int:
        "Soma."
        return a + b

    import harness_eng.harness as facade

    original = facade.AnthropicClient
    facade.AnthropicClient = lambda **kw: answering(says("42"))  # type: ignore[assignment]
    try:
        assert quick("quanto é 17+25?", soma) == "42"
    finally:
        facade.AnthropicClient = original


# ── qualquer IA ──────────────────────────────────────────────────────────────────────

def test_um_provedor_novo_cabe_em_dez_linhas() -> None:
    """
    A prova de que o pacote serve para qualquer IA: um cliente inteiro, do zero, aqui
    dentro. Se isto precisasse de mais que uma dúzia de linhas, a porta estaria errada —
    e este teste é o lugar onde a promessa do README deixa de ser promessa.
    """

    class MeuModelo:
        model = "meu-modelo-caseiro"

        def __init__(self) -> None:
            self.chamadas = 0

        def complete(self, conversation, tools):
            self.chamadas += 1
            if self.chamadas == 1:
                return ModelResponse(
                    tool_calls=(ToolCall(id="x1", name="gritar", arguments={"texto": "oi"}),),
                    stop_reason=StopReason.TOOL_USE,
                )
            return ModelResponse(text="feito", stop_reason=StopReason.END_TURN)

    assert isinstance(MeuModelo(), ModelClient)

    h = Harness(client=MeuModelo())

    @h.tool
    def gritar(texto: str) -> str:
        "Grita o texto."
        return texto.upper()

    run = h.run("grite oi")
    assert run.final_text == "feito"
    assert run.session.results_by_call_id()["x1"].content == "OI"
    # E o trace do provedor caseiro entra nas mesmas métricas.
    assert "1 chamadas, 0 erros" in run.report()


# ── formato chat completions (OpenAI, Ollama, vLLM, Groq...) ──────────────────────────

def completion(finish: str, *, content=None, tool_calls=None, usage=None, model="gpt-x"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(finish_reason=finish, message=message)],
        usage=usage,
    )


def test_argumentos_chegam_como_string_json() -> None:
    """
    Diferença real entre os dois formatos. Repassar a string faria ``ToolCall.arguments``
    virar texto onde o resto do pacote espera mapa — e o detector de loop, que assina a
    chamada pelos argumentos ordenados, pararia de casar repetições.
    """
    raw = SimpleNamespace(
        id="c1", function=SimpleNamespace(name="soma", arguments='{"a": 2, "b": 3}')
    )
    response = from_completion(completion("tool_calls", tool_calls=[raw]))

    assert response.tool_calls[0].arguments == {"a": 2, "b": 3}
    assert response.stop_reason is StopReason.TOOL_USE


def test_argumento_json_invalido_nao_derruba_a_sessao() -> None:
    raw = SimpleNamespace(id="c1", function=SimpleNamespace(name="soma", arguments="{quebrado"))
    response = from_completion(completion("tool_calls", tool_calls=[raw]))

    # Vira mapa vazio: o executor devolve erro legível e o modelo corrige.
    assert response.tool_calls[0].arguments == {}


def test_prompt_tokens_nao_conta_o_cache_duas_vezes() -> None:
    """
    A diferença que erra em silêncio: ``prompt_tokens`` já inclui os tokens de cache.
    Repassá-lo cru inflaria ``context_size`` — num relatório cujo assunto é justamente
    crescimento de contexto.
    """
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=20,
        prompt_tokens_details=SimpleNamespace(cached_tokens=800),
    )
    response = from_completion(completion("stop", content="oi", usage=usage))

    assert response.usage.input_tokens == 200
    assert response.usage.cache_read_tokens == 800
    assert response.usage.context_size == 1000


def test_sem_detalhe_de_cache_o_input_fica_inteiro() -> None:
    usage = SimpleNamespace(prompt_tokens=500, completion_tokens=10, prompt_tokens_details=None)
    response = from_completion(completion("stop", content="oi", usage=usage))

    assert response.usage.input_tokens == 500
    assert response.usage.cache_read_tokens == 0


def test_resultados_paralelos_viram_uma_mensagem_cada() -> None:
    """
    O canônico guarda os três resultados agrupados; este formato quer três mensagens. Dá
    para derivar a forma espalhada a partir da agrupada — o contrário, não. É por isso que
    o formato canônico guarda a agrupada.
    """
    turn = Turn(
        index=1,
        role=Role.USER,
        tool_results=tuple(ToolResult(call_id=f"c{i}", content=str(i)) for i in range(3)),
    )
    messages = to_openai_messages([turn])

    assert [m["role"] for m in messages] == ["tool", "tool", "tool"]
    assert [m["tool_call_id"] for m in messages] == ["c0", "c1", "c2"]


def test_finish_reasons_viram_motivos_canonicos() -> None:
    for finish, expected in [
        ("stop", StopReason.END_TURN),
        ("length", StopReason.MAX_TOKENS),
        ("tool_calls", StopReason.TOOL_USE),
        ("content_filter", StopReason.REFUSAL),
        ("coisa_nova", StopReason.UNKNOWN),
    ]:
        assert from_completion(completion(finish, content="x")).stop_reason is expected


def test_o_cliente_openai_satisfaz_a_porta() -> None:
    assert isinstance(OpenAIClient(model="gpt-x", client=object()), ModelClient)
