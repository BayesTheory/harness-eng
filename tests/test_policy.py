"""
Testes dos níveis de harness: a política, a fronteira onde ela é aplicada e a métrica
que diz se o nível serviu.

Nenhum teste aqui abre conexão de verdade. A fronteira de rede é exercitada pela decisão
de política — que é onde o bloqueio acontece — e não pelo ``urllib``, porque um teste que
depende da internet para provar que a internet está bloqueada é um teste que falha no
avião e não prova nada a mais.
"""
from __future__ import annotations

import sys

import pytest

from harness_eng import Harness
from harness_eng.core.clients import ScriptedClient
from harness_eng.core.loop import AgentLoop, LoopStatus
from harness_eng.core.policy import (
    BUILDER,
    LEVELS,
    OPERATOR,
    READER,
    RESEARCHER,
    SEALED,
    Denial,
    DenialLog,
    FileAccess,
    NetworkAccess,
    Policy,
    ShellAccess,
    level,
)
from harness_eng.core.ports import ModelResponse
from harness_eng.core.toolkit import PolicyDenied, policy_registry
from harness_eng.core.tools import ToolRegistry
from harness_eng.metrics.policy import MIN_SESSIONS_FOR_SIGNAL, analyse_policy, fit_of
from harness_eng.trace.model import StopReason, ToolCall, TraceSet, Usage


def calling(name: str, **arguments) -> ModelResponse:
    return ModelResponse(
        tool_calls=(ToolCall(id=f"c-{name}", name=name, arguments=arguments),),
        stop_reason=StopReason.TOOL_USE,
    )


def ending() -> ModelResponse:
    return ModelResponse(text="fim", stop_reason=StopReason.END_TURN)


# ── os eixos ─────────────────────────────────────────────────────────────────────────

def test_o_bloqueio_vence_a_allowlist() -> None:
    """
    Quem escreve as duas listas quer "tudo em exemplo.com, menos interno.exemplo.com".
    A ordem inversa faria a exceção não valer nada.
    """
    p = RESEARCHER.allowing("exemplo.com").blocking("interno.exemplo.com")

    assert p.check_url("https://api.exemplo.com/x")
    negado = p.check_url("https://interno.exemplo.com/y")
    assert not negado
    assert negado.reason is Denial.DOMAIN_BLOCKED


def test_subdominio_casa_mas_sufixo_parecido_nao() -> None:
    """
    O ponto na comparação de sufixo é o que separa ``api.pypi.org`` de ``naopypi.org``.
    Esquecê-lo é o jeito clássico de uma allowlist deixar passar um domínio parecido —
    e é o tipo de furo que nenhum teste de caminho feliz encontra.
    """
    p = RESEARCHER.allowing("pypi.org")

    assert p.check_url("https://pypi.org/projeto")
    assert p.check_url("https://api.pypi.org/projeto")
    assert not p.check_url("https://naopypi.org/projeto")
    assert not p.check_url("https://pypi.org.evil.com/x")


def test_sem_rede_nega_qualquer_url() -> None:
    negado = READER.check_url("https://exemplo.com")
    assert not negado and negado.reason is Denial.NO_NETWORK


def test_a_allowlist_de_comando_casa_o_executavel() -> None:
    p = OPERATOR.with_(allowed_commands={"pytest", "git"})

    assert p.check_command("pytest tests/ -q")
    negado = p.check_command("rm -rf /")
    assert not negado
    assert negado.reason is Denial.COMMAND_NOT_ALLOWED
    assert negado.target == "rm"


def test_escrita_exige_o_grau_de_escrita() -> None:
    assert READER.check_read()
    negado = READER.check_write()
    assert not negado and negado.reason is Denial.NO_WRITE_ACCESS
    assert BUILDER.check_write()

    sem_arquivo = SEALED.check_write()
    assert sem_arquivo.reason is Denial.NO_FILE_ACCESS


def test_a_politica_e_imutavel() -> None:
    """
    Política que muda no meio da execução torna o trace impossível de interpretar: "o
    agente leu este arquivo" passa a depender de *quando* ele leu.
    """
    original = RESEARCHER
    derivada = original.allowing("exemplo.com")

    assert original.allowed_domains == frozenset()
    assert derivada.allowed_domains == {"exemplo.com"}
    with pytest.raises((AttributeError, TypeError)):
        original.allowed_domains = frozenset({"x"})  # type: ignore[misc]


def test_variante_perde_o_nome_do_nivel() -> None:
    """
    Um trace que diz ``researcher`` mas roda com a allowlist trocada mente sobre o
    experimento — e a mentira só aparece quando alguém tenta reproduzir o resultado.
    """
    derivada = RESEARCHER.allowing("exemplo.com")

    assert derivada.name == "custom"
    assert derivada.level == -1
    assert RESEARCHER.name == "researcher"


# ── os níveis ────────────────────────────────────────────────────────────────────────

def test_cada_nivel_concede_o_seu_conjunto(tmp_path) -> None:
    conjuntos = {
        0: (),
        1: ("read_file", "list_dir", "find_files"),
        2: ("read_file", "list_dir", "find_files", "fetch_url"),
        3: ("read_file", "list_dir", "find_files", "write_file", "fetch_url"),
        4: ("read_file", "list_dir", "find_files", "write_file", "fetch_url", "run_command"),
    }
    for numero, esperado in conjuntos.items():
        registry = policy_registry(level(numero), tmp_path)
        assert tuple(spec.name for spec in registry.specs) == esperado, f"nível {numero}"


def test_a_ordem_das_ferramentas_e_estavel_no_nivel(tmp_path) -> None:
    """A lista entra no prefixo da requisição: reordenar entre execuções mata o cache."""
    primeira = [s.name for s in policy_registry(OPERATOR, tmp_path).specs]
    for _ in range(5):
        assert [s.name for s in policy_registry(OPERATOR, tmp_path).specs] == primeira


def test_nivel_inexistente_lista_os_que_existem() -> None:
    with pytest.raises(ValueError, match="0 \\(sealed\\).*4 \\(operator\\)"):
        level(9)


def test_os_niveis_sobem_em_capacidade() -> None:
    """Um nível maior nunca concede menos que o anterior — senão 'nível' não quer dizer nada."""
    graus = {FileAccess.NONE: 0, FileAccess.READ: 1, FileAccess.WRITE: 2}
    rede = {NetworkAccess.NONE: 0, NetworkAccess.ALLOWLIST: 1, NetworkAccess.FULL: 2}
    shell = {ShellAccess.NONE: 0, ShellAccess.ALLOWLIST: 1, ShellAccess.FULL: 2}

    anterior = None
    for numero in sorted(LEVELS):
        atual = LEVELS[numero]
        if anterior is not None:
            assert graus[atual.files] >= graus[anterior.files]
            assert rede[atual.network] >= rede[anterior.network]
            assert shell[atual.shell] >= shell[anterior.shell]
        anterior = atual


def test_researcher_nasce_com_allowlist_vazia() -> None:
    """
    Deny by default. Quem sobe para o nível de rede tem de dizer **onde** o agente pode
    ir — uma allowlist que já vem cheia decide isso pelo usuário, na direção permissiva.
    """
    assert RESEARCHER.allowed_domains == frozenset()
    assert not RESEARCHER.check_url("https://qualquer.com")


# ── a fronteira ──────────────────────────────────────────────────────────────────────

def test_a_negativa_chega_ao_modelo_com_o_motivo(tmp_path) -> None:
    """
    A mensagem vai para o modelo, e é o que permite ele se adaptar. Filtrar a ferramenta
    da lista, em vez de negar na chamada, esconderia a informação dos dois lados.
    """
    registry = policy_registry(RESEARCHER.allowing("docs.python.org"), tmp_path)
    resultado = registry.execute(
        ToolCall(id="c1", name="fetch_url", arguments={"url": "https://evil.com/x"})
    )

    assert resultado.is_error
    assert "fora da allowlist" in resultado.content
    assert "evil.com" in resultado.content


def test_a_negativa_e_contada_separada_da_falha(tmp_path) -> None:
    """
    Para o modelo as duas são erro. Para a medição são opostas: falha significa que algo
    quebrou, negativa significa que a política funcionou. Somá-las produz uma taxa de erro
    que **sobe quando você aperta a segurança**.
    """
    registry = policy_registry(RESEARCHER.allowing("ok.com"), tmp_path)

    registry.execute(ToolCall(id="c1", name="fetch_url", arguments={"url": "https://evil.com"}))
    registry.execute(ToolCall(id="c2", name="read_file", arguments={"path": "nao_existe.txt"}))

    assert registry.denials.total == 1, "a falha de leitura foi contada como negativa"
    assert registry.denials.by_reason[Denial.DOMAIN_NOT_ALLOWED] == 1
    assert registry.denials.by_target["evil.com"] == 1


def test_esquema_nao_http_e_recusado(tmp_path) -> None:
    """
    Sem esta checagem, ``file://`` transformaria a ferramenta de rede em leitura de disco
    irrestrita — contornando o eixo de arquivo por inteiro.
    """
    registry = policy_registry(RESEARCHER.with_(network=NetworkAccess.FULL), tmp_path)
    resultado = registry.execute(
        ToolCall(id="c1", name="fetch_url", arguments={"url": "file:///etc/passwd"})
    )

    assert resultado.is_error
    assert "só http e https" in resultado.content


def test_o_redirecionamento_revalida_a_politica() -> None:
    """
    Um domínio liberado responde 302 para um bloqueado e o ``urllib`` segue sozinho. É a
    forma clássica de furar allowlist, e não aparece em teste que só busque URL bem-comportada.
    """
    from harness_eng.core.toolkit import _PolicyRedirect

    handler = _PolicyRedirect(RESEARCHER.allowing("ok.com"))
    with pytest.raises(PolicyDenied):
        handler.redirect_request(None, None, 302, "Found", {}, "https://evil.com/roubado")


def test_comando_negado_nao_executa(tmp_path) -> None:
    registry = policy_registry(OPERATOR.with_(allowed_commands={"echo"}), tmp_path)
    resultado = registry.execute(
        ToolCall(id="c1", name="run_command", arguments={"command": "python -c 'print(1)'"})
    )

    assert resultado.is_error
    assert registry.denials.by_reason[Denial.COMMAND_NOT_ALLOWED] == 1


def test_escrita_fica_presa_ao_workspace(tmp_path) -> None:
    registry = policy_registry(BUILDER, tmp_path)

    dentro = registry.execute(
        ToolCall(id="c1", name="write_file", arguments={"path": "a/b.txt", "content": "oi"})
    )
    fora = registry.execute(
        ToolCall(id="c2", name="write_file", arguments={"path": "../fora.txt", "content": "x"})
    )

    assert not dentro.is_error
    assert (tmp_path / "a" / "b.txt").read_text(encoding="utf-8") == "oi"
    assert fora.is_error and "fora da raiz" in fora.content


# ── o orçamento ──────────────────────────────────────────────────────────────────────

def test_o_orcamento_para_o_loop() -> None:
    gastando = ModelResponse(
        text="...", stop_reason=StopReason.PAUSE_TURN, usage=Usage(input_tokens=600)
    )
    politica = RESEARCHER.with_(token_budget=1000)
    loop = AgentLoop(
        ScriptedClient([gastando for _ in range(9)]), ToolRegistry(), policy=politica
    )
    outcome = loop.run("gasta tudo")

    assert outcome.status is LoopStatus.BUDGET_EXHAUSTED
    assert not outcome.status.finished_on_its_own
    assert outcome.iterations == 2
    assert outcome.session.metadata["denials"]["total"] == 1


def test_o_teto_do_nivel_vale_quando_ninguem_diz_outro() -> None:
    """
    O sentinela ``None`` é o que distingue "o chamador escolheu" de "veio do default".
    Com um inteiro padrão, o teto do nível seria ignorado em silêncio.
    """
    do_nivel = AgentLoop(ScriptedClient([]), ToolRegistry(), policy=SEALED)
    explicito = AgentLoop(ScriptedClient([]), ToolRegistry(), policy=SEALED, max_iterations=7)

    assert do_nivel._max_iterations == SEALED.max_iterations == 20
    assert explicito._max_iterations == 7


# ── o trace guarda a política ────────────────────────────────────────────────────────

def test_o_trace_guarda_a_politica_inteira(tmp_path) -> None:
    """
    Nível sozinho é rótulo que depende de uma tabela que muda entre versões. O trace
    precisa dizer o que estava concedido quando aquela execução aconteceu.
    """
    politica = RESEARCHER.allowing("docs.python.org")
    loop = AgentLoop(
        ScriptedClient([ending()]), policy_registry(politica, tmp_path), policy=politica
    )
    gravada = loop.run("x").session.metadata["policy"]

    assert gravada["network"] == "allowlist"
    assert gravada["allowed_domains"] == ["docs.python.org"]
    assert gravada["files"] == "read"


def test_sem_politica_o_campo_fica_nulo() -> None:
    outcome = AgentLoop(ScriptedClient([ending()]), ToolRegistry()).run("x")
    assert outcome.session.metadata["policy"] is None


# ── a métrica ────────────────────────────────────────────────────────────────────────

def executar(tmp_path, politica, chamadas: list[tuple[str, dict]], nome: str):
    respostas = [calling(n, **a) for n, a in chamadas] + [ending()]
    loop = AgentLoop(
        ScriptedClient(respostas), policy_registry(politica, tmp_path), policy=politica
    )
    return loop.run(nome, session_id=nome).session


def test_o_nivel_apertado_aparece_como_parede(tmp_path) -> None:
    politica = RESEARCHER.allowing("docs.python.org")
    tentativa = [("fetch_url", {"url": "https://stackoverflow.com/q"})]
    sessoes = [
        executar(tmp_path, politica, tentativa, f"s{i}")
        for i in range(MIN_SESSIONS_FOR_SIGNAL + 1)
    ]
    relatorio = analyse_policy(TraceSet.of(sessoes))

    assert relatorio.has_signal
    assert relatorio.total_denials == MIN_SESSIONS_FOR_SIGNAL + 1
    assert "apertou" in relatorio.verdict()
    assert next(iter(relatorio.blocked_targets)) == "stackoverflow.com"


def test_o_nivel_folgado_aparece_como_ferramenta_ociosa(tmp_path) -> None:
    """
    Ferramenta concedida que ninguém chamou é risco carregado de graça — e não aparece em
    lugar nenhum, justamente porque nada dá errado.
    """
    sessoes = [
        executar(tmp_path, RESEARCHER, [("list_dir", {"path": "."})], f"t{i}")
        for i in range(MIN_SESSIONS_FOR_SIGNAL + 1)
    ]
    relatorio = analyse_policy(TraceSet.of(sessoes))

    assert relatorio.total_denials == 0
    assert "fetch_url" in relatorio.never_used()
    assert "sobrou" in relatorio.verdict()


def test_ferramenta_usada_as_vezes_nao_conta_como_excesso(tmp_path) -> None:
    """Usar em uma sessão de seis é o caso normal, não desperdício."""
    sessoes = [
        executar(tmp_path, RESEARCHER, [("list_dir", {"path": "."})], f"u{i}") for i in range(5)
    ]
    sessoes.append(executar(tmp_path, RESEARCHER, [("read_file", {"path": "x"})], "u5"))
    relatorio = analyse_policy(TraceSet.of(sessoes))

    assert "read_file" not in relatorio.never_used()


def test_amostra_pequena_sai_como_indicio(tmp_path) -> None:
    """
    Um relatório que chama n=2 de evidência comete o mesmo erro que este repositório
    documenta na própria camada estatística.
    """
    sessoes = [
        executar(tmp_path, RESEARCHER, [("list_dir", {"path": "."})], f"v{i}") for i in range(2)
    ]
    relatorio = analyse_policy(TraceSet.of(sessoes))

    assert not relatorio.has_signal
    assert "indício, não conclusão" in relatorio.verdict()


def test_sessao_sem_politica_nao_quebra_a_metrica() -> None:
    outcome = AgentLoop(ScriptedClient([ending()]), ToolRegistry()).run("x")
    ajuste = fit_of(outcome.session)

    assert not ajuste.has_policy
    assert analyse_policy(TraceSet.of([outcome.session])).verdict().startswith("nenhuma sessão")


def test_o_log_de_negativas_ignora_decisao_permitida() -> None:
    from harness_eng.core.policy import ALLOWED

    log = DenialLog()
    log.record(ALLOWED)
    assert log.total == 0


# ── a fachada ────────────────────────────────────────────────────────────────────────

def test_harness_por_nivel(tmp_path) -> None:
    h = Harness(client=ScriptedClient([ending()]), level=2, workspace=tmp_path)

    assert h.tools == ("read_file", "list_dir", "find_files", "fetch_url")
    assert h.policy is RESEARCHER
    assert h.run("x").session.metadata["policy"]["name"] == "researcher"


def test_nivel_e_politica_juntos_sao_erro() -> None:
    """
    Precedência silenciosa de um sobre o outro só apareceria como "por que este agente tem
    internet?" muito depois.
    """
    with pytest.raises(ValueError, match="não os dois"):
        Harness(client=ScriptedClient([]), level=1, policy=Policy())


def test_sem_nivel_o_registro_nasce_vazio() -> None:
    assert Harness(client=ScriptedClient([])).tools == ()
    assert Harness(client=ScriptedClient([])).policy is None


def test_o_relatorio_sobrevive_a_um_trace_sem_medicao(tmp_path, capsys) -> None:
    """
    Regressão de um crash real, encontrado rodando ``analyze`` sobre o primeiro trace com
    nível: sem turnos com uso de token, ``median_growth`` e ``hit_rate`` devolvem ``None``
    — corretamente, porque "ausência é ausência" atravessa o pacote — e o formatador
    estourava ``TypeError`` no meio do relatório.

    A métrica seguia a regra da casa; a apresentação, não. Um trace curto é dado legítimo.
    """
    from harness_eng.cli import main
    from harness_eng.trace.sources.native import NativeSink

    sessao = AgentLoop(ScriptedClient([ending()]), ToolRegistry(), policy=READER).run("x")
    NativeSink().write(sessao.session, tmp_path / "curto.jsonl")

    assert main(["analyze", str(tmp_path)]) == 0
    saida = capsys.readouterr().out
    assert "CACHE" in saida and "CUSTO" in saida, "o relatório parou no meio"
    # O travessão é a marca de "não deu para medir" — melhor que zero, que seria medição.
    assert "—" in saida


def test_o_relatorio_nao_morre_no_console_do_windows() -> None:
    """
    Regressão de um bug achado rodando `analyze` sobre 55 transcripts REAIS.

    O console do Windows usa cp1252, e o `←` do marcador de outlier levantava
    `UnicodeEncodeError` no meio da impressão — o relatório morria depois de já ter
    escrito metade, com traceback por cima da saída.

    Os 170 testes não pegaram porque rodam sobre traces sintéticos, que não têm outlier:
    o caractere nunca chegava a ser impresso. É exatamente o modo de falha que este
    repositório cobra dos outros — comportamento que só aparece com dado real —
    acontecendo com ele mesmo.
    """
    import io

    from harness_eng.cli import _force_utf8_output

    # Stream redirecionado, sem reconfigure(): não pode levantar. Seguir sem a garantia de
    # acento é correto; derrubar o relatório inteiro, não.
    class SemReconfigure(io.StringIO):
        reconfigure = None

    original = sys.stdout
    sys.stdout = SemReconfigure()
    try:
        _force_utf8_output()
    finally:
        sys.stdout = original


def test_o_marcador_de_outlier_sobrevive_a_um_console_limitado(tmp_path, capsys) -> None:
    """O `←` precisa chegar ao fim do relatório, não ao meio."""
    from harness_eng.cli import _force_utf8_output

    _force_utf8_output()
    print("outlier: ←")
    assert "←" in capsys.readouterr().out
