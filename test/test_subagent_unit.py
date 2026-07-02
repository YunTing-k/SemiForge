# -*- coding: utf-8 -*-
"""
Unit tests for SubAgent stop_mode / subagent_tools / tool_handlers.

Run with: python test/test_subagent_unit.py

These tests mock LLM responses (no real API calls) and validate the
stop_mode branching logic, local tool interception, and exhaustion handling.
"""
import sys
import os
import json
import logging
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import patch, MagicMock, PropertyMock
from src.context.agent_context import AgentContext
from src.agent.subagent import SubAgent
from src.agent.progress import AgentStatus
from src.tool.subagent_tools import (
    get_tools,
    SUBAGENT_TOOL_INSPECT_XLSX,
    SUBAGENT_TOOL_ADJUST_PARAMS,
    SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS,
    SUBAGENT_TOOL_REPORT_VERIFICATION,
    SUBAGENT_TOOL_ACCEPT_QUALITY,
    SUBAGENT_TOOL_CONTRIBUTE_KNOWLEDGE,
)
from src.constants import *
from rich.console import Console

logging.getLogger('logger').setLevel(logging.CRITICAL)

_CONSOLE = Console(quiet=True)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_empty_ctx() -> AgentContext:
    """Build a minimal AgentContext with fake configs so SubAgent.__init__ passes."""
    ctx = AgentContext()
    ctx.session_uuid = "test-unit-001"
    ctx.api_configs = {
        "FAST_MODEL_NAME": "mock-model",
        "FAST_MODEL_TEMPERATURE": 0.3,
        "FAST_MODEL_MAX_TOKENS": 4096,
        "FAST_MODEL_ENABLE_REASONING": False,
        "FAST_MODEL_DEEPSEEK_SUPPORT": False,
        "TIMEOUT_MS": 30000,
    }
    ctx.agent_configs = {}
    ctx.mcps_configs = {}
    ctx.mcp_router = MagicMock()
    ctx.mcp_router.reg_tools = []
    return ctx


def _fake_chat_completion(content: str | None = None, tool_calls=None, reasoning: str | None = None) -> MagicMock:
    """Return a MagicMock that passes isinstance(..., ChatCompletion) checks."""
    from openai.types.chat import ChatCompletion
    resp = MagicMock(spec=ChatCompletion)
    msg_data = {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }
    if reasoning is not None:
        msg_data["reasoning_content"] = reasoning
    msg = MagicMock()
    msg.model_dump.return_value = msg_data
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15
    usage.prompt_tokens_details = None
    resp.usage = usage
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# 0.1.7  stop_mode="plain_text"
# ---------------------------------------------------------------------------

@patch("src.agent.subagent.OpenAI")
def test_plain_text_content_done(mock_openai):
    """plain_text mode: LLM returns content → DONE."""
    ctx = _make_empty_ctx()
    agent = SubAgent(
        parent_ctx=ctx, subagent_type="explorer", prompt="hello",
        agent_id="ut_pt_01", max_steps=3, warn_steps=2, console=_CONSOLE,
        stop_mode="plain_text",
    )
    agent.build_tools()
    agent.build_messages()

    # Verify plain_text guide is present in system prompt
    system_msg = agent.ctx.messages[0]["content"]
    assert "submit your final result as plain text" in system_msg, \
        f"Expected plain_text guide in system prompt, got: {system_msg[:200]}"

    agent.ctx.llm_client = MagicMock()
    agent.ctx.llm_client.chat.completions.create.return_value = _fake_chat_completion(
        content="all done here",
    )

    result = agent.run()
    assert agent.status == AgentStatus.DONE
    assert result == "all done here"
    print("[PASS] 0.1.7a — plain_text content → DONE")


@patch("src.agent.subagent.OpenAI")
def test_plain_text_reasoning_fallback(mock_openai):
    """plain_text mode: LLM returns reasoning only -> DONE with reasoning text."""
    ctx = _make_empty_ctx()
    agent = SubAgent(
        parent_ctx=ctx, subagent_type="explorer", prompt="hello",
        agent_id="ut_pt_02", max_steps=3, warn_steps=2, console=_CONSOLE,
        stop_mode="plain_text",
    )
    agent.build_tools()
    agent.build_messages()
    agent.ctx.llm_client = MagicMock()

    resp = _fake_chat_completion(content=None, reasoning="thinking about it...")
    agent.ctx.llm_client.chat.completions.create.return_value = resp

    result = agent.run()
    assert agent.status == AgentStatus.DONE
    assert result == "thinking about it..."
    print("[PASS] 0.1.7b -- plain_text reasoning-only -> DONE")


# ---------------------------------------------------------------------------
# 0.1.8  stop_mode="schema"
# ---------------------------------------------------------------------------

def _noop_handler(args: dict, ctx: AgentContext) -> dict:
    return {"status": SUCCESS_LABEL, "found": len(args.get("path", ""))}


@patch("src.agent.subagent.OpenAI")
def test_schema_investigation_tool_no_terminate(mock_openai):
    """schema mode: investigation tool (with handler) runs, returns result, does NOT terminate."""
    ctx = _make_empty_ctx()
    inv_tool_schema = {
        "type": "function",
        "function": {
            "name": SUBAGENT_TOOL_INSPECT_XLSX,
            "description": "inspect a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "file path"},
                },
                "required": ["path"],
            },
        },
    }
    out_tool_schema = {
        "type": "function",
        "function": {
            "name": SUBAGENT_TOOL_REPORT_VERIFICATION,
            "description": "submit verification result",
            "parameters": {
                "type": "object",
                "properties": {
                    "overall_ok": {"type": "boolean"},
                },
                "required": ["overall_ok"],
            },
        },
    }

    agent = SubAgent(
        parent_ctx=ctx, subagent_type="explorer", prompt="verify data",
        agent_id="ut_sc_01", max_steps=5, warn_steps=2, console=_CONSOLE,
        stop_mode="schema",
        stop_schemas=[SUBAGENT_TOOL_REPORT_VERIFICATION],
        subagent_tools=[inv_tool_schema, out_tool_schema],
        tool_handlers={"inspect_xlsx": _noop_handler},
    )
    agent.build_tools()
    agent.build_messages()
    agent.ctx.llm_client = MagicMock()
    client = agent.ctx.llm_client.chat.completions
    client.create.return_value = _fake_chat_completion(
        tool_calls=[{
            "id": "call_inv_01",
            "type": "function",
            "function": {
                "name": SUBAGENT_TOOL_INSPECT_XLSX,
                "arguments": json.dumps({"path": "/tmp/test.xlsx"}),
            },
        }],
    )

    agent.run()

    assert agent.status == AgentStatus.ERROR  # max steps exhausted (schema mode without stop call)
    assert f"`{SUBAGENT_TOOL_REPORT_VERIFICATION}`" in (agent.error or "")
    # check that the investigation tool result was appended to messages
    msgs = agent.ctx.messages
    tool_msgs = [m for m in msgs if m["role"] == "tool" and m.get("content")]
    assert any(SUCCESS_LABEL in m["content"] for m in tool_msgs if isinstance(m.get("content"), str))
    print("[PASS] 0.1.8a — investigation tool (handler) → executed, not terminated")


@patch("src.agent.subagent.OpenAI")
def test_schema_output_tool_capture_terminate(mock_openai):
    """schema mode: output tool (no handler, in stop_schemas) → capture arguments → DONE."""
    ctx = _make_empty_ctx()
    out_tool_schema = {
        "type": "function",
        "function": {
            "name": SUBAGENT_TOOL_ADJUST_PARAMS,
            "description": "submit parameter adjustments",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {"type": "string"},
                    "adjustments": {"type": "object"},
                    "reasoning": {"type": "string"},
                },
                "required": ["section", "adjustments", "reasoning"],
            },
        },
    }

    agent = SubAgent(
        parent_ctx=ctx, subagent_type="explorer", prompt="tune params",
        agent_id="ut_sc_02", max_steps=5, warn_steps=2, console=_CONSOLE,
        stop_mode="schema",
        stop_schemas=[SUBAGENT_TOOL_ADJUST_PARAMS],
        subagent_tools=[out_tool_schema],
    )
    agent.build_tools()
    agent.build_messages()
    agent.ctx.llm_client = MagicMock()
    client = agent.ctx.llm_client.chat.completions

    client.create.return_value = _fake_chat_completion(
        tool_calls=[{
            "id": "call_adj_01",
            "type": "function",
            "function": {
                "name": SUBAGENT_TOOL_ADJUST_PARAMS,
                "arguments": json.dumps({
                    "section": "physical_fit",
                    "adjustments": {"max_nfev_scale": 3.0},
                    "reasoning": "need more convergence",
                }),
            },
        }],
    )

    result = agent.run()
    assert agent.status == AgentStatus.DONE, f"Expected DONE, got {agent.status.value}: {agent.error}"
    assert result is not None
    captured = json.loads(result)
    assert captured["section"] == "physical_fit"
    assert captured["adjustments"]["max_nfev_scale"] == 3.0
    print("[PASS] 0.1.8b — output tool (stop_schema) → captured → DONE")


@patch("src.agent.subagent.OpenAI")
def test_schema_plain_text_not_done(mock_openai):
    """schema mode: LLM returns pure text → reminder appended, does NOT stop."""
    ctx = _make_empty_ctx()
    out_tool_schema = {
        "type": "function",
        "function": {
            "name": SUBAGENT_TOOL_REPORT_VERIFICATION,
            "description": "submit report",
            "parameters": {
                "type": "object",
                "properties": {"overall_ok": {"type": "boolean"}},
                "required": ["overall_ok"],
            },
        },
    }

    agent = SubAgent(
        parent_ctx=ctx, subagent_type="explorer", prompt="verify",
        agent_id="ut_sc_03", max_steps=3, warn_steps=2, console=_CONSOLE,
        stop_mode="schema",
        stop_schemas=[SUBAGENT_TOOL_REPORT_VERIFICATION],
        subagent_tools=[out_tool_schema],
    )
    agent.build_tools()
    agent.build_messages()
    agent.ctx.llm_client = MagicMock()
    client = agent.ctx.llm_client.chat.completions

    # LLM returns pure text every time → reminders appended, never terminates as DONE
    client.create.return_value = _fake_chat_completion(content="I think the data looks fine.")

    result = agent.run()
    # Should hit max_steps → ERROR because stop_schema was never called
    assert agent.status == AgentStatus.ERROR, f"Expected ERROR, got {agent.status.value}"
    assert SUBAGENT_TOOL_REPORT_VERIFICATION in (agent.error or "")
    # Verify reminders were appended
    user_msgs = [m for m in agent.ctx.messages if m.get("role") == "user"]
    reminders = [m for m in user_msgs if "must call" in str(m.get("content", ""))]
    assert len(reminders) >= 1, f"Expected >= 1 reminder, got {len(reminders)}"
    print("[PASS] 0.1.8c — schema mode plain text → reminder, not DONE")


# ---------------------------------------------------------------------------
# 0.1.9  stop_mode="any"
# ---------------------------------------------------------------------------

@patch("src.agent.subagent.OpenAI")
def test_any_content_done(mock_openai):
    """any mode: LLM returns content → DONE (like plain_text)."""
    ctx = _make_empty_ctx()
    agent = SubAgent(
        parent_ctx=ctx, subagent_type="explorer", prompt="do stuff",
        agent_id="ut_an_01", max_steps=3, warn_steps=2, console=_CONSOLE,
        stop_mode="any",
    )
    agent.build_tools()
    agent.build_messages()
    agent.ctx.llm_client = MagicMock()
    agent.ctx.llm_client.chat.completions.create.return_value = _fake_chat_completion(
        content="done",
    )
    result = agent.run()
    assert agent.status == AgentStatus.DONE
    assert result == "done"
    print("[PASS] 0.1.9a — any mode content → DONE")


@patch("src.agent.subagent.OpenAI")
def test_any_reasoning_done(mock_openai):
    """any mode: LLM returns reasoning only -> DONE."""
    ctx = _make_empty_ctx()
    agent = SubAgent(
        parent_ctx=ctx, subagent_type="explorer", prompt="do stuff",
        agent_id="ut_an_02", max_steps=3, warn_steps=2, console=_CONSOLE,
        stop_mode="any",
    )
    agent.build_tools()
    agent.build_messages()
    agent.ctx.llm_client = MagicMock()

    resp = _fake_chat_completion(content=None, reasoning="analysis done")
    agent.ctx.llm_client.chat.completions.create.return_value = resp

    result = agent.run()
    assert agent.status == AgentStatus.DONE
    assert result == "analysis done"
    print("[PASS] 0.1.9b -- any mode reasoning -> DONE")


# ---------------------------------------------------------------------------
# 0.1.10  stop_mode="schema" max_steps exhaustion
# ---------------------------------------------------------------------------

@patch("src.agent.subagent.OpenAI")
def test_schema_exhausted_error(mock_openai):
    """schema mode: max_steps exhausted without calling stop_schema → ERROR."""
    ctx = _make_empty_ctx()
    out_tool_schema = {
        "type": "function",
        "function": {
            "name": SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS,
            "description": "request fallback",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["target", "reason"],
            },
        },
    }

    agent = SubAgent(
        parent_ctx=ctx, subagent_type="explorer", prompt="diagnose",
        agent_id="ut_ex_01", max_steps=2, warn_steps=1, console=_CONSOLE,
        stop_mode="schema",
        stop_schemas=[SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS],
        subagent_tools=[out_tool_schema],
    )
    agent.build_tools()
    agent.build_messages()
    agent.ctx.llm_client = MagicMock()
    client = agent.ctx.llm_client.chat.completions

    # LLM keeps returning pure text → never calls stop_schema
    client.create.return_value = _fake_chat_completion(content="analyzing...")

    result = agent.run()
    assert agent.status == AgentStatus.ERROR, f"Expected ERROR, got {agent.status.value}"
    assert SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS in agent.error
    assert result is None
    print("[PASS] 0.1.10 — schema max_steps exhausted → ERROR")


# ---------------------------------------------------------------------------
# get_tools() integration tests
# ---------------------------------------------------------------------------

def test_get_tools_fit_physical():
    """get_tools('fit-physical') returns schemas + handlers."""
    schemas, handlers = get_tools("fit-physical")
    assert len(schemas) == 4
    schema_names = {s["function"]["name"] for s in schemas}
    assert schema_names == {
        SUBAGENT_TOOL_INSPECT_XLSX,
        SUBAGENT_TOOL_ADJUST_PARAMS,
        SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS,
        SUBAGENT_TOOL_ACCEPT_QUALITY,
    }
    assert SUBAGENT_TOOL_INSPECT_XLSX in handlers
    assert SUBAGENT_TOOL_ADJUST_PARAMS not in handlers
    assert SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS not in handlers
    print("[PASS] get_tools fit-physical — 4 schemas, 1 handler")


def test_get_tools_check_data():
    """get_tools('check-data') has fallback_to_previous only, no handlers."""
    schemas, handlers = get_tools("check-data")
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS
    assert len(handlers) == 0
    print("[PASS] get_tools check-data — 1 schema, 0 handlers")


def test_get_tools_unknown_stage():
    """get_tools on unknown stage returns empty."""

    schemas, handlers = get_tools("nonexistent")
    assert schemas == []
    assert handlers == {}
    print("[PASS] get_tools unknown — empty fallback")


def test_get_tools_train_static():
    """get_tools('train-static') returns adjust + fallback + accept_quality."""
    schemas, handlers = get_tools("train-static")
    schema_names = {s["function"]["name"] for s in schemas}
    assert schema_names == {
        SUBAGENT_TOOL_INSPECT_XLSX,
        SUBAGENT_TOOL_ADJUST_PARAMS,
        SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS,
        SUBAGENT_TOOL_ACCEPT_QUALITY,
    }
    assert SUBAGENT_TOOL_INSPECT_XLSX in handlers
    print("[PASS] get_tools train-static — 4 schemas")


def test_get_tools_train_hysteresis():
    """get_tools('train-hysteresis') returns adjust + fallback + accept_quality."""
    schemas, handlers = get_tools("train-hysteresis")
    schema_names = {s["function"]["name"] for s in schemas}
    assert schema_names == {
        SUBAGENT_TOOL_INSPECT_XLSX,
        SUBAGENT_TOOL_ADJUST_PARAMS,
        SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS,
        SUBAGENT_TOOL_ACCEPT_QUALITY,
    }
    print("[PASS] get_tools train-hysteresis — 4 schemas")


def test_get_tools_final_interpret():
    """get_tools('final-interpret') returns contribute_knowledge without handler (stop_schema tool)."""
    schemas, handlers = get_tools("final-interpret")
    schema_names = {s["function"]["name"] for s in schemas}
    assert schema_names == {SUBAGENT_TOOL_CONTRIBUTE_KNOWLEDGE}
    assert SUBAGENT_TOOL_CONTRIBUTE_KNOWLEDGE not in handlers
    print("[PASS] get_tools final-interpret — 1 stop_schema tool")


def test_get_tools_adjust_params_override():
    """get_tools with adjust_params_override replaces the static schema."""
    custom = {"type": "function", "function": {"name": SUBAGENT_TOOL_ADJUST_PARAMS, "description": "Custom"}}
    schemas, handlers = get_tools("fit-physical", adjust_params_override=custom)
    schema_names = {s["function"]["name"] for s in schemas}
    assert SUBAGENT_TOOL_ADJUST_PARAMS in schema_names
    adjust_schema = next(s for s in schemas if s["function"]["name"] == SUBAGENT_TOOL_ADJUST_PARAMS)
    assert adjust_schema["function"]["description"] == "Custom"
    print("[PASS] get_tools override — custom adjust_params injected")


def test_get_tools_adjust_params_override_new_stage():
    """get_tools with override on stage without adjust_params appends it."""
    custom = {"type": "function", "function": {"name": SUBAGENT_TOOL_ADJUST_PARAMS, "description": "Auto"}}
    schemas, handlers = get_tools("check-data", adjust_params_override=custom)
    schema_names = {s["function"]["name"] for s in schemas}
    assert SUBAGENT_TOOL_ADJUST_PARAMS in schema_names
    print("[PASS] get_tools override — adjust_params appended to check-data")


def test_contribute_knowledge_handler_writes_file():
    """contribute_knowledge handler writes to knowledge/_discovered/ with slugified title."""
    import tempfile
    import os
    import json
    from src.tool.subagent_tools import _contribute_knowledge_handler

    orig_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:

            class FakeCtx:
                run_id = 999

            result = _contribute_knowledge_handler(
                {"title": "Test Discovery", "content": "# Finding\nSome interesting pattern."},
                FakeCtx(),
            )
            assert result["status"] == "written"
            assert "test_discovery_" in result["file"]
            assert result["title"] == "Test Discovery"
            assert os.path.exists(result["file"])

            with open(result["file"], "r", encoding="utf-8") as f:
                text = f.read()
            assert "# Test Discovery" in text
            assert "Some interesting pattern" in text

        finally:
            os.chdir(orig_cwd)

    print("[PASS] contribute_knowledge handler — slugified filename, file written")


@patch("src.agent.subagent.OpenAI")
def test_get_tools_routing_investigation(mock_openai):
    """Full routing through get_tools: investigation tool — handler — continue."""

    ctx = _make_empty_ctx()
    schemas, handlers = get_tools("fit-physical")

    agent = SubAgent(
        parent_ctx=ctx, subagent_type="explorer", prompt="inspect file",
        agent_id="ut_gt_01", max_steps=3, warn_steps=2, console=_CONSOLE,
        stop_mode="schema",
        stop_schemas=[SUBAGENT_TOOL_ADJUST_PARAMS, SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS],
        subagent_tools=schemas,
        tool_handlers=handlers,
    )
    agent.build_tools()
    agent.build_messages()
    agent.ctx.llm_client = MagicMock()
    client = agent.ctx.llm_client.chat.completions

    client.create.return_value = _fake_chat_completion(
        tool_calls=[{
            "id": "call_inv_01",
            "type": "function",
            "function": {
                "name": SUBAGENT_TOOL_INSPECT_XLSX,
                "arguments": json.dumps({"path": "/tmp/test.xlsx"}),
            },
        }],
    )
    agent.run()

    assert agent.status == AgentStatus.ERROR
    assert SUBAGENT_TOOL_ADJUST_PARAMS in agent.error
    msgs = agent.ctx.messages
    tool_msgs = [m for m in msgs if m["role"] == "tool" and isinstance(m.get("content"), str)]
    assert len(tool_msgs) >= 3
    print("[PASS] get_tools routing — investigation tool via get_tools() — handler — continue")


@patch("src.agent.subagent.OpenAI")
def test_get_tools_routing_output_tool(mock_openai):
    """Full routing through get_tools: output tool — capture — DONE."""

    ctx = _make_empty_ctx()
    schemas, handlers = get_tools("fit-physical")

    agent = SubAgent(
        parent_ctx=ctx, subagent_type="explorer", prompt="adjust",
        agent_id="ut_gt_02", max_steps=5, warn_steps=2, console=_CONSOLE,
        stop_mode="schema",
        stop_schemas=[SUBAGENT_TOOL_ADJUST_PARAMS, SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS],
        subagent_tools=schemas,
        tool_handlers=handlers,
    )
    agent.build_tools()
    agent.build_messages()
    agent.ctx.llm_client = MagicMock()
    client = agent.ctx.llm_client.chat.completions

    client.create.return_value = _fake_chat_completion(
        tool_calls=[{
            "id": "call_adj_01",
            "type": "function",
            "function": {
                "name": SUBAGENT_TOOL_ADJUST_PARAMS,
                "arguments": json.dumps({
                    "section": "physical_fit",
                    "adjustments": {"max_nfev_scale": 3.0},
                    "reasoning": "need more convergence",
                }),
            },
        }],
    )

    result = agent.run()
    assert agent.status == AgentStatus.DONE
    captured = json.loads(result)
    assert captured["section"] == "physical_fit"
    print("[PASS] get_tools routing — output tool via get_tools() — captured — DONE")


@patch("src.agent.subagent.OpenAI")
def test_get_tools_no_handler_not_stop_schema(mock_openai):
    """Tool from get_tools() with no handler and NOT in stop_schemas — FAIL to LLM."""

    ctx = _make_empty_ctx()
    schemas, handlers = get_tools("fit-physical")

    agent = SubAgent(
        parent_ctx=ctx, subagent_type="explorer", prompt="check",
        agent_id="ut_gt_03", max_steps=3, warn_steps=2, console=_CONSOLE,
        stop_mode="schema",
        stop_schemas=[SUBAGENT_TOOL_ADJUST_PARAMS],
        subagent_tools=schemas,
        tool_handlers=handlers,
    )
    agent.build_tools()
    agent.build_messages()
    agent.ctx.llm_client = MagicMock()
    agent.ctx.llm_client.chat.completions.create.return_value = _fake_chat_completion(
        tool_calls=[{
            "id": "call_fb_01",
            "type": "function",
            "function": {
                "name": SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS,
                "arguments": json.dumps({"target": "check-data", "reason": "bad data"}),
            },
        }],
    )

    agent.run()
    assert agent.status == AgentStatus.ERROR
    fail_msgs = [m for m in agent.ctx.messages
                 if m["role"] == "tool" and FAIL_LABEL in str(m.get("content", ""))]
    assert len(fail_msgs) >= 1, f"Expected FAIL message for misconfigured tool, got none"
    print("[PASS] get_tools routing — no-handler non-stop tool — FAIL to LLM")


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------

def test_stop_mode_validation():
    """Validate stop_mode and stop_schemas consistency at init time."""
    ctx = _make_empty_ctx()

    # schema mode without stop_schemas → ValueError
    try:
        SubAgent(
            parent_ctx=ctx, subagent_type="explorer", prompt="x",
            agent_id="ut_val_01", stop_mode="schema", console=_CONSOLE,
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # unknown stop_mode → ValueError
    try:
        SubAgent(
            parent_ctx=ctx, subagent_type="explorer", prompt="x",
            agent_id="ut_val_02", stop_mode="invalid", console=_CONSOLE,
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # valid schema with stop_schemas → ok
    agent = SubAgent(
        parent_ctx=ctx, subagent_type="explorer", prompt="x",
        agent_id="ut_val_03", stop_mode="schema",
        stop_schemas=["tool_a"], console=_CONSOLE,
    )
    assert agent.stop_mode == "schema"
    assert agent.stop_schemas == ["tool_a"]

    # defaults
    agent2 = SubAgent(
        parent_ctx=ctx, subagent_type="explorer", prompt="x",
        agent_id="ut_val_04", console=_CONSOLE,
    )
    assert agent2.stop_mode == "plain_text"
    assert agent2.stop_schemas == []
    assert agent2._subagent_tool_schemas == []
    assert agent2._tool_handlers == {}

    print("[PASS] 0.1.1b — stop_mode validation at init")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Validation (no mocks needed)
    test_stop_mode_validation()

    # plain_text
    test_plain_text_content_done()
    test_plain_text_reasoning_fallback()

    # schema
    test_schema_investigation_tool_no_terminate()
    test_schema_output_tool_capture_terminate()
    test_schema_plain_text_not_done()

    # any
    test_any_content_done()
    test_any_reasoning_done()

    # exhaustion
    test_schema_exhausted_error()

    # get_tools integration
    test_get_tools_fit_physical()
    test_get_tools_check_data()
    test_get_tools_unknown_stage()
    test_get_tools_routing_investigation()
    test_get_tools_routing_output_tool()
    test_get_tools_no_handler_not_stop_schema()

    print("\n=== All SubAgent unit tests passed! ===")

    # Cleanup test artifacts
    _td = os.path.join(SESSION_PATH, "test-unit-001")
    if os.path.exists(_td):
        shutil.rmtree(_td, ignore_errors=True)
