from __future__ import annotations

import json
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from openai import OpenAI
from sqlalchemy.orm import Session

from investigator.config import Settings
from investigator.grounding import extract_json, ground_findings
from investigator.loop import ensure_event_loop
from investigator.tools import build_tools
from investigator.vectorstore import PassageStore


SYSTEM = """You investigate uploaded documents. You must use tools before answering.
Tools: list_library, search_documents, read_passage.
Do not invent quotes. Copy exact verbatim substrings from the tool output into the "quote" field.
When finished, respond with JSON only:
{"summary":"...","findings":[{"claim":"...","quote":"...","document":"...","chunk_id":"..."}]}
If the library does not support a claim, omit it. Prefer 2-6 findings.
"""

FALLBACK_FREE_MODELS = [
    "openrouter/free",
    "qwen/qwen3.8-27b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-lightning:free",
]


class AgentState(TypedDict):
    brief: str
    messages: Annotated[list, add_messages]


def _evidence_pool(messages: list) -> list[str]:
    pool: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            pool.append(str(message.content))
    return pool


def _tool_schema(tool) -> dict:
    schema: dict[str, Any] = {"type": "object", "properties": {}}
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        return schema
    raw = args_schema.model_json_schema()
    properties = {}
    for key, value in (raw.get("properties") or {}).items():
        if not isinstance(value, dict):
            continue
        item = {
            field: value[field]
            for field in ("type", "description", "enum", "items")
            if field in value
        }
        item.setdefault("type", "string")
        properties[key] = item
    schema["properties"] = properties
    if raw.get("required"):
        schema["required"] = list(raw["required"])
    return schema


def _openai_tool(tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or tool.name,
            "parameters": _tool_schema(tool),
        },
    }


def _to_openai_messages(messages: list) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            formatted.append({"role": "system", "content": str(message.content)})
        elif isinstance(message, HumanMessage):
            formatted.append({"role": "user", "content": str(message.content)})
        elif isinstance(message, AIMessage):
            if message.tool_calls:
                calls = []
                for idx, call in enumerate(message.tool_calls):
                    name = call["name"] if isinstance(call, dict) else call.name
                    args = (
                        call.get("args", {})
                        if isinstance(call, dict)
                        else getattr(call, "args", {})
                    )
                    call_id = (
                        (call.get("id") if isinstance(call, dict) else getattr(call, "id", None))
                        or f"call_{name}_{idx}"
                    )
                    calls.append(
                        {
                            "id": str(call_id),
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args or {}),
                            },
                        }
                    )
                formatted.append(
                    {
                        "role": "assistant",
                        "content": str(message.content) if message.content else None,
                        "tool_calls": calls,
                    }
                )
            else:
                formatted.append({"role": "assistant", "content": str(message.content)})
        elif isinstance(message, ToolMessage):
            formatted.append(
                {
                    "role": "tool",
                    "tool_call_id": str(getattr(message, "tool_call_id", "") or "call_0"),
                    "name": message.name or "tool",
                    "content": str(message.content),
                }
            )
    return formatted


def _parse_tool_args(raw_args: Any) -> dict[str, Any]:
    if not raw_args:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def build_graph(settings: Settings, store: PassageStore, session: Session):
    tools = build_tools(store, session)
    openai_tools = [_openai_tool(tool) for tool in tools]
    tool_node = ToolNode(tools)

    api_key = settings.openrouter_api_key.strip()
    client = OpenAI(
        base_url=settings.openrouter_base_url.strip() or "https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    primary_model = settings.resolved_openrouter_model
    candidate_models = [primary_model] + [
        m for m in FALLBACK_FREE_MODELS if m != primary_model
    ]

    def agent(state: AgentState) -> dict:
        ensure_event_loop()
        openai_messages = _to_openai_messages(state["messages"])
        has_tool_history = any(isinstance(m, ToolMessage) for m in state["messages"])

        last_err: Exception | None = None
        response = None
        for idx, model_name in enumerate(candidate_models):
            fallback_slice = candidate_models[idx : idx + 3]
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=openai_messages,
                    tools=openai_tools,
                    tool_choice="auto",
                    temperature=0.2,
                    extra_body={"models": fallback_slice},
                )
                if response and getattr(response, "choices", None):
                    break
            except Exception as exc:
                last_err = exc
                continue

        if response is None or not getattr(response, "choices", None):
            raise RuntimeError(f"OpenRouter request failed: {last_err or 'No choices returned.'}")

        msg = response.choices[0].message
        content_text = (getattr(msg, "content", None) or "").strip()
        raw_tool_calls = getattr(msg, "tool_calls", None) or []

        tool_calls = []
        for index, call in enumerate(raw_tool_calls):
            fn = getattr(call, "function", None)
            fn_name = getattr(fn, "name", None) if fn else None
            if not fn_name:
                continue
            tool_calls.append(
                {
                    "name": fn_name,
                    "args": _parse_tool_args(getattr(fn, "arguments", None)),
                    "id": getattr(call, "id", None) or f"{fn_name}_{index}",
                    "type": "tool_call",
                }
            )

        # If a free model skipped calling tools on the very first turn, trigger
        # search_documents + list_library automatically so Qdrant passages are retrieved.
        if not tool_calls and not has_tool_history:
            tool_calls = [
                {
                    "name": "list_library",
                    "args": {},
                    "id": "auto_list_library_0",
                    "type": "tool_call",
                },
                {
                    "name": "search_documents",
                    "args": {"query": state["brief"]},
                    "id": "auto_search_documents_1",
                    "type": "tool_call",
                },
            ]

        return {
            "messages": [AIMessage(content=content_text, tool_calls=tool_calls)],
        }

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "finalize"

    def finalize(state: AgentState) -> dict:
        return {}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "finalize": "finalize"})
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_investigation(
    settings: Settings,
    store: PassageStore,
    session: Session,
    brief: str,
) -> dict[str, Any]:
    if not settings.openrouter_api_key.strip():
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    ensure_event_loop()
    compiled = build_graph(settings, store, session)
    start = {
        "brief": brief,
        "messages": [
            SystemMessage(content=SYSTEM),
            HumanMessage(
                content=(
                    "Investigate this brief against the document library. Use tools.\n\n"
                    f"{brief}"
                )
            ),
        ],
    }
    result = compiled.invoke(start, {"recursion_limit": settings.agent_max_steps * 2})
    messages = result["messages"]
    trace = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                name = call["name"] if isinstance(call, dict) else call.name
                args = (
                    call.get("args", {})
                    if isinstance(call, dict)
                    else getattr(call, "args", {})
                )
                trace.append(f"{name}({args})")
        if isinstance(message, ToolMessage):
            preview = str(message.content).replace("\n", " ")[:180]
            trace.append(f"→ {preview}")

    last_ai = ""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
            last_ai = str(message.content)
            break

    try:
        payload = extract_json(last_ai) if last_ai else {"summary": "", "findings": []}
    except Exception:
        payload = {"summary": "Investigator did not return structured findings.", "findings": []}

    findings = ground_findings(payload.get("findings") or [], _evidence_pool(messages))
    return {
        "summary": payload.get("summary") or "",
        "findings": [item.model_dump() for item in findings],
        "trace": trace,
        "step_count": sum(1 for step in trace if not step.startswith("→")),
    }
