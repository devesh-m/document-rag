from __future__ import annotations

from typing import Annotated, Any, TypedDict

from google import genai
from google.genai import types
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from sqlalchemy.orm import Session

from investigator.config import Settings
from investigator.grounding import extract_json, ground_findings
from investigator.loop import ensure_event_loop
from investigator.tools import build_tools
from investigator.vectorstore import PassageStore


SYSTEM = """You investigate uploaded documents. You must use tools before answering.
Tools: list_library, search_documents, read_passage.
Do not invent quotes. When finished, respond with JSON only:
{"summary":"...","findings":[{"claim":"...","quote":"...","document":"...","chunk_id":"..."}]}
If the library does not support a claim, omit it. Prefer 2-6 findings.
"""


class AgentState(TypedDict):
    brief: str
    messages: Annotated[list, add_messages]
    contents: list


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
        item = {field: value[field] for field in ("type", "description", "enum", "items") if field in value}
        item.setdefault("type", "string")
        properties[key] = item
    schema["properties"] = properties
    if raw.get("required"):
        schema["required"] = list(raw["required"])
    return schema


def _function_declaration(tool) -> types.FunctionDeclaration:
    schema = _tool_schema(tool)
    try:
        return types.FunctionDeclaration(
            name=tool.name,
            description=tool.description or tool.name,
            parameters=schema,
        )
    except Exception:
        return types.FunctionDeclaration(
            name=tool.name,
            description=tool.description or tool.name,
            parameters_json_schema=schema,
        )


def _function_args(call) -> dict:
    raw = getattr(call, "args", None)
    if not raw:
        return {}
    try:
        return dict(raw)
    except Exception:
        return {}


def _pending_tool_messages(messages: list) -> list[ToolMessage]:
    pending: list[ToolMessage] = []
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            pending.append(message)
            continue
        break
    pending.reverse()
    return pending


def _generate_config(tools) -> types.GenerateContentConfig:
    declarations = [_function_declaration(tool) for tool in tools]
    kwargs: dict[str, Any] = {
        "system_instruction": SYSTEM,
        "tools": [types.Tool(function_declarations=declarations)],
        "temperature": 0.2,
    }
    try:
        kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
    except Exception:
        pass
    return types.GenerateContentConfig(**kwargs)


def build_graph(settings: Settings, store: PassageStore, session: Session):
    tools = build_tools(store, session)
    model_name = settings.resolved_gemini_model
    client = genai.Client(api_key=settings.gemini_api_key)
    config = _generate_config(tools)
    tool_node = ToolNode(tools)

    def agent(state: AgentState) -> dict:
        ensure_event_loop()
        contents = list(state.get("contents") or [])
        if not contents:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=(
                                "Investigate this brief against the document library. Use tools.\n\n"
                                f"{state['brief']}"
                            )
                        )
                    ],
                )
            ]
        else:
            pending = _pending_tool_messages(state["messages"])
            if pending:
                parts = [
                    types.Part.from_function_response(
                        name=message.name or "tool",
                        response={"result": str(message.content)},
                    )
                    for message in pending
                ]
                contents.append(types.Content(role="user", parts=parts))
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc
        candidates = getattr(response, "candidates", None) or []
        if not candidates or not candidates[0].content:
            raise RuntimeError("Gemini returned no content.")
        model_content = candidates[0].content
        contents = contents + [model_content]
        tool_calls = []
        text_bits: list[str] = []
        for index, part in enumerate(model_content.parts or []):
            call = getattr(part, "function_call", None)
            if call and getattr(call, "name", None):
                tool_calls.append(
                    {
                        "name": call.name,
                        "args": _function_args(call),
                        "id": getattr(call, "id", None) or f"{call.name}_{index}",
                        "type": "tool_call",
                    }
                )
                continue
            text = getattr(part, "text", None)
            if text and not getattr(part, "thought", False):
                text_bits.append(text)
        return {
            "messages": [AIMessage(content="\n".join(text_bits), tool_calls=tool_calls)],
            "contents": contents,
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
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    ensure_event_loop()
    compiled = build_graph(settings, store, session)
    start = {
        "brief": brief,
        "contents": [],
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
                args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})
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
