from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from sqlalchemy.orm import Session

from investigator.config import Settings
from investigator.grounding import extract_json, ground_findings
from investigator.tools import build_tools
from investigator.vectorstore import PassageStore
from investigator.loop import ensure_event_loop


SYSTEM = """You investigate uploaded documents. You must use tools before answering.
Tools: list_library, search_documents, read_passage.
Do not invent quotes. When finished, respond with JSON only:
{"summary":"...","findings":[{"claim":"...","quote":"...","document":"...","chunk_id":"..."}]}
If the library does not support a claim, omit it. Prefer 2-6 findings.
"""


class AgentState(TypedDict):
    brief: str
    messages: Annotated[list, add_messages]


def _evidence_pool(messages: list) -> list[str]:
    pool: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            pool.append(str(message.content))
    return pool


def build_graph(settings: Settings, store: PassageStore, session: Session):
    tools = build_tools(store, session)
    model_name = settings.resolved_gemini_model
    try:
        model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.gemini_api_key,
            temperature=0.2,
            transport="rest",
        ).bind_tools(tools)
    except Exception:
        model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.gemini_api_key,
            temperature=0.2,
        ).bind_tools(tools)
    tool_node = ToolNode(tools)

    def agent(state: AgentState) -> dict:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

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
