"""Autonomous pipeline builder agent."""
from langgraph.graph import StateGraph, END
from langchain_google_vertexai import ChatVertexAI
from typing import TypedDict, List

class PipelineState(TypedDict):
    data_source: str
    schema_info: str
    architecture: str
    generated_code: str
    test_results: str
    deployed: bool
    messages: List[str]

llm = ChatVertexAI(model_name="gemini-1.5-pro-002", temperature=0.1)

def discover_schema(state: PipelineState) -> PipelineState:
    prompt = f"Analyze this data source and describe schema, volume estimates, data types, and quality issues: {state['data_source']}"
    schema = llm.invoke(prompt).content
    state["schema_info"] = schema
    state["messages"].append(f"Schema discovered: {len(schema)} chars")
    return state

def design_architecture(state: PipelineState) -> PipelineState:
    prompt = f"""Given schema: {state["schema_info"]}
Design optimal ETL architecture. Specify: tools (Airflow/Beam/dbt), pattern (incremental/full), schedule, transformations needed.
Return concise architecture description."""
    arch = llm.invoke(prompt).content
    state["architecture"] = arch
    return state

def generate_code(state: PipelineState) -> PipelineState:
    prompt = f"""Generate production Python ETL code for:
Schema: {state["schema_info"][:1000]}
Architecture: {state["architecture"]}
Include: Airflow DAG, transformations, error handling, logging, tests.
Return complete, runnable code."""
    code = llm.invoke(prompt).content
    if "```python" in code: code = code.split("```python")[1].split("```")[0]
    state["generated_code"] = code
    return state

def build_agent() -> StateGraph:
    graph = StateGraph(PipelineState)
    graph.add_node("discover", discover_schema)
    graph.add_node("design", design_architecture)
    graph.add_node("codegen", generate_code)
    graph.set_entry_point("discover")
    graph.add_edge("discover", "design")
    graph.add_edge("design", "codegen")
    graph.add_edge("codegen", END)
    return graph.compile()
