import operator

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END

from agents import (
    planner_agent,
    paper_researcher,
    web_researcher,
    critic_writer_agent,
)


# ============================================================
# GRAPH STATE
# ============================================================

class DeepResearchState(TypedDict, total=False):

    # User input
    question: str
    paper_summary: str

    # Current research plan
    plan: List[dict]

    # Accumulated research
    paper_research: Annotated[List[dict], operator.add]
    web_research: Annotated[List[dict], operator.add]

    # Feedback from critic for another research iteration
    research_feedback: str

    # Critic output
    critic_result: dict

    # Final output
    final_answer: str

    # Prevent infinite research loops
    iteration: int
    max_iterations: int


# ============================================================
# GRAPH BUILDER
# ============================================================

def build_research_graph(retriever):

    # --------------------------------------------------------
    # PLANNER NODE
    # --------------------------------------------------------

    def planner_node(state: DeepResearchState):

        feedback = state.get(
            "research_feedback",
            "No previous research has been performed."
        )

        plan = planner_agent(
            question=state["question"],
            paper_summary=state["paper_summary"],
            research_feedback=feedback
        )

        tasks = [
            task.model_dump()
            for task in plan.tasks
        ]

        return {
            "plan": tasks,
            "iteration": state.get("iteration", 0) + 1
        }


    # --------------------------------------------------------
    # PAPER RESEARCH NODE
    # --------------------------------------------------------

    def paper_research_node(state: DeepResearchState):

        paper_tasks = [
            task["task"]
            for task in state["plan"]
            if task["source"] == "paper"
        ]

        results = []

        for task in paper_tasks:

            result = paper_researcher(
                task=task,
                retriever=retriever
            )

            results.append(result)

        return {
            "paper_research": results
        }


    # --------------------------------------------------------
    # WEB RESEARCH NODE
    # --------------------------------------------------------

    def web_research_node(state: DeepResearchState):

        web_tasks = [
            task["task"]
            for task in state["plan"]
            if task["source"] == "web"
        ]

        results = []

        for task in web_tasks:

            result = web_researcher(task)

            results.append(result)

        return {
            "web_research": results
        }


    # --------------------------------------------------------
    # CRITIC / WRITER NODE
    # --------------------------------------------------------

    def critic_writer_node(state: DeepResearchState):

        result = critic_writer_agent(
            question=state["question"],
            paper_research=state.get("paper_research", []),
            web_research=state.get("web_research", [])
        )

        return {
            "critic_result": result.model_dump(),
            "research_feedback": result.feedback,
            "final_answer": result.final_answer
        }


    # --------------------------------------------------------
    # CONDITIONAL ROUTING
    # --------------------------------------------------------

    def critic_router(state: DeepResearchState):

        critic_result = state["critic_result"]

        # Evidence is sufficient → finish
        if critic_result["sufficient"]:
            return "finish"

        # Prevent an infinite Planner → Research → Critic loop
        if state.get("iteration", 0) >= state.get("max_iterations", 3):
            return "finish"

        # Evidence insufficient → research again
        return "retry"


    # ========================================================
    # BUILD LANGGRAPH
    # ========================================================

    builder = StateGraph(DeepResearchState)

    builder.add_node(
        "planner",
        planner_node
    )

    builder.add_node(
        "paper_researcher",
        paper_research_node
    )

    builder.add_node(
        "web_researcher",
        web_research_node
    )

    builder.add_node(
        "critic_writer",
        critic_writer_node
    )


    # --------------------------------------------------------
    # EDGES
    # --------------------------------------------------------

    builder.add_edge(
        START,
        "planner"
    )

    # Planner sends work to both researcher branches.
    # Each researcher processes only tasks assigned to it.
    builder.add_edge(
        "planner",
        "paper_researcher"
    )

    builder.add_edge(
        "planner",
        "web_researcher"
    )

    # Both research branches converge on the critic.
    builder.add_edge(
        "paper_researcher",
        "critic_writer"
    )

    builder.add_edge(
        "web_researcher",
        "critic_writer"
    )

    # Critic decides whether to finish or research again.
    builder.add_conditional_edges(
        "critic_writer",
        critic_router,
        {
            "retry": "planner",
            "finish": END
        }
    )


    # --------------------------------------------------------
    # COMPILE
    # --------------------------------------------------------

    graph = builder.compile()

    return graph