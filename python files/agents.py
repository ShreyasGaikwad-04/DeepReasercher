from typing import List, Literal

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from rag import retrieve_context
from tools import web_search

from prompts import (
    planner_prompt,
    paper_researcher_prompt,
    web_researcher_prompt,
    critic_writer_prompt,
)

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash"
)


# ============================================================
# STRUCTURED OUTPUT MODELS
# ============================================================

class ResearchTask(BaseModel):
    task: str = Field(
        description="A specific research task that needs to be completed."
    )

    source: Literal["paper", "web"] = Field(
        description="The source that should be used for this task."
    )


class ResearchPlan(BaseModel):
    tasks: List[ResearchTask] = Field(
        description="Research tasks required to answer the user's question."
    )


class CriticResult(BaseModel):
    sufficient: bool = Field(
        description="Whether the collected evidence is sufficient."
    )

    feedback: str = Field(
        description="Specific missing information if more research is needed."
    )

    final_answer: str = Field(
        description="Final answer if evidence is sufficient, otherwise empty."
    )


# ============================================================
# STRUCTURED LLMs
# ============================================================

planner_llm = llm.with_structured_output(ResearchPlan)

critic_llm = llm.with_structured_output(CriticResult)


# ============================================================
# PLANNER AGENT
# ============================================================

def planner_agent(
    question: str,
    paper_summary: str,
    research_feedback: str = "No previous research has been performed."
) -> ResearchPlan:

    prompt = planner_prompt.invoke({
        "question": question,
        "paper_summary": paper_summary,
        "research_feedback": research_feedback
    })

    return planner_llm.invoke(prompt)


# ============================================================
# PAPER RESEARCHER AGENT
# ============================================================

def paper_researcher(
    task: str,
    retriever
) -> dict:

    # rag.py performs retrieval + combines top-k chunks
    context = retrieve_context(
        task=task,
        retriever=retriever
    )

    prompt = paper_researcher_prompt.invoke({
        "task": task,
        "context": context
    })

    response = llm.invoke(prompt)

    return {
        "task": task,
        "source": "paper",
        "research": response.content
    }


# ============================================================
# WEB RESEARCHER AGENT
# ============================================================

def web_researcher(task: str) -> dict:

    search_results = web_search.invoke({
        "query": task
    })

    prompt = web_researcher_prompt.invoke({
        "task": task,
        "search_results": search_results
    })

    response = llm.invoke(prompt)

    return {
        "task": task,
        "source": "web",
        "research": response.content,
        "raw_results": search_results
    }


# ============================================================
# CRITIC / WRITER AGENT
# ============================================================

def critic_writer_agent(
    question: str,
    paper_research: List[dict],
    web_research: List[dict]
) -> CriticResult:

    # Convert accumulated paper research into text
    paper_text = "\n\n".join(
        f"""
Task: {item["task"]}
Research:
{item["research"]}
"""
        for item in paper_research
    )

    # Convert accumulated web research into text
    web_text = "\n\n".join(
        f"""
Task: {item["task"]}
Research:
{item["research"]}
"""
        for item in web_research
    )

    # Handle cases where one source was not needed
    if not paper_text:
        paper_text = "No paper research was required."

    if not web_text:
        web_text = "No web research was required."

    prompt = critic_writer_prompt.invoke({
        "question": question,
        "paper_research": paper_text,
        "web_research": web_text
    })

    return critic_llm.invoke(prompt)