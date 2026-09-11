from rag import create_retriever
from graph import build_research_graph

# 1. Build retriever from your test paper
retriever = create_retriever("data/paper.pdf")

# 2. Build graph
graph = build_research_graph(retriever)

# 3. Temporary paper summary for testing
paper_summary = """
This paper proposes a NaCl-betaine enhanced chitosan-based conductive hydrogel
for wearable strain and pressure sensors. It focuses on improving mechanical
strength, conductivity, self-healing, antifreeze performance, and sensing ability.
"""

# 4. Test question
question = """
What are the main advantages of the hydrogel proposed in this paper,
and how does it compare with recent conductive hydrogels used for wearable sensors?
"""

# 5. Initial graph state
initial_state = {
    "question": question,
    "paper_summary": paper_summary,
    "paper_research": [],
    "web_research": [],
    "research_feedback": "No previous research has been performed.",
    "iteration": 0,
    "max_iterations": 3
}

# 6. Run graph
result = graph.invoke(initial_state)

print(result["final_answer"])