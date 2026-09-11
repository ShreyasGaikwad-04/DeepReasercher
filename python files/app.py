import os
import tempfile
import streamlit as st

from rag import create_retriever,process_paper
from graph import build_research_graph


st.set_page_config(
    page_title="Deep Researcher",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Deep Researcher")
st.caption(
    "Upload a research paper and ask questions using multi-agent RAG + web research."
)


# ---------------------------------------------------------
# PDF upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a research paper",
    type=["pdf"]
)


# ---------------------------------------------------------
# Question
# ---------------------------------------------------------

question = st.text_area(
    "What would you like to research?",
    placeholder="Compare the approach in this paper with recent alternatives..."
)


# ---------------------------------------------------------
# Run research
# ---------------------------------------------------------

if uploaded_file and question and st.button("Research"):

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(uploaded_file.getbuffer())
        pdf_path = tmp.name

    try:

        # ---------------------------------------------
        # Create RAG retriever
        # ---------------------------------------------

        with st.status("Preparing research paper...", expanded=True) as status:

            st.write("Creating document embeddings...")

            retriever, paper_summary = process_paper(pdf_path)

            st.write("Research paper indexed.")


            # ---------------------------------------------
            # Create graph
            # ---------------------------------------------

            graph = build_research_graph(retriever)

            initial_state = {
                "question": question,
                "paper_summary": paper_summary,
                "paper_research": [],
                "web_research": [],
                "research_feedback":
                    "No previous research has been performed.",
                "iteration": 0,
                "max_iterations": 2
            }

            final_answer = ""

            # ---------------------------------------------
            # Stream graph progress
            # ---------------------------------------------

            for event in graph.stream(
                initial_state,
                stream_mode="updates"
            ):

                for node_name, output in event.items():

                    if node_name == "planner":

                        st.write("🧠 Research plan created")

                        for task in output["plan"]:
                            st.write(
                                f"- **{task['source']}**: "
                                f"{task['task']}"
                            )

                    elif node_name == "paper_researcher":

                        st.write(
                            "📄 Researching uploaded paper..."
                        )

                    elif node_name == "web_researcher":

                        st.write(
                            "🌐 Searching external sources..."
                        )

                    elif node_name == "critic_writer":

                        if output.get("final_answer"):

                            final_answer = output["final_answer"]

                            st.write(
                                "✅ Evidence evaluated"
                            )

                        else:

                            st.write(
                                "🔁 More research required"
                            )

                            st.write(
                                output.get(
                                    "research_feedback",
                                    ""
                                )
                            )

            status.update(
                label="Research complete",
                state="complete"
            )


        # ---------------------------------------------
        # Final answer
        # ---------------------------------------------

        st.subheader("Research Result")

        st.markdown(final_answer)


    finally:

        # Delete temporary PDF
        if os.path.exists(pdf_path):
            os.remove(pdf_path)