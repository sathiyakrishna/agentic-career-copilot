from rag.candidate_knowledge import (
    resume_exists,
    get_resume_hash,
    load_master_resume,
    index_resume,
    retrieve_resume_evidence,
)

from agents.fit_agent import run_fit_agent


# =========================================================
# FIT WORKFLOW
# =========================================================

def execute_fit_workflow(
    job_description: str
):
    """
    Execute the existing Jobnext FIT capability.

    Flow:
    Resume
        ↓
    RAG
        ↓
    Evidence
        ↓
    Fit Agent
        ↓
    Structured analysis
    """

    job_description = (
        job_description.strip()
    )

    if not job_description:
        raise ValueError(
            "Job description cannot be empty."
        )

    if not resume_exists():
        raise ValueError(
            "No master resume is available."
        )

    # ---------------------------------------------
    # Load candidate knowledge
    # ---------------------------------------------

    resume_text = load_master_resume()

    resume_version = get_resume_hash()

    # ---------------------------------------------
    # Ensure RAG index exists
    # ---------------------------------------------

    chunk_count = index_resume(
        resume_text,
        resume_version
    )

    # ---------------------------------------------
    # Retrieve relevant evidence
    # ---------------------------------------------

    evidence = retrieve_resume_evidence(
        job_description,
        top_k=6
    )

    if not evidence:
        raise ValueError(
            "No relevant resume evidence found."
        )

    # ---------------------------------------------
    # Existing Fit Agent
    # ---------------------------------------------

    fit_result = run_fit_agent(
        job_description,
        evidence
    )

    # ---------------------------------------------
    # Workflow response
    # ---------------------------------------------

    return {
        "capability": "FIT",

        "resume_version":
            resume_version,

        "chunks_indexed":
            chunk_count,

        "evidence_count":
            len(evidence),

        "fit_analysis":
            fit_result,

        "retrieved_evidence":
            evidence,
    }