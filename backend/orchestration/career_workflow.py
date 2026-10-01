from rag.candidate_knowledge import (
    resume_exists,
    get_resume_hash,
    load_master_resume,
    index_resume,
    retrieve_resume_evidence,
)

from agents.fit_agent import run_fit_agent
from agents.resume_agent import run_resume_agent


# =========================================================
# FIT WORKFLOW
# =========================================================

def execute_fit_workflow(job_description: str):

    job_description = job_description.strip()

    if not job_description:
        raise ValueError("Job description cannot be empty.")

    if not resume_exists():
        raise ValueError("No master resume is available.")

    resume_text = load_master_resume()
    resume_version = get_resume_hash()

    chunk_count = index_resume(
        resume_text,
        resume_version
    )

    evidence = retrieve_resume_evidence(
        job_description,
        top_k=6
    )

    if not evidence:
        raise ValueError(
            "No relevant resume evidence found."
        )

    fit_result = run_fit_agent(
        job_description,
        evidence
    )

    return {
        "capability": "FIT",
        "resume_version": resume_version,
        "chunks_indexed": chunk_count,
        "evidence_count": len(evidence),
        "fit_analysis": fit_result,
        "retrieved_evidence": evidence,
    }


# =========================================================
# RESUME WORKFLOW
# =========================================================

def execute_resume_workflow(job_description: str):

    job_description = job_description.strip()

    if not job_description:
        raise ValueError("Job description cannot be empty.")

    if not resume_exists():
        raise ValueError("No master resume is available.")

    resume_text = load_master_resume()
    resume_version = get_resume_hash()

    chunk_count = index_resume(
        resume_text,
        resume_version
    )

    evidence = retrieve_resume_evidence(
        job_description,
        top_k=6
    )

    if not evidence:
        raise ValueError(
            "No relevant resume evidence found."
        )

    fit_result = run_fit_agent(
        job_description,
        evidence
    )

    tailored_resume = run_resume_agent(
        job_description=job_description,
        resume_text=resume_text,
        evidence=evidence,
        fit_result=fit_result,
    )

    return {
        "capability": "RESUME",
        "resume_version": resume_version,
        "chunks_indexed": chunk_count,
        "evidence_count": len(evidence),
        "fit_analysis": fit_result,
        "tailored_resume": tailored_resume,
    }
