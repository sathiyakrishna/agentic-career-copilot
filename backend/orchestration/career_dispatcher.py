from backend.orchestration.career_orchestrator import (
    route_request,
)

from backend.orchestration.career_workflow import (
    execute_fit_workflow,
)

from rag.candidate_knowledge import (
    resume_exists,
    get_resume_hash,
    load_master_resume,
    index_resume,
    retrieve_resume_evidence,
)

from agents.fit_agent import (
    run_fit_agent,
)

from agents.resume_agent import (
    run_resume_agent,
)


# =========================================================
# CAREER DISPATCHER
# =========================================================

async def dispatch_career_request(
    user_request: str,
    job_description: str | None = None,
):
    """
    Route a user request with AutoGen and execute
    the appropriate Jobnext capability.

    V0 capabilities:
    - FIT
    - RESUME
    """

    user_request = user_request.strip()

    if not user_request:
        raise ValueError(
            "User request cannot be empty."
        )

    # -----------------------------------------------------
    # AutoGen routing
    # -----------------------------------------------------

    routing_response = await route_request(
        user_request
    )

    routing_text = str(
        routing_response
    ).strip()

    capability = (
        routing_text
        .splitlines()[0]
        .strip()
        .upper()
    )

    # -----------------------------------------------------
    # FIT
    # -----------------------------------------------------

    if capability == "FIT":

        if not job_description:
            raise ValueError(
                "A job description is required "
                "for FIT analysis."
            )

        fit_result = execute_fit_workflow(
            job_description
        )

        return {
            "routed_to": "FIT",
            "routing_response": routing_text,
            "result": fit_result,
        }

    # -----------------------------------------------------
    # RESUME
    # -----------------------------------------------------

    if capability == "RESUME":

        if not job_description:
            raise ValueError(
                "A job description is required "
                "for resume tailoring."
            )

        if not resume_exists():
            raise ValueError(
                "No master resume found. "
                "Upload a resume first."
            )

        # -------------------------------------------------
        # Load master resume
        # -------------------------------------------------

        resume_text = load_master_resume()

        if not resume_text.strip():
            raise ValueError(
                "The master resume contains no "
                "extractable text."
            )

        # -------------------------------------------------
        # Resume version
        # -------------------------------------------------

        resume_version = get_resume_hash()

        # -------------------------------------------------
        # Index candidate knowledge
        # -------------------------------------------------

        chunk_count = index_resume(
            resume_text,
            resume_version,
        )

        # -------------------------------------------------
        # Retrieve relevant resume evidence
        # -------------------------------------------------

        evidence = retrieve_resume_evidence(
            job_description,
            top_k=6,
        )

        if not evidence:
            raise ValueError(
                "No relevant resume evidence "
                "could be retrieved."
            )

        # -------------------------------------------------
        # Fit Agent
        # -------------------------------------------------

        fit_result = run_fit_agent(
            job_description,
            evidence,
        )

        # -------------------------------------------------
        # Resume Agent
        # -------------------------------------------------

        tailored_resume = run_resume_agent(
            job_description,
            resume_text,
            evidence,
            fit_result,
        )

        # -------------------------------------------------
        # Return result
        # -------------------------------------------------

        return {
            "routed_to": "RESUME",
            "routing_response": routing_text,
            "result": {
                "capability": "RESUME",
                "resume_version": resume_version,
                "chunks_indexed": chunk_count,
                "evidence_count": len(evidence),
                "fit_score": fit_result.get(
                    "fit_score",
                    0,
                ),
                "tailored_resume": tailored_resume,
                "unsupported_requirements":
                    tailored_resume.get(
                        "unsupported_requirements",
                        [],
                    ),
            },
        }

    # -----------------------------------------------------
    # CAPABILITIES NOT IMPLEMENTED YET
    # -----------------------------------------------------

    return {
        "routed_to": capability,
        "routing_response": routing_text,
        "status": "NOT_IMPLEMENTED",
        "message": (
            f"{capability} routing works, "
            "but execution has not been "
            "implemented yet."
        ),
    }