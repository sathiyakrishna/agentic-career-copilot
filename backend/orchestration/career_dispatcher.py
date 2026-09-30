from backend.orchestration.career_orchestrator import (
    route_request,
)

from backend.orchestration.career_workflow import (
    execute_fit_workflow,
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

    V0 currently executes the FIT capability.
    Other capabilities will be added incrementally.
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
    # Capabilities not implemented yet
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