import os

from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# MODEL CLIENT
# =========================================================

def create_model_client():
    """
    Create the OpenAI model client used by AutoGen.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing from .env"
        )

    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key,
    )

    return model_client


# =========================================================
# CAREER ORCHESTRATOR
# =========================================================

def create_career_orchestrator():
    """
    Create the Jobnext.ai AutoGen career orchestrator.

    The orchestrator identifies which specialist
    capability should handle the user's request.
    """

    model_client = create_model_client()

    orchestrator = AssistantAgent(
        name="career_orchestrator",

        model_client=model_client,

        system_message="""
You are the Career Orchestrator for Jobnext.ai.

Your responsibility is to understand what the user
wants to accomplish in their job search and determine
which specialist capability should handle the request.

Available capabilities:

SEARCH
Find relevant job opportunities.

FIT
Compare a job description with the candidate's
verified resume evidence.

RESEARCH
Research a company, role, industry, or hiring context.

RESUME
Tailor the candidate's resume for a specific role.

APPLICATION
Prepare information required for a job application.

TRACKER
Track submitted job applications and their status.

INTERVIEW
Generate role-specific interview preparation.


ROUTING RULES:

If the user wants to find jobs:
SEARCH

If the user wants to know whether they match a job:
FIT

If the user wants information about a company or role:
RESEARCH

If the user wants to tailor or modify their resume:
RESUME

If the user wants help completing a job application:
APPLICATION

If the user wants to record or check an application:
TRACKER

If the user wants interview questions or preparation:
INTERVIEW


IMPORTANT RULES:

1. Do not fabricate candidate experience.

2. Do not perform specialist work yourself.

3. Your job is routing and orchestration.

4. Select the single most appropriate capability.

5. Return the capability name first.

6. After the capability name, provide one short sentence
   explaining why the request was routed there.


Example:

FIT
The user wants to compare their profile with the
requirements of a specific job.
"""
    )

    return orchestrator, model_client


# =========================================================
# ROUTE USER REQUEST
# =========================================================

async def route_request(user_request: str):
    """
    Route a user request to the appropriate
    Jobnext.ai specialist capability.
    """

    if not user_request.strip():
        raise ValueError(
            "User request cannot be empty."
        )

    orchestrator, model_client = (
        create_career_orchestrator()
    )

    try:

        result = await orchestrator.run(
            task=user_request
        )

        if not result.messages:
            raise RuntimeError(
                "AutoGen returned no routing response."
            )

        final_message = result.messages[-1]

        return final_message.content

    finally:

        await model_client.close()