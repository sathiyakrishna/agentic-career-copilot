import os

from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


load_dotenv()


# =========================================================
# MODEL CLIENT
# =========================================================

def create_model_client():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing from .env"
        )

    return OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key,
    )


# =========================================================
# CAREER ORCHESTRATOR
# =========================================================

def create_career_orchestrator():

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

1. SEARCH
   Find relevant job opportunities.

2. FIT
   Compare a job description with the candidate's
   verified resume evidence.

3. RESEARCH
   Research a company, role, industry, or hiring context.

4. RESUME
   Tailor the candidate's resume for a specific role.

5. APPLICATION
   Prepare information required for a job application.

6. TRACKER
   Track submitted job applications and their status.

7. INTERVIEW
   Generate role-specific interview preparation.

Do not fabricate candidate experience.

Do not perform specialist work yourself.

Your job is routing and orchestration.

When given a request, identify the appropriate
capability and explain briefly why it should handle
the request.
"""
    )

    return orchestrator