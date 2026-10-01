import json

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI()


# =========================================================
# SEARCH AGENT
# =========================================================

def run_search_agent(
    user_request: str,
    resume_text: str,
):
    """
    Convert a natural-language job search request
    into structured search criteria using the
    candidate's master resume as context.
    """

    system_prompt = """
You are the Search Agent for Jobnext,
an AI Career Copilot.

Your job is to convert the user's job-search request
into structured search criteria.

Use the candidate's resume only as supporting context.

Do not invent candidate experience, skills,
certifications, employers, or qualifications.

Return ONLY valid JSON.

Use exactly this structure:

{
    "target_roles": [],
    "keywords": [],
    "locations": [],
    "experience_level": "",
    "work_modes": [],
    "domains": [],
    "search_query": ""
}

SEARCH QUERY RULES:

1. Create a concise job-search query.
2. Prioritize the user's explicit request.
3. Use resume context only when useful.
4. Do not add unsupported qualifications.
5. Keep the query suitable for a job-search provider.
"""

    user_prompt = f"""
USER REQUEST:

{user_request}


CANDIDATE MASTER RESUME:

{resume_text}


Create structured job-search criteria.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={
            "type": "json_object"
        },
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    result_text = (
        response
        .choices[0]
        .message
        .content
    )

    return json.loads(result_text)