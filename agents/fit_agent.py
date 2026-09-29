import json

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI()


def run_fit_agent(
    job_description,
    evidence
):
    """
    Evaluate job fit using only verified
    candidate evidence retrieved from RAG.
    """

    evidence_text = "\n\n".join(
        [
            f"EVIDENCE {i + 1}:\n{chunk}"
            for i, chunk in enumerate(evidence)
        ]
    )

    system_prompt = """
You are the Fit Agent for an AI Career Copilot.

Evaluate how well VERIFIED candidate resume evidence
matches a job description.

STRICT RULES:

1. Use ONLY candidate evidence supplied to you.
2. Never invent experience.
3. Never invent skills.
4. Never invent certifications.
5. Never invent achievements or metrics.
6. Never assume experience merely because two concepts
   are semantically related.
7. Unsupported requirements must be classified as GAP.

Classify requirements as:

STRONG MATCH
PARTIAL MATCH
GAP

Return ONLY valid JSON.

Use exactly this structure:

{
  "fit_score": 0,
  "summary": "",
  "strong_matches": [
    {
      "requirement": "",
      "evidence": "",
      "reason": ""
    }
  ],
  "partial_matches": [
    {
      "requirement": "",
      "evidence": "",
      "reason": ""
    }
  ],
  "gaps": [
    {
      "requirement": "",
      "reason": ""
    }
  ],
  "recommendation": ""
}

fit_score must be an integer from 0 to 100.
"""

    user_prompt = f"""
JOB DESCRIPTION:

{job_description}


VERIFIED CANDIDATE EVIDENCE:

{evidence_text}


Evaluate the candidate's fit for this job.

Do not assume unsupported experience.
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
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    result_text = (
        response
        .choices[0]
        .message
        .content
    )

    return json.loads(result_text)