import os
import requests

from dotenv import load_dotenv


load_dotenv()


# =========================================================
# JOB SEARCH TOOL
# =========================================================

SERPER_URL = "https://google.serper.dev/search"


def search_jobs(
    search_query: str,
    num_results: int = 10,
):
    """
    Search the web for current job opportunities
    using Serper.
    """

    search_query = search_query.strip()

    if not search_query:
        raise ValueError(
            "Search query cannot be empty."
        )

    api_key = os.getenv("SERPER_API_KEY")

    if not api_key:
        raise ValueError(
            "SERPER_API_KEY is not configured."
        )

    # Keep P0 search simple and stable.
    query = (
        f"{search_query} "
        f"jobs careers hiring"
    )

    response = requests.post(
        SERPER_URL,
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        },
        json={
            "q": query,
            "num": num_results,
        },
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    results = []

    for item in data.get("organic", []):
        title = item.get("title", "")
        link = item.get("link", "")
        snippet = item.get("snippet", "")

        if not title or not link:
            continue

        results.append(
            {
                "title": title,
                "url": link,
                "snippet": snippet,
                "source": "serper",
            }
        )

    return {
        "query": query,
        "result_count": len(results),
        "jobs": results,
    }
