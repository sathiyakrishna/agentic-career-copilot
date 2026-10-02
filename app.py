import asyncio
import io
import streamlit as st
from dotenv import load_dotenv

from rag.candidate_knowledge import (
    resume_exists,
    save_master_resume,
    get_resume_hash,
    load_master_resume,
    index_resume,
)

from agents.search_agent import run_search_agent
from agents.resume_agent import create_resume_docx

from backend.orchestration.career_workflow import (
    execute_fit_workflow,
    execute_resume_workflow,
)

from backend.orchestration.career_dispatcher import (
    dispatch_career_request,
)


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="Jobnext.ai | Agentic Career Copilot",
    page_icon="💼",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "job_description": "",
    "fit_result": None,
    "resume_result": None,
    "job_search_response": None,
    "copilot_response": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
# ============================================================

def run_async(coro):
    """
    Safely execute an async coroutine from Streamlit.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def show_result(data):
    """
    Display dictionaries, lists or plain text safely.
    """
    if data is None:
        return

    if isinstance(data, (dict, list)):
        st.json(data)
    else:
        st.write(data)


def extract_search_payload(response):
    """
    Normalize the dispatcher/search-agent response.

    Expected possibilities include:
        {"result": {"live_search": {...}}}
        {"live_search": {...}}
        {"jobs": [...]}
    """
    if not isinstance(response, dict):
        return response

    result = response.get("result", response)

    if isinstance(result, dict):
        return result.get("live_search", result)

    return result


def display_jobs(search_response):
    """
    Render search results as readable job cards.
    """
    payload = extract_search_payload(search_response)

    if not isinstance(payload, dict):
        show_result(payload)
        return

    jobs = payload.get("jobs", [])

    if not jobs:
        st.info("No structured job results were returned.")
        show_result(payload)
        return

    result_count = payload.get("result_count", len(jobs))

    st.success(f"Found {result_count} job results.")

    for index, job in enumerate(jobs, start=1):

        if not isinstance(job, dict):
            st.write(job)
            continue

        title = job.get("title") or "Job Opportunity"
        url = job.get("url") or job.get("link") or ""
        snippet = job.get("snippet") or ""
        source = job.get("source") or "Web"

        with st.container(border=True):

            st.markdown(f"### {index}. {title}")

            if snippet:
                st.write(snippet)

            st.caption(f"Source: {source}")

            if url:
                st.link_button(
                    "View Job →",
                    url,
                )


def build_docx_download(resume_result):
    """
    Try to create the DOCX using the existing resume-agent helper.
    """
    try:
        docx_result = create_resume_docx(resume_result)

        if docx_result is None:
            return None

        if isinstance(docx_result, bytes):
            return docx_result

        if isinstance(docx_result, bytearray):
            return bytes(docx_result)

        if isinstance(docx_result, io.BytesIO):
            return docx_result.getvalue()

        if hasattr(docx_result, "getvalue"):
            return docx_result.getvalue()

        if hasattr(docx_result, "save"):
            buffer = io.BytesIO()
            docx_result.save(buffer)
            buffer.seek(0)
            return buffer.getvalue()

        return None

    except Exception:
        return None


# ============================================================
# HEADER
# ============================================================

st.title("💼 Jobnext.ai")

st.subheader("Agentic Career Copilot")

st.caption(
    "Discover → Evaluate → Tailor → Apply → Track → Prepare"
)

st.divider()


# ============================================================
# CANDIDATE KNOWLEDGE BASE
# ============================================================

st.header("1. Candidate Knowledge Base")

if resume_exists():

    st.success("✓ Master resume is active")

    try:
        resume_hash = get_resume_hash()

        if resume_hash:
            st.caption(f"Resume version ID: {resume_hash}")

    except Exception:
        pass

    try:
        master_resume = load_master_resume()

        if master_resume:
            word_count = len(master_resume.split())

            st.write(
                f"Resume contains approximately "
                f"**{word_count} words**."
            )

            with st.expander("Preview extracted resume"):
                st.text(master_resume)

    except Exception as error:
        st.warning(
            f"Resume exists, but preview could not be loaded: {error}"
        )

    upload_label = "Replace master resume"

else:

    st.warning(
        "No master resume found. Upload your resume to activate "
        "the Candidate Knowledge Base."
    )

    upload_label = "Upload master resume"


uploaded_resume = st.file_uploader(
    upload_label,
    type=["pdf"],
    key="master_resume_upload",
)


if uploaded_resume is not None:

    if st.button(
        "Save Master Resume",
        type="primary",
        key="save_master_resume_button",
    ):

        try:

            with st.spinner(
                "Building Candidate Knowledge Base..."
            ):

                save_master_resume(uploaded_resume)

                resume_text = load_master_resume()
                resume_version = get_resume_hash()

                if resume_text and resume_version:
                    index_resume(
                        resume_text,
                        resume_version,
                    )

            st.success(
                "Master resume saved and indexed successfully."
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"Resume upload error: {error}"
            )


st.divider()


# ============================================================
# DISCOVER JOBS
# ============================================================

st.header("2. Discover Jobs")

job_search_request = st.text_input(
    "What jobs are you looking for?",
    placeholder=(
        "Example: AI Product Manager jobs in Chennai"
    ),
    key="job_search_request",
)


if st.button(
    "Discover Jobs",
    type="primary",
    key="discover_jobs_button",
):

    if not job_search_request.strip():

        st.warning(
            "Enter the role, skill or location you want to search."
        )

    elif not resume_exists():

        st.warning(
            "Upload your master resume first."
        )

    else:

        try:

            with st.spinner(
                "Search Agent is discovering opportunities..."
            ):

                resume_text = load_master_resume()

                search_response = run_search_agent(
                    job_search_request.strip(),
                    resume_text,
                )

                st.session_state[
                    "job_search_response"
                ] = search_response

        except Exception as error:

            st.error(
                f"Job search error: {error}"
            )


if st.session_state["job_search_response"] is not None:

    display_jobs(
        st.session_state["job_search_response"]
    )


st.divider()


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.header("3. Evaluate Job Fit")

job_description = st.text_area(
    "Paste the complete job description",
    value=st.session_state["job_description"],
    height=300,
    placeholder=(
        "Paste the complete job description here..."
    ),
    key="job_description_input",
)


# ============================================================
# FIT ANALYSIS
# ============================================================

if st.button(
    "Analyze Job Fit",
    type="primary",
    key="analyze_fit_button",
):

    if not job_description.strip():

        st.warning(
            "Paste a job description first."
        )

    elif not resume_exists():

        st.warning(
            "Upload your master resume first."
        )

    else:

        try:

            with st.spinner(
                "Fit Agent is evaluating your profile..."
            ):

                fit_result = execute_fit_workflow(
                    job_description.strip()
                )

                st.session_state[
                    "job_description"
                ] = job_description.strip()

                st.session_state[
                    "fit_result"
                ] = fit_result

                st.session_state[
                    "resume_result"
                ] = None

            st.success(
                "Job-fit analysis completed."
            )

        except Exception as error:

            st.error(
                f"Fit analysis error: {error}"
            )


if st.session_state["fit_result"] is not None:

    st.subheader("Fit Analysis")

    show_result(
        st.session_state["fit_result"]
    )


st.divider()


# ============================================================
# TAILORED RESUME
# ============================================================

st.header("4. Tailor Resume")

st.write(
    "Generate a role-specific resume using your master resume, "
    "retrieved evidence and job-fit analysis."
)


if st.button(
    "Generate Tailored Resume",
    type="primary",
    key="generate_resume_button",
):

    active_jd = job_description.strip()

    if not active_jd:

        st.warning(
            "Paste a job description first."
        )

    elif not resume_exists():

        st.warning(
            "Upload your master resume first."
        )

    else:

        try:

            with st.spinner(
                "Resume Agent is tailoring your resume..."
            ):

                resume_result = execute_resume_workflow(
                    active_jd
                )

                st.session_state[
                    "job_description"
                ] = active_jd

                st.session_state[
                    "resume_result"
                ] = resume_result

            st.success(
                "Tailored resume generated successfully."
            )

        except Exception as error:

            st.error(
                f"Resume workflow error: {error}"
            )


if st.session_state["resume_result"] is not None:

    st.subheader("Tailored Resume")

    resume_workflow_result = st.session_state[
        "resume_result"
    ]

    if isinstance(resume_workflow_result, dict):

        tailored_resume = resume_workflow_result.get(
            "tailored_resume",
            resume_workflow_result,
        )

    else:

        tailored_resume = resume_workflow_result

    show_result(tailored_resume)

    docx_bytes = build_docx_download(
        tailored_resume
    )

    if docx_bytes:

        st.download_button(
            label="Download Tailored Resume (.docx)",
            data=docx_bytes,
            file_name="jobnext_tailored_resume.docx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            key="download_tailored_resume",
        )


st.divider()


# ============================================================
# AGENTIC CAREER COPILOT
# ============================================================

st.header("5. Agentic Career Copilot")

st.write(
    "Ask Jobnext.ai to decide which career capability should "
    "handle your request."
)

copilot_request = st.text_area(
    "What would you like Jobnext.ai to do?",
    height=120,
    placeholder=(
        "Examples:\n"
        "Find AI Product Manager roles in Chennai\n"
        "Evaluate my fit for this role\n"
        "Tailor my resume for this job"
    ),
    key="copilot_request",
)


if st.button(
    "Run Career Copilot",
    type="primary",
    key="run_copilot_button",
):

    if not copilot_request.strip():

        st.warning(
            "Enter a career request first."
        )

    elif not resume_exists():

        st.warning(
            "Upload your master resume first."
        )

    else:

        try:

            with st.spinner(
                "Career Orchestrator is routing your request..."
            ):

                current_jd = (
                    job_description.strip()
                    if job_description.strip()
                    else None
                )

                copilot_response = run_async(
                    dispatch_career_request(
                        user_request=copilot_request.strip(),
                        job_description=current_jd,
                    )
                )

                st.session_state[
                    "copilot_response"
                ] = copilot_response

            st.success(
                "Career request completed."
            )

        except Exception as error:

            st.error(
                f"Career Copilot error: {error}"
            )


if st.session_state["copilot_response"] is not None:

    response = st.session_state[
        "copilot_response"
    ]

    if isinstance(response, dict):

        routed_to = response.get("routed_to")

        if routed_to:
            st.info(
                f"Routed to: {routed_to}"
            )

        routing_response = response.get(
            "routing_response"
        )

        if routing_response:
            with st.expander(
                "Orchestrator reasoning"
            ):
                st.write(routing_response)

        copilot_result = response.get(
            "result",
            response,
        )

        if routed_to == "SEARCH":
            display_jobs(response)
        else:
            show_result(copilot_result)

    else:

        show_result(response)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Jobnext.ai • Agentic AI Career Platform • "
    "RAG + AI Agents + AutoGen"
)
