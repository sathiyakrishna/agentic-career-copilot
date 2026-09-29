import streamlit as st
from dotenv import load_dotenv

# RAG layer
from rag.candidate_knowledge import (
    resume_exists,
    save_master_resume,
    get_resume_hash,
    load_master_resume,
    index_resume,
    retrieve_resume_evidence,
)

# Agent layer
from agents.fit_agent import run_fit_agent

from agents.resume_agent import (
    run_resume_agent,
    create_resume_docx,
)


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="Agentic Career Copilot",
    page_icon="💼",
    layout="wide",
)


# =========================================================
# HEADER
# =========================================================

st.title("💼 Agentic Career Copilot")

st.caption(
    "Discover → Evaluate → Tailor → "
    "Apply → Track → Prepare"
)

st.divider()


# =========================================================
# CANDIDATE KNOWLEDGE BASE
# =========================================================

st.subheader("Candidate Knowledge Base")


if resume_exists():

    resume_version = get_resume_hash()

    resume_text = load_master_resume()

    st.success("✓ Master resume is active")

    st.caption(
        f"Resume version ID: {resume_version}"
    )

    st.write(
        f"Resume contains approximately "
        f"**{len(resume_text.split())} words**."
    )

    with st.expander(
        "Preview extracted resume"
    ):
        st.text(
            resume_text[:4000]
        )

    replace_resume = st.file_uploader(
        "Replace master resume",
        type=["pdf"],
        key="replace_resume",
    )

    if replace_resume:

        save_master_resume(
            replace_resume
        )

        # Clear previous analysis because
        # candidate evidence has changed
        for key in [
            "fit_result",
            "evidence",
            "job_description",
            "resume_text",
            "tailored_resume",
        ]:
            st.session_state.pop(
                key,
                None
            )

        st.success(
            "Master resume replaced successfully."
        )

        st.rerun()


else:

    uploaded_resume = st.file_uploader(
        "Upload your master resume",
        type=["pdf"],
    )

    if uploaded_resume:

        save_master_resume(
            uploaded_resume
        )

        st.success(
            "Master resume saved."
        )

        st.rerun()


st.divider()


# =========================================================
# JOB DESCRIPTION
# =========================================================

st.subheader("Job Description")

job_description = st.text_area(
    "Paste the job description",
    height=300,
    placeholder=(
        "Paste the complete job description here..."
    ),
)


# =========================================================
# ANALYZE JOB
# =========================================================

if st.button(
    "Analyze Job Fit",
    type="primary",
):

    if not resume_exists():

        st.warning(
            "Upload your master resume first."
        )

        st.stop()

    if not job_description.strip():

        st.warning(
            "Paste a job description first."
        )

        st.stop()


    # -----------------------------------------------------
    # Load persistent candidate profile
    # -----------------------------------------------------

    resume_text = load_master_resume()

    resume_version = get_resume_hash()


    # -----------------------------------------------------
    # Build / load RAG index
    # -----------------------------------------------------

    with st.spinner(
        "Loading Candidate Knowledge Base..."
    ):

        chunk_count = index_resume(
            resume_text,
            resume_version,
        )


    # -----------------------------------------------------
    # Retrieve relevant candidate evidence
    # -----------------------------------------------------

    with st.spinner(
        "Retrieving relevant candidate evidence..."
    ):

        evidence = retrieve_resume_evidence(
            job_description,
            top_k=6,
        )


    if not evidence:

        st.error(
            "No resume evidence could be retrieved."
        )

        st.stop()


    # -----------------------------------------------------
    # Fit Agent
    # -----------------------------------------------------

    with st.spinner(
        "Fit Agent is analyzing the role..."
    ):

        try:

            fit_result = run_fit_agent(
                job_description,
                evidence,
            )

        except Exception as error:

            st.error(
                f"Fit Agent error: {error}"
            )

            st.stop()


    # -----------------------------------------------------
    # Save state
    # -----------------------------------------------------

    st.session_state[
        "fit_result"
    ] = fit_result

    st.session_state[
        "evidence"
    ] = evidence

    st.session_state[
        "job_description"
    ] = job_description

    st.session_state[
        "resume_text"
    ] = resume_text

    st.session_state[
        "chunk_count"
    ] = chunk_count

    # New JD = old tailored resume invalid
    st.session_state.pop(
        "tailored_resume",
        None
    )


# =========================================================
# DISPLAY FIT RESULTS
# =========================================================

if "fit_result" in st.session_state:

    fit_result = st.session_state[
        "fit_result"
    ]

    evidence = st.session_state[
        "evidence"
    ]

    st.divider()

    st.header("Job Fit Analysis")


    # -----------------------------------------------------
    # Score
    # -----------------------------------------------------

    score = fit_result.get(
        "fit_score",
        0
    )

    col1, col2 = st.columns(
        [1, 3]
    )

    with col1:

        st.metric(
            "Overall Fit",
            f"{score}%"
        )

    with col2:

        st.progress(
            max(
                0,
                min(
                    score / 100,
                    1
                )
            )
        )

        st.write(
            fit_result.get(
                "summary",
                ""
            )
        )


    # -----------------------------------------------------
    # Strong Matches
    # -----------------------------------------------------

    st.subheader(
        "✅ Strong Matches"
    )

    strong_matches = fit_result.get(
        "strong_matches",
        []
    )

    if strong_matches:

        for match in strong_matches:

            st.markdown(
                f"**{match.get('requirement', '')}**"
            )

            st.write(
                match.get(
                    "reason",
                    ""
                )
            )

            with st.expander(
                "View supporting evidence"
            ):

                st.write(
                    match.get(
                        "evidence",
                        ""
                    )
                )

    else:

        st.write(
            "No strong matches identified."
        )


    # -----------------------------------------------------
    # Partial Matches
    # -----------------------------------------------------

    st.subheader(
        "🟡 Partial Matches"
    )

    partial_matches = fit_result.get(
        "partial_matches",
        []
    )

    if partial_matches:

        for match in partial_matches:

            st.markdown(
                f"**{match.get('requirement', '')}**"
            )

            st.write(
                match.get(
                    "reason",
                    ""
                )
            )

            with st.expander(
                "View supporting evidence"
            ):

                st.write(
                    match.get(
                        "evidence",
                        ""
                    )
                )

    else:

        st.write(
            "No partial matches identified."
        )


    # -----------------------------------------------------
    # Gaps
    # -----------------------------------------------------

    st.subheader(
        "🔴 Gaps"
    )

    gaps = fit_result.get(
        "gaps",
        []
    )

    if gaps:

        for gap in gaps:

            st.markdown(
                f"**{gap.get('requirement', '')}**"
            )

            st.write(
                gap.get(
                    "reason",
                    ""
                )
            )

    else:

        st.write(
            "No major gaps identified."
        )


    # -----------------------------------------------------
    # Fit Agent Summary
    # -----------------------------------------------------

    st.subheader(
        "Fit Agent Summary"
    )

    st.info(
        fit_result.get(
            "recommendation",
            ""
        )
    )


    # -----------------------------------------------------
    # RAG transparency
    # -----------------------------------------------------

    with st.expander(
        "🔎 RAG Evidence Retrieved"
    ):

        st.caption(
            f"Resume version: "
            f"{get_resume_hash()}"
        )

        st.caption(
            f"{st.session_state.get('chunk_count', 0)} "
            f"resume chunks indexed"
        )

        for number, chunk in enumerate(
            evidence,
            start=1
        ):

            st.markdown(
                f"**Evidence {number}**"
            )

            st.write(chunk)

            st.divider()


# =========================================================
# RESUME AGENT
# =========================================================

if (
    "fit_result" in st.session_state
    and
    "evidence" in st.session_state
):

    st.divider()

    st.header(
        "Tailored Resume"
    )

    st.caption(
        "Evidence-locked tailoring: existing experience "
        "can be prioritized or rephrased, but unsupported "
        "experience cannot be added."
    )

    if st.button(
        "Generate Tailored Resume"
    ):

        with st.spinner(
            "Resume Agent is tailoring your resume..."
        ):

            try:

                tailored_resume = (
                    run_resume_agent(
                        st.session_state[
                            "job_description"
                        ],
                        st.session_state[
                            "resume_text"
                        ],
                        st.session_state[
                            "evidence"
                        ],
                        st.session_state[
                            "fit_result"
                        ],
                    )
                )

                st.session_state[
                    "tailored_resume"
                ] = tailored_resume

            except Exception as error:

                st.error(
                    f"Resume Agent error: {error}"
                )

                st.stop()


# =========================================================
# DISPLAY TAILORED RESUME
# =========================================================

if "tailored_resume" in st.session_state:

    tailored_resume = st.session_state[
        "tailored_resume"
    ]

    st.success(
        "Tailored resume generated."
    )


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    st.subheader(
        "Professional Summary"
    )

    st.write(
        tailored_resume.get(
            "professional_summary",
            ""
        )
    )


    # -----------------------------------------------------
    # Skills
    # -----------------------------------------------------

    st.subheader(
        "Skills"
    )

    skills = tailored_resume.get(
        "skills",
        []
    )

    if skills:

        st.write(
            " • ".join(skills)
        )


    # -----------------------------------------------------
    # Experience
    # -----------------------------------------------------

    st.subheader(
        "Professional Experience"
    )

    for experience in tailored_resume.get(
        "experience",
        []
    ):

        role = experience.get(
            "role",
            ""
        )

        company = experience.get(
            "company",
            ""
        )

        st.markdown(
            f"### {role} — {company}"
        )

        dates = experience.get(
            "dates",
            ""
        )

        if dates:

            st.caption(dates)

        for bullet in experience.get(
            "bullets",
            []
        ):

            st.write(
                f"• {bullet}"
            )


    # -----------------------------------------------------
    # Projects
    # -----------------------------------------------------

    projects = tailored_resume.get(
        "projects",
        []
    )

    if projects:

        st.subheader(
            "Projects"
        )

        for project in projects:

            st.markdown(
                f"### "
                f"{project.get('name', '')}"
            )

            for bullet in project.get(
                "bullets",
                []
            ):

                st.write(
                    f"• {bullet}"
                )


    # -----------------------------------------------------
    # Unsupported requirements
    # -----------------------------------------------------

    unsupported = tailored_resume.get(
        "unsupported_requirements",
        []
    )

    if unsupported:

        with st.expander(
            "⚠️ JD requirements not added to resume"
        ):

            st.caption(
                "No supporting evidence was found "
                "in the candidate knowledge base."
            )

            for item in unsupported:

                st.write(
                    f"• {item}"
                )


    # -----------------------------------------------------
    # DOCX
    # -----------------------------------------------------

    docx_file = create_resume_docx(
        tailored_resume
    )

    st.download_button(
        label=(
            "⬇️ Download Tailored Resume (.docx)"
        ),
        data=docx_file.getvalue(),
        file_name="tailored_resume.docx",
        mime=(
            "application/vnd.openxmlformats-"
            "officedocument.wordprocessingml.document"
        ),
    )