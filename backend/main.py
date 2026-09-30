from pathlib import Path
from io import BytesIO
import shutil

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
)

from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# =========================================================
# RAG
# =========================================================

from rag.candidate_knowledge import (
    resume_exists,
    get_resume_hash,
    load_master_resume,
    index_resume,
    retrieve_resume_evidence,
)

# =========================================================
# AGENTS
# =========================================================

from agents.fit_agent import run_fit_agent

from agents.resume_agent import (
    run_resume_agent,
    create_resume_docx,
)

# =========================================================
# AUTOGEN ORCHESTRATION
# =========================================================

from backend.orchestration.career_dispatcher import (
    dispatch_career_request,
)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Jobnext.ai API",
    description=(
        "Agentic AI Career Copilot API powered by "
        "RAG, specialist agents and AutoGen orchestration."
    ),
    version="0.4.0",
)


# =========================================================
# REQUEST MODELS
# =========================================================

class JobAnalysisRequest(BaseModel):
    job_description: str


class ResumeTailorRequest(BaseModel):
    job_description: str


class CareerRequest(BaseModel):
    user_request: str
    job_description: str | None = None


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "product": "Jobnext.ai",
        "version": "0.4.0",
        "status": "running",
        "architecture": "Agentic Career Copilot",
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }


# =========================================================
# RESUME STATUS
# =========================================================

@app.get("/api/resume")
def get_resume_status():

    if not resume_exists():

        return {
            "resume_active": False
        }

    try:

        resume_text = load_master_resume()

        return {
            "resume_active": True,
            "resume_version": get_resume_hash(),
            "word_count": len(
                resume_text.split()
            ),
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =========================================================
# RESUME UPLOAD
# =========================================================

@app.post("/api/resume/upload")
async def upload_resume(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # Validate file
    # -----------------------------------------------------

    if file.content_type != "application/pdf":

        raise HTTPException(
            status_code=400,
            detail="Only PDF resumes are supported."
        )

    # -----------------------------------------------------
    # Resume storage path
    # -----------------------------------------------------

    resume_path = Path(
        "data/resume/master_resume.pdf"
    )

    resume_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        # -------------------------------------------------
        # Save PDF
        # -------------------------------------------------

        with open(
            resume_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # -------------------------------------------------
        # Extract resume
        # -------------------------------------------------

        resume_text = load_master_resume()

        if not resume_text.strip():

            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not extract text "
                    "from the uploaded PDF."
                )
            )

        # -------------------------------------------------
        # Resume version
        # -------------------------------------------------

        resume_version = get_resume_hash()

        # -------------------------------------------------
        # RAG indexing
        # -------------------------------------------------

        chunk_count = index_resume(
            resume_text,
            resume_version
        )

        return {
            "message":
                "Master resume uploaded successfully.",

            "resume_version":
                resume_version,

            "word_count":
                len(resume_text.split()),

            "chunks_indexed":
                chunk_count,
        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        await file.close()


# =========================================================
# JOB FIT ANALYSIS
# =========================================================

@app.post("/api/jobs/analyze")
def analyze_job(
    request: JobAnalysisRequest
):

    job_description = (
        request.job_description.strip()
    )

    if not job_description:

        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty."
        )

    if not resume_exists():

        raise HTTPException(
            status_code=400,
            detail=(
                "No master resume found. "
                "Upload a resume first."
            )
        )

    try:

        # -------------------------------------------------
        # Candidate knowledge
        # -------------------------------------------------

        resume_text = load_master_resume()

        resume_version = get_resume_hash()

        chunk_count = index_resume(
            resume_text,
            resume_version
        )

        # -------------------------------------------------
        # RAG evidence
        # -------------------------------------------------

        evidence = retrieve_resume_evidence(
            job_description,
            top_k=6
        )

        if not evidence:

            raise HTTPException(
                status_code=404,
                detail=(
                    "No relevant resume evidence "
                    "could be retrieved."
                )
            )

        # -------------------------------------------------
        # Fit Agent
        # -------------------------------------------------

        fit_result = run_fit_agent(
            job_description,
            evidence
        )

        return {
            "resume_version":
                resume_version,

            "chunks_indexed":
                chunk_count,

            "evidence_count":
                len(evidence),

            "fit_analysis":
                fit_result,

            "retrieved_evidence":
                evidence,
        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =========================================================
# TAILORED RESUME — JSON
# =========================================================

@app.post("/api/resume/tailor")
def tailor_resume(
    request: ResumeTailorRequest
):

    job_description = (
        request.job_description.strip()
    )

    if not job_description:

        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty."
        )

    if not resume_exists():

        raise HTTPException(
            status_code=400,
            detail=(
                "No master resume found. "
                "Upload a resume first."
            )
        )

    try:

        # -------------------------------------------------
        # Candidate knowledge
        # -------------------------------------------------

        resume_text = load_master_resume()

        resume_version = get_resume_hash()

        index_resume(
            resume_text,
            resume_version
        )

        # -------------------------------------------------
        # RAG
        # -------------------------------------------------

        evidence = retrieve_resume_evidence(
            job_description,
            top_k=6
        )

        if not evidence:

            raise HTTPException(
                status_code=404,
                detail=(
                    "No relevant resume evidence "
                    "could be retrieved."
                )
            )

        # -------------------------------------------------
        # Fit Agent
        # -------------------------------------------------

        fit_result = run_fit_agent(
            job_description,
            evidence
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

        return {
            "resume_version":
                resume_version,

            "fit_score":
                fit_result.get(
                    "fit_score",
                    0
                ),

            "tailored_resume":
                tailored_resume,

            "unsupported_requirements":
                tailored_resume.get(
                    "unsupported_requirements",
                    []
                ),
        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =========================================================
# DOWNLOAD TAILORED RESUME
# =========================================================

@app.post("/api/resume/tailor/download")
def download_tailored_resume(
    request: ResumeTailorRequest
):

    job_description = (
        request.job_description.strip()
    )

    if not job_description:

        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty."
        )

    if not resume_exists():

        raise HTTPException(
            status_code=400,
            detail="No master resume found."
        )

    try:

        # -------------------------------------------------
        # Candidate knowledge
        # -------------------------------------------------

        resume_text = load_master_resume()

        resume_version = get_resume_hash()

        index_resume(
            resume_text,
            resume_version
        )

        # -------------------------------------------------
        # RAG evidence
        # -------------------------------------------------

        evidence = retrieve_resume_evidence(
            job_description,
            top_k=6
        )

        if not evidence:

            raise HTTPException(
                status_code=404,
                detail=(
                    "No relevant resume evidence "
                    "could be retrieved."
                )
            )

        # -------------------------------------------------
        # Fit Agent
        # -------------------------------------------------

        fit_result = run_fit_agent(
            job_description,
            evidence
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
        # DOCX
        # -------------------------------------------------

        docx_file = create_resume_docx(
            tailored_resume
        )

        docx_bytes = docx_file.getvalue()

        return StreamingResponse(
            BytesIO(docx_bytes),

            media_type=(
                "application/"
                "vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),

            headers={
                "Content-Disposition":
                    'attachment; filename="tailored_resume.docx"'
            },
        )

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =========================================================
# AUTOGEN CAREER ORCHESTRATOR
# =========================================================

@app.post("/api/career")
async def career_orchestrator(
    request: CareerRequest
):

    user_request = (
        request.user_request.strip()
    )

    if not user_request:

        raise HTTPException(
            status_code=400,
            detail="User request cannot be empty."
        )

    try:

        # -------------------------------------------------
        # AutoGen
        #
        # User intent
        #     ↓
        # Career Orchestrator
        #     ↓
        # Career Dispatcher
        #     ↓
        # Specialist workflow
        # -------------------------------------------------

        result = await dispatch_career_request(
            user_request=user_request,
            job_description=request.job_description,
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )