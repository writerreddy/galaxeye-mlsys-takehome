"""
FastAPI application entrypoint.

Run with: uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.db.database import init_db
from app.ml.classifier import classifier

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        classifier.load()
    except Exception:
        # Don't crash the whole app if the model artifact is missing —
        # /health will report model_loaded=False and /tiles/classify
        # will return a clean 500, which is more debuggable than a
        # process that won't start at all.
        logger.exception(
            "Failed to load model at startup. /tiles/classify will fail "
            "until a valid model artifact exists at %s",
            settings.model_path,
        )
    yield


app = FastAPI(
    title=settings.app_name,
    description=(
        "Offline satellite tile land-use classification service. "
        "See /docs for interactive API documentation."
    ),
    version=settings.model_version,
    lifespan=lifespan,
)

app.include_router(router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    # Last-resort safety net: never leak a raw traceback to the client.
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
