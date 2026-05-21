from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import API_PREFIX, DATA_DIR
from app.core.storage import ensure_storage_dirs

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(
    title="Two Face Composite",
    description="Generate a face composite and visual similarity score from two uploaded face photos",
    version="1.0.0",
)

ensure_storage_dirs()
app.mount("/static", StaticFiles(directory=str(DATA_DIR)), name="static")
app.mount("/assets", StaticFiles(directory=str(WEB_DIR)), name="assets")
app.include_router(api_router, prefix=API_PREFIX)


@app.get("/")
def root() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/privacy")
def privacy() -> FileResponse:
    return FileResponse(WEB_DIR / "privacy.html")


@app.get("/how-it-works")
def how_it_works() -> FileResponse:
    return FileResponse(WEB_DIR / "how-it-works.html")
