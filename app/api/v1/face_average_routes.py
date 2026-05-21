from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.schemas.face_average import TwoFaceCompositeResponse
from app.services.face_average.service import FaceProcessingError, create_two_face_composite

router = APIRouter(prefix="/composite", tags=["Face Composite"])


@router.post("/two-face", response_model=TwoFaceCompositeResponse)
async def create_composite(
    request: Request,
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
) -> TwoFaceCompositeResponse:
    contents_a = await image_a.read()
    contents_b = await image_b.read()

    try:
        result = create_two_face_composite(contents_a, contents_b)
    except FaceProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    composite_url = str(request.url_for("static", path=f"composites/{result['file']}"))
    return TwoFaceCompositeResponse(
        message="Two-face composite generated",
        file=str(result["file"]),
        composite_url=composite_url,
        similarity=result["metrics"],
        strategies=result["strategies"],
    )
