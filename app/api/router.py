from fastapi import APIRouter

from app.api.v1.face_average_routes import router as face_average_router

api_router = APIRouter()
api_router.include_router(face_average_router)
