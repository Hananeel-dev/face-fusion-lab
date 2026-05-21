from pydantic import BaseModel


class SimilarityMetrics(BaseModel):
    overall_similarity: float
    landmark_geometry_similarity: float
    warp_similarity: float
    texture_similarity: float
    composite_agreement: float
    landmark_rms: float
    average_warp_effort_px: float
    mean_absolute_texture_difference: float
    pixel_variance: float


class CompositeStrategies(BaseModel):
    image_a: str
    image_b: str
    warp: str
    blend: str


class TwoFaceCompositeResponse(BaseModel):
    message: str
    file: str
    composite_url: str
    similarity: SimilarityMetrics
    strategies: CompositeStrategies
