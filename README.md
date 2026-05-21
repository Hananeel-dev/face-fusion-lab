# Two Face Composite

A FastAPI web app that accepts two face photos, aligns both faces into the same landmark coordinate space, blends them into one composite image, and returns a visual similarity breakdown.

This is a face-averaging and image-processing demo. The similarity score describes how easily two uploaded photos align and blend; it is not identity verification.

## Features

- Upload any two clear face photos.
- Detect, crop, align, and resize each face.
- Warp both faces toward their shared average landmark shape.
- Generate a 50/50 composite image.
- Return a similarity score with landmark, warp, texture, and composite-agreement metrics.
- Display a technical explanation directly in the web app.

## How It Works

1. MediaPipe estimates facial landmarks for each image.
2. The face crop is rotated so the eyes are level and resized to `256x256`.
3. The two landmark sets are averaged into a shared target shape.
4. A Delaunay triangle mesh warps both faces into that target shape.
5. The two warped images are blended equally.
6. The app scores visual similarity using normalized landmark distance, average warp effort, grayscale texture difference, and pixel-level variation.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## API

Create a composite from two photos:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/composite/two-face" \
  -F "image_a=@/path/to/first-face.jpg" \
  -F "image_b=@/path/to/second-face.jpg"
```

Example response:

```json
{
  "message": "Two-face composite generated",
  "file": "two_face_composite_abc123.png",
  "composite_url": "http://127.0.0.1:8000/static/composites/two_face_composite_abc123.png",
  "similarity": {
    "overall_similarity": 72.4,
    "landmark_geometry_similarity": 80.1,
    "warp_similarity": 74.8,
    "texture_similarity": 61.2,
    "composite_agreement": 66.9
  }
}
```

## Runtime Data

Generated runtime files are intentionally ignored by Git:

- `app/storage/composites/two_face_composite_*.png`
- uploaded or temporary images
- Python cache files

The app does not intentionally store input photos or individual extracted face crops.
