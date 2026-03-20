from pathlib import Path
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from detect import run_detection

app = FastAPI()

UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory="results"), name="results")


@app.get("/", response_class=HTMLResponse)
async def home():
    return FileResponse("static/index.html")


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        return {"error": "Invalid file type. Please upload an image."}

    filename = f"{uuid.uuid4()}{ext}"
    filepath = UPLOAD_DIR / filename

    content = await file.read()
    filepath.write_bytes(content)

    try:
        result = run_detection(str(filepath), str(RESULTS_DIR))
    finally:
        filepath.unlink(missing_ok=True)

    return {
        "detections": result["detections"],
        "image_url": f"/results/{result['output_filename']}",
        "stats": result["stats"]
    }
