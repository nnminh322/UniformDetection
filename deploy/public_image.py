from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os

app = FastAPI()

BASE_FOLDER = "/home/minhnn/research/deploy_uniform/deploy/image"

@app.get("/image/{date}/{violation_type}/{filename}")
def serve_image(date: str, violation_type: str, filename: str):
    """
    URL mẫu: /image/20251204/uniform/123456_full.jpg
    """
    if ".." in date or ".." in violation_type or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path")
    
    if violation_type not in ["uniform", "hat"]:
        raise HTTPException(status_code=400, detail="Invalid violation type")
    
    image_path = os.path.join(BASE_FOLDER, date, violation_type, filename)
    
    if not os.path.isfile(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(image_path, media_type="image/jpeg")