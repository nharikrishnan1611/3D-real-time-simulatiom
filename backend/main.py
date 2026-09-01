import base64
import random
import uuid
import requests
from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from sqlalchemy.orm import Session
from database import get_db, Building
from ml_pipeline import predictor

app = FastAPI(title="Coastal City 2D->3D Digital Twin API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReconstructRequest(BaseModel):
    image: str # Base64 encoded image

class PredictRequest(BaseModel):
    flood_level: float
    building_ids: List[str]

import cv2
import numpy as np

@app.post("/api/reconstruct")
async def reconstruct_city(req: ReconstructRequest, db: Session = Depends(get_db)):
    """
    Accepts a 2D image, uses OpenCV to extract actual real-world shapes and contours 
    from the image, and converts them into 3D building polygons.
    """
    
    # 1. Decode the actual image from the camera/upload
    image_data = base64.b64decode(req.image.split(",")[1])
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 2. Process image with OpenCV to extract real shapes
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours (outlines of buildings/objects in the photo)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    buildings = []
    db.query(Building).delete()
    
    # Map image pixel coordinates to 3D world coordinates
    # We assume the image represents a 200x200 area in the 3D world
    img_h, img_w = img.shape[:2]
    
    count = 0
    for contour in contours:
        # Ignore very small shapes
        if cv2.contourArea(contour) < 500:
            continue
            
        # Get bounding box for the actual object in the photo
        x, y, w, h = cv2.boundingRect(contour)
        
        # Convert pixel coordinates to 3D world space coordinates
        world_x = (x / img_w) * 200 - 100
        world_z = (y / img_h) * 200 - 100
        world_w = (w / img_w) * 200
        world_d = (h / img_h) * 200
        
        # Estimate height based on object size (larger footprint = taller building)
        estimated_height = max(10, min((world_w * world_d) / 5, 80))
        
        b_id = f"bld_{uuid.uuid4().hex[:8]}"
        b_type = 'skyscraper' if estimated_height > 50 else ('modern' if estimated_height > 20 else 'residential')
        
        polygon = {
            "x": world_x + (world_w / 2),
            "z": world_z + (world_d / 2),
            "width": world_w,
            "depth": world_d
        }
        
        b = Building(
            id=b_id,
            type=b_type,
            estimated_height=estimated_height,
            estimated_elevation=0,
            polygon=polygon,
            vulnerability_score=0.0
        )
        db.add(b)
        buildings.append({
            "id": b_id,
            "type": b_type,
            "polygon": polygon,
            "estimated_height": estimated_height,
            "estimated_elevation": 0
        })
        
        count += 1
        if count >= 100:  # Cap at 100 buildings to prevent browser crash
            break
            
    db.commit()

    return {
        "status": "success",
        "message": f"Successfully extracted {count} real objects from the image.",
        "buildings": buildings,
        "roads": [] # Procedural roads can be added here
    }

@app.post("/api/predict")
async def predict_flood_impact(req: PredictRequest, db: Session = Depends(get_db)):
    """
    Runs XGBoost / Logistic Regression to predict building vulnerability based on flood levels.
    """
    results = []
    
    buildings = db.query(Building).filter(Building.id.in_(req.building_ids)).all()
    
    for b in buildings:
        # Distance to water is simulated as distance from origin (0,0) for the demo
        dist = (b.polygon['x']**2 + b.polygon['z']**2)**0.5
        
        pred = predictor.predict(
            elevation=b.estimated_elevation,
            building_height=b.estimated_height,
            distance_to_water=dist,
            building_type=b.type,
            flood_level=req.flood_level
        )
        
        b.vulnerability_score = pred['vulnerability_score']
        
        results.append({
            "id": b.id,
            "prediction": pred
        })
        
    db.commit()
    return {"status": "success", "predictions": results}

@app.get("/api/weather")
async def get_weather():
    """
    Connects to Open-Meteo API to fetch real-time data.
    """
    try:
        # Fetching for a sample coastal city (e.g. Miami: lat 25.76, lon -80.19)
        url = "https://api.open-meteo.com/v1/forecast?latitude=25.76&longitude=-80.19&current_weather=true"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        return {"status": "success", "data": data['current_weather']}
    except Exception as e:
        return {"status": "error", "message": str(e), "data": {"weathercode": 0}}

