from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import time
import uuid
import os
import uvicorn
from gradio_client import Client, handle_file
import tempfile
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Hunyuan3D-2 Local API Proxy")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("models", exist_ok=True)
# Mount the models directory so the frontend can download the generated glb
app.mount("/models", StaticFiles(directory="models"), name="models")

class ImageRequest(BaseModel):
    image: str

@app.post("/api/camera-to-3d")
async def convert_to_3d(req: ImageRequest):
    start_time = time.time()
    
    # Save the base64 image to a temporary file
    try:
        header, encoded = req.image.split(",", 1)
        image_data = base64.b64decode(encoded)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            tmp_img.write(image_data)
            tmp_img_path = tmp_img.name
            
        print(f"Image saved to {tmp_img_path}")
    except Exception as e:
        return {"status": "error", "message": f"Invalid image format: {e}"}

    try:
        # Use HuggingFace public space as a proxy for the local Hunyuan3D-2 API
        print("Sending to Hunyuan3D-2 HuggingFace Space...")
        client = Client("tencent/Hunyuan3D-2")
        
        # Call the image to 3d generation API endpoint in the space
        # According to standard Hunyuan3D-2 Gradio space API
        result = client.predict(
            image=handle_file(tmp_img_path),
            seed=0,
            steps=30,
            guidance_scale=5.5,
            api_name="/image_to_3d"
        )
        
        print("API Response:", result)
        
        # Result is usually a tuple where the second element is the path to the glb file
        if isinstance(result, tuple) and len(result) >= 2:
            glb_path = result[1]
        elif isinstance(result, str):
            glb_path = result
        else:
            glb_path = result[0] # Fallback
            
        # Copy the glb to our local models directory to serve it
        unique_id = uuid.uuid4().hex[:8]
        filename = f"hunyuan_{unique_id}.glb"
        local_glb_path = os.path.join("models", filename)
        
        import shutil
        shutil.copy2(glb_path, local_glb_path)
        
        return {
            "status": "success", 
            "glb_url": f"http://localhost:9000/models/{filename}",
            "filename": filename,
            "processing_time": round(time.time() - start_time, 2)
        }
        
    except Exception as e:
        print(f"Hunyuan3D-2 API error: {e}")
        # Return fallback demo response if HuggingFace space fails or is busy
        return {
            "status": "demo",
            "seed": 0.42,
            "message": "Public API failed or busy. Using procedural demo.",
            "processing_time": round(time.time() - start_time, 2)
        }
    finally:
        if os.path.exists(tmp_img_path):
            os.remove(tmp_img_path)

if __name__ == "__main__":
    uvicorn.run("main_phase3:app", host="0.0.0.0", port=9000, reload=True)
