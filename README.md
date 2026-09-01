# 🌊 Coastal City Digital Twin

A **3D Digital Twin** of a coastal city built with Three.js, featuring:

- 🏙️ **Phase 2** — Full coastal city with AI disaster prediction, weather system, flood simulation, lighthouse, bridge, boats, birds, pedestrians, and vehicles.
- 🏗️ **Phase 3** — Grid-based city layout (vehicles locked to road lanes, buildings only in city blocks, park zone properly bounded) + **Hunyuan3D-2** 2D→3D camera integration.

## 🚀 Quick Start

### Frontend (No install needed)
Just open in browser:
```
coastal_city_phase2.html   ← Original city
coastal_city_phase3.html   ← Phase 3 perfect city
```

Or serve with Python:
```bash
python -m http.server 8080
# Open http://localhost:8080/coastal_city_phase3.html
```

### Backend (Hunyuan3D-2 on Port 9000)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --port 9000 --reload
```

## 🎮 Features
| Feature | Phase 2 | Phase 3 |
|---------|---------|---------|
| 3D City | ✅ | ✅ |
| Grid Road System | ❌ | ✅ |
| Vehicles on Roads Only | ❌ | ✅ |
| Park Zone Bounded | ❌ | ✅ |
| Hunyuan3D-2 2D→3D | ❌ | ✅ |
| AI Disaster Prediction | ✅ | ✅ |
| Flood Simulation | ✅ | ✅ |
| Weather System | ✅ | ✅ |
| Day/Night Cycle | ✅ | ✅ |

## 🛠️ Tech Stack
- **Frontend**: Three.js, Vanilla JS, Vanilla CSS
- **Backend**: FastAPI (Python), Uvicorn
- **3D Generation**: Hunyuan3D-2 (Tencent)
- **AI**: Custom disaster prediction models

## 📁 Structure
```
HACKATHON/
├── coastal_city_phase2.html   # Original city
├── coastal_city_phase3.html   # Phase 3 — Perfect city
├── backend/
│   ├── main.py                # FastAPI backend (port 9000)
│   ├── ml_pipeline.py         # ML/AI models
│   ├── database.py            # Data layer
│   └── requirements.txt       # Python dependencies
└── README.md
```
