import logging
import asyncio
from fastapi import FastAPI, Depends 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from models.anomaly_detector import AnomalyDetector
from models.dataset import StreamDataset
from utils import AnomalyStorage, VideoProcessor, get_from_state, setup_logging, run_video_processing
from routes import anomaly_router

setup_logging("logs/app.log")


logger = logging.getLogger(__name__)

app = FastAPI(title="Security Camera Anomaly Detector", version="1.0.0")
app.include_router(anomaly_router)

@app.on_event("startup")
async def startup():
    
    try:
        logger.info("Initializing anomaly detector...")
        app.state.anomaly_detector = AnomalyDetector()

        logger.info("Initializing anomaly storage...")
        app.state.anomaly_storage = AnomalyStorage(max_items=5000, retention_hours=48)
        app.state.stream_dataset = StreamDataset(buffer_size=200)
        app.state.video_processor = VideoProcessor(app.state)
        app.state.model = app.state.video_processor.model 
        asyncio.create_task(run_video_processing(app.state.video_processor))

        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise


@app.get("/health")
async def health_check(all_states = Depends(get_from_state("anomaly_detector", "anomaly_storage", "stream_dataset"))):
    try:
        detector , storage, stream_dataset = all_states
        stats = storage.get_stats()

        return {
            "status": "healthy",
            "anomaly_detector": "initialized" if detector else "not_initialized",
            "storage": stats,
            "stream_buffer_size": len(stream_dataset)
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/")
async def serve_html():
    with open("index.html", "r") as f:
        return HTMLResponse(f.read())




# @app.get("/")
# async def root(vp: VideoProcessor = Depends(get_from_state("video_processor"))):
#     return {
#         "status": "running",
#         "service": "Security Camera Anomaly Detector",
#         "version": "1.0.0",
#         "processed_frames": vp.processed_frames,
#         "total_anomalies": vp.total_anomalies
#     }

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
