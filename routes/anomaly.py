import time
import json
import logging
from io import BytesIO
from typing import Optional

from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException, Query, Depends
from utils import TimeFrameManager, get_from_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

anomaly_router = APIRouter(prefix="/anomalies", tags=["Anomalies"])

@anomaly_router.get("/")
async def get_anomalies(
    limit: Optional[int] = Query(50, ge=1, le=1000),
    timeframe: Optional[str] = Query(None, regex="^(1h|6h|12h|24h|7d|30d)$"),
    label_filter: Optional[str] = Query(None),
    storage = Depends(get_from_state("anomaly_storage"))
):
    try:
        since_timestamp = None
        if timeframe:
            since_timestamp = TimeFrameManager.get_timeframe_start(timeframe)
        
        anomalies = storage.get_anomalies(
            limit=limit,
            since_timestamp=since_timestamp,
            label_filter=label_filter
        )
        
        return {
            "anomalies": anomalies,
            "count": len(anomalies),
            "timeframe": timeframe,
            "filter_applied": label_filter is not None
        }
        
    except Exception as e:
        logger.error(f"Error retrieving anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@anomaly_router.get("/{anomaly_id}")
async def get_anomaly_details(anomaly_id: str, storage = Depends(get_from_state("anomaly_storage"))):
    """Get detailed information about a specific anomaly"""
    try:
        anomaly = storage.get_anomaly_by_id(anomaly_id)
        if not anomaly:
            raise HTTPException(status_code=404, detail="Anomaly not found")
        
        return anomaly
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving anomaly {anomaly_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@anomaly_router.get("/stats/summary")
async def get_anomaly_stats(as_and_vp = Depends(get_from_state("anomaly_storage", "video_processor"))):
    """Get anomaly statistics and summary"""
    try:
        storage, vp = as_and_vp
        stats = storage.get_stats()
        
        recent_anomalies = storage.get_anomalies(
            since_timestamp=TimeFrameManager.get_timeframe_start('24h')
        )
        
        hourly_data = TimeFrameManager.group_by_hour(recent_anomalies)
        
        return {
            "summary": stats,
            "trends": {
                "hourly_counts": hourly_data,
                "peak_hour": max(hourly_data.items(), key=lambda x: x[1])[0] if hourly_data else None
            },
            "processing_stats": {
                "total_frames_processed": vp.processed_frames,
                "anomaly_detection_rate": (
                    vp.total_anomalies / vp.processed_frames * 100
                    if vp.processed_frames > 0 else 0
                )
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating anomaly stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@anomaly_router.delete("/")
async def clear_anomalies(storage = Depends(get_from_state("anomaly_storage"))):
    try:
        storage.clear_all()
        return {"message": "All anomalies cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@anomaly_router.post("/export")
async def export_anomalies(storage = Depends(get_from_state("anomaly_storage"))):
    try:
        data = {
            'export_timestamp': time.time(),
            'export_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'anomalies': storage.get_anomalies()
        }
        
        buffer = BytesIO()
        json.dump(data, buffer, indent=2)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=anomalies_export.json"}
        )

    except Exception as e:
        logger.error(f"Error exporting anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))
 