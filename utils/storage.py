# utils/storage.py
import json
import time
from typing import Dict, List, Optional
from collections import deque
import threading
from pathlib import Path

class AnomalyStorage:
    """
    Temporary storage for anomaly detections with timeframe management
    Thread-safe implementation for concurrent access
    """
    
    def __init__(self, max_items: int = 1000, retention_hours: int = 24):
        self.max_items = max_items
        self.retention_seconds = retention_hours * 3600
        self.anomalies = deque(maxlen=max_items)
        self.lock = threading.Lock()
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_items, daemon=True)
        self.cleanup_thread.start()
    
    def store_anomaly(self, 
                     anomaly_label: str, 
                     confidence: float,
                     timestamp: Optional[float] = None,
                     metadata: Optional[Dict] = None) -> str:
        """
        Store an anomaly detection result
        Returns: anomaly_id for reference
        """
        if timestamp is None:
            timestamp = time.time()
        
        anomaly_id = f"anomaly_{timestamp}_{len(self.anomalies)}"
        
        anomaly_record = {
            'id': anomaly_id,
            'label': anomaly_label,
            'confidence': confidence,
            'timestamp': timestamp,
            'human_readable_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)),
            'metadata': metadata or {}
        }
        
        with self.lock:
            self.anomalies.append(anomaly_record)
        
        return anomaly_id
    
    def get_anomalies(self, 
                     limit: Optional[int] = None,
                     since_timestamp: Optional[float] = None,
                     label_filter: Optional[str] = None) -> List[Dict]:
        """
        Retrieve anomalies with optional filtering
        """
        with self.lock:
                filtered_anomalies = list(self.anomalies)
        
        if since_timestamp:
            filtered_anomalies = [
                a for a in filtered_anomalies 
                if a['timestamp'] >= since_timestamp
            ]
        
        if label_filter:
            filtered_anomalies = [
                a for a in filtered_anomalies 
                if label_filter.lower() in a['label'].lower()
            ]
        
        filtered_anomalies.sort(key=lambda x: x['timestamp'], reverse=True)
        
        if limit:
            filtered_anomalies = filtered_anomalies[:limit]
        
        return filtered_anomalies
    
    def get_anomaly_by_id(self, anomaly_id: str) -> Optional[Dict]:
        with self.lock:
            for anomaly in self.anomalies:
                if anomaly['id'] == anomaly_id:
                    return anomaly
        return None
    
    def get_stats(self) -> Dict:
        with self.lock:
            current_time = time.time()
            total_count = len(self.anomalies)
            
            recent_count = sum(
                1 for a in self.anomalies 
                if current_time - a['timestamp'] <= 3600
            )
            
            label_counts = {}
            for anomaly in self.anomalies:
                label = anomaly['label']
                label_counts[label] = label_counts.get(label, 0) + 1
        
        return {
            'total_anomalies': total_count,
            'recent_anomalies_1h': recent_count,
            'storage_utilization': f"{total_count}/{self.max_items}",
            'label_distribution': label_counts,
            'oldest_timestamp': self.anomalies[0]['timestamp'] if self.anomalies else None,
            'newest_timestamp': self.anomalies[-1]['timestamp'] if self.anomalies else None
        }
    
    def clear_all(self):
        """Clear all stored anomalies"""
        with self.lock:
            self.anomalies.clear()
    
    def export_to_json(self, filepath: str):
        """Export all anomalies to JSON file"""
        with self.lock:
            data = {
                'export_timestamp': time.time(),
                'export_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'anomalies': list(self.anomalies)
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _cleanup_old_items(self):
        while True:
            try:
                current_time = time.time()
                cutoff_time = current_time - self.retention_seconds
                
                with self.lock:
                    while (self.anomalies and 
                           self.anomalies[0]['timestamp'] < cutoff_time):
                        self.anomalies.popleft()
                
                time.sleep(300)
                
            except Exception as e:
                print(f"Cleanup thread error: {e}")
                time.sleep(60) 

class TimeFrameManager:
    @staticmethod
    def get_timeframe_start(timeframe: str) -> float:
        current_time = time.time()
        timeframe_seconds = {
            '1h': 3600,
            '6h': 6 * 3600,
            '12h': 12 * 3600,
            '24h': 24 * 3600,
            '7d': 7 * 24 * 3600,
            '30d': 30 * 24 * 3600
        }
        
        if timeframe in timeframe_seconds:
            return current_time - timeframe_seconds[timeframe]
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    @staticmethod
    def group_by_hour(anomalies: List[Dict]) -> Dict[str, int]:
        hourly_counts = {}
        
        for anomaly in anomalies:
            hour_key = time.strftime('%Y-%m-%d %H:00', 
                                   time.localtime(anomaly['timestamp']))
            hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1
        
        return hourly_counts
    
    @staticmethod
    def group_by_day(anomalies: List[Dict]) -> Dict[str, int]:
        daily_counts = {}
        
        for anomaly in anomalies:
            day_key = time.strftime('%Y-%m-%d', 
                                  time.localtime(anomaly['timestamp']))
            daily_counts[day_key] = daily_counts.get(day_key, 0) + 1
        
        return daily_counts