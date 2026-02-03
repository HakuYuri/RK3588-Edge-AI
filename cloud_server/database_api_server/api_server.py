import sqlite3
import json
import os
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# load config
load_dotenv()

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 9008))
# str 2 bool
API_RELOAD = os.getenv("API_RELOAD", "True").lower() == "true"
DB_PATH = os.getenv("DB_PATH", "alarms.db")

# data model
class AlarmRecord(BaseModel):
    id: int
    device_id: str
    timestamp: int
    image_url: str
    detections: List[dict]
    created_at: str

# database operation
class DBManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def query_alarms(
        self, 
        device_id: Optional[str] = None, 
        start_time: Optional[int] = None, 
        end_time: Optional[int] = None, 
        limit: int = 20,
        offset: int = 0
    ) -> List[dict]:
        query = "SELECT * FROM alarms WHERE 1=1"
        params = []

        if device_id:
            query += " AND device_id = ?"
            params.append(device_id)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
            
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                item = dict(row)
                try:
                    item['detections'] = json.loads(item['detections'])
                except:
                    item['detections'] = []
                result.append(item)
            return result

# fastapi
app = FastAPI(title="RK3588 Detection API", version="1.0.0")
db = DBManager(DB_PATH) # 使用环境变量传入的路径

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# api router

@app.get("/", tags=["Root"])
def read_root():
    return {"status": "online", "message": "RK3588 Alarm Query Service is running"}

@app.get("/alarms", response_model=List[AlarmRecord], tags=["Alarms"])
async def get_alarms(
    device_id: Optional[str] = Query(None, description="设备 ID 筛选"),
    start: Optional[int] = Query(None, description="开始时间戳"),
    end: Optional[int] = Query(None, description="结束时间戳"),
    limit: int = Query(20, gt=0, le=100, description="返回条数"),
    offset: int = Query(0, ge=0, description="跳过条数")
):
    try:
        data = db.query_alarms(device_id, start, end, limit, offset)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/devices", tags=["Utility"])
async def get_unique_devices():
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT device_id FROM alarms")
        return [row["device_id"] for row in cursor.fetchall()]

if __name__ == "__main__":
    import uvicorn

    script_name = os.path.basename(__file__).replace(".py", "")
    uvicorn.run(f"{script_name}:app", host=API_HOST, port=API_PORT, reload=API_RELOAD)