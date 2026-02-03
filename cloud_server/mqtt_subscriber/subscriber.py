import json
import sqlite3
import os
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion
from dotenv import load_dotenv

# load config
load_dotenv()

MQTT_BROKER = os.getenv("MQTT_BROKER", "127.0.0.1")
# convert str
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "rk3588/alarms")
DB_NAME = os.getenv("DB_NAME", "alarms.db")

# database operation
class AlarmDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """ init database """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alarms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    device_id TEXT,
                    timestamp INTEGER,
                    image_url TEXT,
                    detections TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def save_alarm(self, data):
        """ save message to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO alarms (device_id, timestamp, image_url, detections) VALUES (?, ?, ?, ?)",
                    (data.get('device_id'), data.get('timestamp'), data.get('image_url'), json.dumps(data.get('detections')))
                )
                conn.commit()
                print(f"✅ [{time.strftime('%Y-%m-%d %H:%M:%S')}] 已记录来自 {data.get('device_id')} 的报警")
        except Exception as e:
            print(f"数据库写入失败: {e}")

# mqtt callback
db = AlarmDatabase(DB_NAME)

def on_connect(client, userdata, flags, rc, properties):
    if rc == 0:
        print(f"🌐 已成功连接到 MQTT Broker ({MQTT_BROKER})")
        client.subscribe(MQTT_TOPIC)
        print(f"📡 已订阅主题: {MQTT_TOPIC}")
    else:
        print(f"连接失败，错误代码: {rc}")

def on_message(client, userdata, msg):
    try:
        # parse json
        payload = json.loads(msg.payload.decode())
        # save to db
        db.save_alarm(payload)
    except Exception as e:
        print(f"解析消息时出错: {e}")

# --- 主程序 ---
def run():
    # 使用 CallbackAPIVersion.VERSION2 兼容最新 paho-mqtt
    client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2)
    
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"正在尝试连接到 {MQTT_BROKER}:{MQTT_PORT}...")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        # loop_forever 会阻塞在这里，持续监听
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n程序已手动停止")
    except Exception as e:
        print(f"运行出错: {e}")

if __name__ == "__main__":
    import time # 用于日志输出的时间戳
    run()