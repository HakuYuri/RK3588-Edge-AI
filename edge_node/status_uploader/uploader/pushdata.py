import requests
import time
import os
from dotenv import load_dotenv

# 加载 .env 文件中的变量
load_dotenv()

# 从环境变量中读取配置 同时给个默认值以防万一
NODE_EXPORTER_URL = os.getenv("NODE_EXPORTER_URL", "http://localhost:9100/metrics")
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL")
# env读到的是字符串，转int
PUSH_INTERVAL = int(os.getenv("PUSH_INTERVAL", 1))

def push_metrics():
    if not PUSHGATEWAY_URL:
        print("Error: PUSHGATEWAY_URL is not set in .env file.")
        return

    try:
        # 抓 Node Exporter 数据
        resp = requests.get(NODE_EXPORTER_URL, timeout=2)
        if resp.status_code != 200:
            print(f"Failed to fetch metrics: {resp.status_code}")
            return

        # 筛选温度指标
        lines = resp.text.splitlines()
        # 仅保留包含 node_thermal_zone_temp 的行，并确保包含数据行
        relevant_metrics = [l for l in lines if "node_thermal_zone_temp" in l and not l.startswith("#")]
        
        if not relevant_metrics:
            print("No thermal metrics found.")
            return

        payload = "\n".join(relevant_metrics) + "\n"

        # 推送至服务端
        push_resp = requests.post(PUSHGATEWAY_URL, data=payload, timeout=5)
        print(f"[{time.strftime('%H:%M:%S')}] Pushed to server: {push_resp.status_code}")
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    print(f"Starting RK3588 Thermal Pusher...")
    print(f"Source: {NODE_EXPORTER_URL}")
    print(f"Target: {PUSHGATEWAY_URL}")
    
    while True:
        push_metrics()
        time.sleep(PUSH_INTERVAL)