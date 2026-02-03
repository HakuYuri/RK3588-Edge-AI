# RK3588 AIoT 智能边缘监控系统

## 项目简介

本项目是一个基于 Rockchip RK3588 边缘计算平台的端云协同智能监控系统。系统利用 RK3588 的 NPU 进行本地实时目标检测（YOLO），利用 VPU 进行硬件视频编码推流。边缘端将识别结果（图片、坐标、类别）和设备状态数据上传至云端服务器，云端提供数据存储、视频分发、报警记录查询及可视化监控面板。

## 系统架构

系统主要由以下三部分组成：

1. **边缘计算节点 (Edge Node)**
* **硬件平台**: RK3588 (Arm64)。
* **核心功能**: 视频采集、图像预处理、RKNN 模型推理 (YOLO)、MPP 硬件编码推流、报警数据上传 (MQTT/MinIO)、设备状态监控 (Node Exporter)。


2. **云端服务 (Cloud Server)**
* **基础服务**: MQTT Broker (消息队列), MinIO (对象存储), SRS (RTMP/WebRTC 流媒体服务器)。
* **业务服务**: Python 后端 (数据存储与 API), React 前端 (报警查询与实时预览)。
* **监控服务**: Prometheus + Pushgateway + Grafana (设备性能监控)。


3. **PC 端 (Model Export)**
* 用于将 PyTorch/ONNX 模型转换为适配 RK3588 NPU 的 `.rknn` 模型文件。



## 目录结构

```text
.
├── cloud_server               # 云端服务部署代码
│   ├── database_api_server    # 报警查询 API 服务 (FastAPI + SQLite)
│   ├── frontend               # 前端可视化界面 (React + Vite)
│   ├── minio                  # 图片存储服务配置
│   ├── mosquitto              # MQTT 消息代理配置
│   ├── mqtt_subscriber        # 报警消息订阅与入库服务
│   ├── nginx_proxy_manager    # 反向代理服务
│   ├── rk3588_monitor         # 设备状态监控 (Prometheus + Grafana)
│   └── srs_server             # 流媒体服务器配置
├── edge_node                  # RK3588 边缘端代码
│   ├── 3rdparty               # 第三方依赖 (RKNN Toolkit Lite2)
│   ├── main                   # 核心检测与推流程序
│   └── status_uploader        # 设备温度与负载上报程序
└── pc                         # 模型转换工具 (x86/PC)
    ├── 3rdparty               # RKNN Toolkit2 依赖
    └── model_export           # YOLO 模型转 RKNN 脚本

```

## 部署指南

### 1. 云端服务部署 (Cloud Server)

云端服务主要通过 Docker Compose 进行容器化部署，业务逻辑通过 Python 和 Node.js 运行。

**环境要求**: Docker, Docker Compose, Python 3.9+, Node.js 18+

#### 基础中间件启动

进入各个服务的目录并启动容器：

```bash
# 启动 MinIO (存储), Mosquitto (通信), SRS (流媒体), Monitor (监控), Nginx (代理)
cd cloud_server/minio && docker-compose up -d
cd ../mosquitto && docker-compose up -d
cd ../srs_server && docker-compose up -d
cd ../rk3588_monitor && docker-compose up -d
cd ../nginx_proxy_manager && docker-compose up -d

```

**服务端口说明**:

(可根据实际情况修改docker-compose.yml)
* SRS RTMP: 1935, HTTP: 8080
* MinIO API: 9002, Console: 9003
* Mosquitto MQTT: 1883
* Grafana: 9006
* Prometheus: 9004
* Pushgateway: 9005

#### 业务服务启动

1. **报警订阅服务**:
配置 `.env` 文件（参考 `subscriber.py` 中的环境变量），启动订阅脚本将 MQTT 数据写入 SQLite 数据库。
```bash
cd cloud_server/mqtt_subscriber
pip install -r requirements.txt
python subscriber.py

```


2. **API 后端服务**:
启动 FastAPI 服务，提供前端查询接口。
```bash
cd cloud_server/database_api_server
# 确保 DB_PATH 指向 subscriber 生成的数据库文件
python api_server.py

```


*默认端口: 9008*
3. **前端界面**:
编译并运行 React 前端。
```bash
cd cloud_server/frontend/rk3588_alarm
npm install lucide-react
# 修改 src/App.jsx 中的 API_BASE_URL 指向 API 服务器地址
npm install @tailwindcss/vite
npm run dev

```



### 2. 边缘端部署 (Edge Node)

该部分代码运行在 RK3588 开发板上。

**环境要求**: Ubuntu 20.04/22.04 (Rockchip BSP), Python 3.8/3.9

#### 依赖安装

安装 RKNN Toolkit Lite2（位于 `edge_node/3rdparty` 目录）及 Python 依赖。

```bash
cd edge_node/3rdparty/rknn-toolkit-lite2/packages
pip install rknn_toolkit_lite2-*-cp3x-*-aarch64.whl
cd ../../../main
pip install -r requirements.txt

```

*注意: 必须确保系统已安装对应版本的 librknnrt.so 和 MPP 驱动。*（建议参考网络上具体的开发板教程，MPP及FFmpeg需要编译(或交叉编译)）
参考：[rkmpp](https://gitee.com/mirrors_rockchip-linux/mpp)

#### 运行主程序

配置 `edge_node/main/.env` 文件，填入云端服务器的 MQTT、MinIO 和 RTMP 地址。

```bash
cd edge_node/main
python app.py

```

该程序将自动开启两个线程：

* **推流线程**: 调用 FFmpeg 硬件编码接口将摄像头画面推送到 SRS 服务器。
* **推理线程**: 将画面输入 NPU 进行推理，检测到目标后上传图片至 MinIO 并发送 MQTT 报警。

#### 运行状态上报

用于上报 RK3588 的 CPU 温度及负载信息至云端 Prometheus。

```bash
cd edge_node/status_uploader/uploader
python pushdata.py

```

### 3. 模型转换 (PC 端)

如果需要更新检测模型，需在 PC 端将 PyTorch/ONNX 模型转换为 RKNN 格式。（建议参考网络上具体的开发板教程）
参考：
[rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)


1. 进入 `pc/model_export` 目录。
2. 确保已安装 `ultralytics` 和 `rknn-toolkit2`。
3. 运行导出脚本：
```bash
python export_rk3588_model.py

```

*注意: 转换模型需要在x86_64或arm64下，MacOS无法直接转换，可以用docker部署arm64 Ubuntu进行转换*
生成的 `.rknn` 文件需复制到 `edge_node/main` 目录下使用。

## 配置说明

本项目大量使用环境变量 (`.env`) 进行配置。部署前请务必检查以下关键配置项：

* **MQTT_BROKER / MQTT_HOST**: MQTT 服务器 IP 地址。
* **MINIO_ENDPOINT**: MinIO API 地址 (不带 http 前缀)。
* **RTMP_URL**: SRS 服务器推流地址 (例如 `rtmp://<server_ip>/live/stream`).
* **PUSHGATEWAY_URL**: Prometheus Pushgateway 地址。

## 许可证

本项目采用 LICENSE 文件中规定的许可证。