import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Camera, 
  RefreshCw, 
  Cpu, 
  Clock, 
  Tag, 
  ExternalLink,
  Filter,
  AlertTriangle,
  Monitor,
  Loader2,
  WifiOff
} from 'lucide-react';

const API_BASE_URL = "https://sqlite_api_server";
const BATCH_SIZE = 20; 

const App = () => {
  const [alarms, setAlarms] = useState([]);
  const [devices, setDevices] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState(null);
  const [selectedDevice, setSelectedDevice] = useState("");
  const [isAutoRefresh, setIsAutoRefresh] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [offset, setOffset] = useState(0);

  // 哨兵元素引用
  const sentinelRef = useRef(null);

  const formatTime = (ts) => {
    return new Date(ts * 1000).toLocaleString('zh-CN', {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
  };

  const fetchAlarms = useCallback(async (isAppend = false) => {
    // 防止重复触发请求
    if (loading || loadingMore) return;

    if (isAppend) {
      setLoadingMore(true);
    } else {
      setLoading(true);
      setError(null);
      setOffset(0); 
    }

    try {
      const currentOffset = isAppend ? offset + BATCH_SIZE : 0;
      let url = `${API_BASE_URL}/alarms?limit=${BATCH_SIZE}&offset=${currentOffset}`;
      if (selectedDevice) url += `&device_id=${selectedDevice}`;
      
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);

      const newData = await response.json();

      if (isAppend) {
        setAlarms(prev => [...prev, ...newData]);
        setOffset(currentOffset);
      } else {
        setAlarms(newData);
        setOffset(0);
      }

      setHasMore(newData.length === BATCH_SIZE);
    } catch (err) {
      console.error("Fetch alarms failed:", err);
      setError("无法连接到 API，请检查后端状态。");
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, [selectedDevice, offset, loading, loadingMore]);

  // 初始化获取设备列表
  useEffect(() => {
    const fetchDevices = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/devices`);
        if (response.ok) setDevices(await response.json());
      } catch (err) { console.error(err); }
    };
    fetchDevices();
  }, []);

  // 初始加载和筛选加载
  useEffect(() => {
    fetchAlarms(false);
  }, [selectedDevice]);

  // 自动刷新逻辑
  useEffect(() => {
    let interval;
    if (isAutoRefresh && !error) {
      interval = setInterval(() => fetchAlarms(false), 10000); 
    }
    return () => clearInterval(interval);
  }, [isAutoRefresh, fetchAlarms, error]);

  // 使用 IntersectionObserver 监听哨兵元素
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore && !loading && !loadingMore && !error) {
          fetchAlarms(true);
        }
      },
      { threshold: 0.1, rootMargin: '100px' } // 提前 100px 触发加载，减少等待感
    );

    if (sentinelRef.current) {
      observer.observe(sentinelRef.current);
    }

    return () => observer.disconnect();
  }, [hasMore, loading, loadingMore, fetchAlarms, error]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-2">
              <div className="p-2 bg-indigo-600 rounded-lg shadow-lg">
                <Cpu className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-violet-600">
                RK3588 监控
              </h1>
            </div>
            <button onClick={() => fetchAlarms(false)} className="p-2 hover:bg-slate-100 rounded-full">
              <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* 工具栏 */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8 bg-white p-4 rounded-2xl shadow-sm border border-slate-100">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-slate-400" />
              <select 
                className="bg-slate-50 border-none text-sm rounded-lg p-2 outline-none"
                value={selectedDevice}
                onChange={(e) => setSelectedDevice(e.target.value)}
              >
                <option value="">所有设备</option>
                {devices.map(dev => <option key={dev} value={dev}>{dev}</option>)}
              </select>
            </div>
            <span className="text-xs text-slate-400">已载入: {alarms.length}</span>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input type="checkbox" className="sr-only peer" checked={isAutoRefresh} onChange={() => setIsAutoRefresh(!isAutoRefresh)} />
            <div className="w-11 h-6 bg-slate-200 rounded-full peer peer-checked:bg-indigo-600 after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:after:translate-x-full"></div>
            <span className="ml-3 text-sm font-medium text-slate-600">自动刷新</span>
          </label>
        </div>

        {error && (
          <div className="mb-8 p-4 bg-red-50 text-red-700 rounded-xl flex justify-between items-center">
            <div className="flex items-center gap-2"><WifiOff className="w-5 h-5"/> {error}</div>
            <button onClick={() => fetchAlarms(false)} className="underline font-bold">重试</button>
          </div>
        )}

        {/* 报警列表 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {alarms.map((alarm, index) => (
            <div key={`${alarm.id}-${index}`} className="bg-white rounded-2xl overflow-hidden shadow-sm border border-slate-100 flex flex-col hover:shadow-md transition-shadow">
              <div className="relative aspect-video bg-slate-200">
                <img src={alarm.image_url} alt="Detection" className="w-full h-full object-cover" loading="lazy" />
              </div>
              <div className="p-4 flex-1 flex flex-col">
                <div className="flex justify-between items-center mb-3">
                  <span className="text-xs font-bold text-indigo-600 truncate">{alarm.device_id}</span>
                  <span className="text-[10px] text-slate-400">{formatTime(alarm.timestamp)}</span>
                </div>
                <div className="space-y-1 mb-4">
                  {Array.isArray(alarm.detections) && alarm.detections.map((det, i) => (
                    <div key={i} className="flex justify-between text-xs bg-slate-50 p-1.5 rounded">
                      <span className="font-medium">{det.class || det.class_name}</span>
                      <span className="text-amber-600 font-mono">{(det.score * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
                <a href={alarm.image_url} target="_blank" rel="noreferrer" className="mt-auto block text-center py-2 bg-slate-900 text-white rounded-xl text-sm font-medium hover:bg-indigo-600 transition-colors">
                  查看详情
                </a>
              </div>
            </div>
          ))}
        </div>

        {/* 哨兵元素与加载指示器 */}
        <div 
          ref={sentinelRef} 
          className="w-full flex justify-center items-center mt-10"
          style={{ minHeight: '80px' }} // 固定高度，防止抖动
        >
          {loadingMore && (
            <div className="flex items-center gap-2 text-indigo-600">
              <Loader2 className="w-6 h-6 animate-spin" />
              <span className="text-sm">加载更多中...</span>
            </div>
          )}
          {!hasMore && alarms.length > 0 && (
            <div className="text-slate-400 text-sm italic">—— 已经到底部了 ——</div>
          )}
        </div>
      </main>
    </div>
  );
};

export default App;