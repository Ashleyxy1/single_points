"""
WebSocket 发送模块
将手势识别结果发送到网关，支持自动重连
"""
import json
import time
import threading
from collections import deque

try:
    import websocket
    HAS_WS = True
except ImportError:
    HAS_WS = False


class WSSender:
    """WebSocket 发送器，后台线程发送，不阻塞主循环。"""

    def __init__(self, url):
        if not HAS_WS:
            raise ImportError("请先安装 websocket-client: pip install websocket-client")
        self.url = url
        self.ws = None
        self._queue = deque()
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def connect(self):
        """连接并启动后台发送线程。"""
        self._running = True
        self._try_connect()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self._connected()

    def _connected(self):
        return self.ws is not None

    def _try_connect(self):
        try:
            self.ws = websocket.create_connection(self.url, timeout=3)
            print(f"[WS] 已连接 {self.url}")
        except Exception as e:
            print(f"[WS] 连接失败 ({self.url}): {e}")
            self.ws = None

    def send(self, data):
        """加入发送队列（非阻塞）。data 为 dict。"""
        if not self._running:
            return
        data["timestamp"] = time.time()
        with self._lock:
            self._queue.append(data)

    def _loop(self):
        """后台循环：发送队列中的消息，断线自动重连。"""
        reconnect_interval = 5
        last_reconnect = 0

        while self._running:
            now = time.time()

            # 断线重连
            if self.ws is None and now - last_reconnect > reconnect_interval:
                self._try_connect()
                last_reconnect = now

            # 发送
            if self.ws is not None:
                with self._lock:
                    if self._queue:
                        msg = self._queue.popleft()
                    else:
                        msg = None

                if msg is not None:
                    try:
                        text = json.dumps(msg, ensure_ascii=False)
                        self.ws.send(text)
                    except Exception:
                        print("[WS] 发送失败，断开")
                        try:
                            self.ws.close()
                        except Exception:
                            pass
                        self.ws = None
                        # 重新入队
                        with self._lock:
                            self._queue.appendleft(msg)
                        last_reconnect = now - reconnect_interval
                    continue
            # 无消息时休眠，避免空转抢 CPU
            time.sleep(0.05)

    def close(self):
        """关闭连接和线程。"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        self.ws = None
        print("[WS] 已断开")


# ---- 便捷工厂 ----

def create_sender(url):
    """创建并连接 WSSender，失败返回 None。"""
    if not HAS_WS:
        print("[WS] 未安装 websocket-client，跳过 (pip install websocket-client)")
        return None
    sender = WSSender(url)
    sender.connect()
    if not sender._connected():
        print("[WS] 将在后台自动重连，识别结果会在连接恢复后发送")
    return sender
