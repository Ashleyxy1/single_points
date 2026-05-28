"""
WebSocket 网关接收端
接收来自 realtime_translate.py / realtime_dynamic.py 的手势识别结果

用法:
    python ws_receiver.py                  # 默认 0.0.0.0:8765
    python ws_receiver.py --port 9000      # 自定义端口
"""
import json
import asyncio
import sys

try:
    import websockets
except ImportError:
    print("请先安装: pip install websockets")
    sys.exit(1)


async def handler(websocket):
    """处理单个客户端连接。"""
    addr = websocket.remote_address
    print(f"[+] 客户端连接: {addr}")
    try:
        async for raw in websocket:
            data = json.loads(raw)
            gesture = data.get("gesture", "?")
            gtype = data.get("type", "?")
            conf = data.get("confidence", 0)

            if gtype == "static":
                stability = data.get("stability", 0)
                print(f"[静态] {gesture}  |  置信度: {conf:.1%}  |  稳定性: {stability:.1%}")
            else:
                frames = data.get("frames", 0)
                dtw = data.get("dtw_ms", 0)
                print(f"[动态] {gesture}  |  置信度: {conf:.1%}  |  帧数: {frames}  |  DTW: {dtw}ms")

            # ---- 在这里做你自己的业务逻辑 ----
            # 比如: 控制 IoT 设备、写入数据库、触发脚本、HTTP 转发等
    except websockets.exceptions.ConnectionClosed:
        print(f"[-] 客户端断开: {addr}")


async def main(host="0.0.0.0", port=8765):
    print(f"网关启动: ws://{host}:{port}")
    async with websockets.serve(handler, host, port):
        print("等待手势识别客户端连接...\n")
        await asyncio.Future()  # 永久运行


if __name__ == "__main__":
    port = 8765
    for arg in sys.argv[1:]:
        if arg.startswith("--port="):
            port = int(arg[7:])
    asyncio.run(main(port=port))
