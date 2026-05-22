import cv2
import os
from datetime import datetime

def take_photos(save_dir="captured_images", camera_id=0):
    """
    参数:
        save_dir: 图片保存的文件夹名 (默认 "captured_images")
        camera_id: 摄像头设备 ID (默认0,通常是内置摄像头)
    """
    # 创建保存文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"已创建文件夹: {save_dir}")
    
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("错误：无法打开摄像头，请检查设备是否可用。")
        return
    
    print("按 'SPACE' 键拍照并保存")
    print("按 'ESC' 或 'Q' 键退出程序")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取画面，请检查摄像头。")
            break
        
        # 显示画面
        cv2.imshow("按空格拍照，按 ESC/Q 退出", frame)
        
        # 监听键盘事件
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):          # 空格键拍照
            # 生成文件名：时间戳 + 序号
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"hand_{timestamp}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"已保存: {filepath}")
        
        elif key == 27 or key == ord('q'):  # ESC 或 Q 键退出
            print("退出程序")
            break
    
    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 可以修改保存目录和摄像头ID
    take_photos(save_dir="hand_sign_data", camera_id=0)