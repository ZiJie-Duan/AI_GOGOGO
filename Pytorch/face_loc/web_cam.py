import cv2

# 初始化摄像头
cap = cv2.VideoCapture(0)  # 0代表计算机的默认摄像头

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # 从摄像头读取一帧
    ret, frame = cap.read()

    # 如果正确读取帧，ret为True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 在帧上绘制一个绿色的边框
    # 参数分别是：帧、左上角坐标、右下角坐标、颜色(B,G,R)、边框厚度
    cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), 3)

    # 显示结果帧
    cv2.imshow('Frame with Border', frame)

    # 按'q'退出循环
    if cv2.waitKey(1) == ord('q'):
        break

# 释放摄像头资源
cap.release()
# 关闭所有OpenCV窗口
cv2.destroyAllWindows()
