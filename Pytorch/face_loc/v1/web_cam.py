
import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F

IMG_INPUT_SIZE = [12,12]
device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")
# 定义转换操作
transform = transforms.Compose([
    transforms.Resize(IMG_INPUT_SIZE[0]),
    transforms.CenterCrop(IMG_INPUT_SIZE[0]),
    transforms.ToTensor(),  # 将PIL图像或NumPy ndarray转换为FloatTensor。
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化，使用ImageNet的均值和标准差
                         std=[0.229, 0.224, 0.225])
])

class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()

        # 定义网络层
        self.conv1 = nn.Conv2d(3, 10, 3)  #12 -> 10 -> maxp -> 5
        self.conv2 = nn.Conv2d(10, 16, 3) #5 -> 3
        self.conv3 = nn.Conv2d(16, 32, 3) #3 -> 1

        self.face_det = nn.Conv2d(32, 2, 1) #1 -> 1
        self.bbox = nn.Conv2d(32, 4, 1) #1 -> 1
        self.landmark = nn.Conv2d(32, 10, 1) #1 -> 1

    def forward(self, x):
        # 定义前向传播
        x = F.relu(self.conv1(x)) #10
        x = F.max_pool2d(x, 2) #5
        x = F.relu(self.conv2(x)) #3
        x = F.relu(self.conv3(x)) #1

        facedet = self.face_det(x)
        bbox = self.bbox(x)
        landmark = self.landmark(x)

        facedet = torch.flatten(facedet, 1)
        bbox = torch.flatten(bbox, 1)
        landmark = torch.flatten(landmark, 1)

        return facedet, bbox, landmark


def generate_image_pyramid(img, scale_factor=1.2, min_size=(24, 24)):
    """
    生成图像的金字塔。
    
    :param img: 原始图像
    :param scale_factor: 缩放因子
    :param min_size: 图像在金字塔中的最小尺寸
    :return: 金字塔图像列表
    """
    pyramid_images = []
    scale_factor_base = 5

    while True:
        new_width = int(img.shape[1] / scale_factor_base)
        new_height = int(img.shape[0] / scale_factor_base)

        if new_width < min_size[0] or new_height < min_size[1]:
            break

        img2 = cv2.resize(img, (new_width, new_height))
        pyramid_images.append([img2, scale_factor_base])
        scale_factor_base *= scale_factor # 可以调整以控制金字塔的级别间隔

    return pyramid_images


def sliding_window(image, step_size, window_size, model_trained):
    """
    对图像应用滑动窗口，并使用提供的模型检测人脸。
    
    :param image: 输入的原始图像
    :param step_size: 每次滑动的像素数
    :param window_size: 窗口大小 (宽度, 高度)
    :param model_trained: 训练好的人脸检测模型
    """
    # 图像尺寸
    (h, w) = image.shape[:2]

    result = []
    
    # 逐步移动窗口
    for y in range(0, h - window_size[1], step_size):
        for x in range(0, w - window_size[0], step_size):
            # 提取当前窗口的图像片段
            window = image[y:y + window_size[1], x:x + window_size[0]]

            image_rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
            # 将NumPy数组转换为PIL.Image对象
            image_pil = Image.fromarray(image_rgb)
            

            window_tensor = transform(image_pil).unsqueeze(0).to(device)
            x_scale = 12 / w
            y_scale = 12 / h

            with torch.no_grad():
                face_det, bbox, _ = model_trained(window_tensor)
            
            if face_det[0][0] - face_det[0][1] > 2:
                result.append((x, y, window_size[0], window_size[1]))
                # nx = bbox[0][0].item() * x_scale + x
                # ny = bbox[0][1].item() * y_scale + y
                # nw = bbox[0][2].item() * x_scale
                # nh = bbox[0][3].item() * y_scale
                # result.append((nx, ny, nw, nh))
                
        
    return result


# 初始化摄像头
cap = cv2.VideoCapture(0)  # 0代表计算机的默认摄像头


model_trained = torch.load(r"C:\Users\lucyc\Desktop\AI_GOGOGO\Pytorch\face_loc\face_loc_p_1.pth")
model_trained.eval()  # 设置模型为评估/测试模式
# face_det, bbox, landmark = model_trained(frame)

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

    pyramid = generate_image_pyramid(frame)

    result = []
    for img, scal in pyramid:
        res = sliding_window(img, step_size=12, window_size=(24, 24), model_trained=model_trained)
        res = [[x*scal for x in y] for y in res]
        result += res


    for x, y, w, h in result:
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # 显示结果帧

    cv2.imshow('Frame with Border', frame)

    # 按'q'退出循环
    if cv2.waitKey(1) == ord('q'):
        break

# 释放摄像头资源
cap.release()
# 关闭所有OpenCV窗口
cv2.destroyAllWindows()
