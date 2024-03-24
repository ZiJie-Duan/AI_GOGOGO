from torch.utils.data import Dataset
from PIL import Image
import csv
import cv2
import matplotlib.pyplot as plt
import torch

class FLCDataset(Dataset):
    
    def __init__(self, csv_dir, img_dir, transform=None):
        self.csv_dir = csv_dir
        self.img_dir = img_dir
        self.transform = transform
        self.datalines = self.read_csv_file(csv_dir)

        self.sample_type = 0
        # 0 positive, 1 mixed, 2 negative, 3 landmark


    def __len__(self):
        return len(self.datalines)
    

    def __get_one_on_forth(self):
        pass


    def __getitem__(self, idx):
        # 修复问题，数据集设计，每一次返回 必须是一个图像和一个标签
        # 缩放标尺计算，缩放bbox和landmark 正确的缩放大小
        # 对于数据集 加入一个新的维度 用于标注其 数据的类型
        # 考虑直接从 getitem 中返回一个图像和一个标签
        # 修改 init dataset 存储数据的方式

        # 假设data_info是一个包含图像路径和标签的列表
        data_info = self.datalines[idx]
        
        # 初始化一个用于存储图像和标签Tensor的列表
        data_unit = []
        
        for i in range(0, len(data_info), 2):
            # 读取图像
            img = self.read_img(data_info[i])
            
            # 处理标签数据
            args_str = data_info[i+1].split()
            args = [int(x) for x in args_str]

            # get the difference between the img size and 224 standard size
            width, height = img.size
            x_scale = 224 // width
            y_scale = 224 // height

            # scale the bbox and landmark
            for k in range(len(args)):
                if k % 2 == 0:
                    args[k] = int(args[k] * x_scale)
                else:
                    args[k] = int(args[k] * y_scale)

            # 将标签列表转换为Tensor
            labels_tensor = torch.tensor(args, dtype=torch.long)  # 或者使用torch.float32根据需要

            if self.transform:
                img = self.transform(img)
            
            # 将图像Tensor和标签Tensor添加到data_unit列表
            data_unit.append([img, labels_tensor])
        
        # 在这个点，data_unit是一个列表，其中每个元素都是[img_tensor, labels_tensor]
        # 通常，你会在这里返回单个图像和标签，而不是列表
        # 如果你的目的是构建批次，应该让DataLoader的collate_fn来处理
        # 例如，返回单个图像和标签：return img_tensor, labels_tensor
        
        return data_unit  # 注意：这种方式不适合标准DataLoader流程
        
    def read_img(self, i):
        img = Image.open(self.to_path(i))
        return img
    
    def read_csv_file(self, file_path):
        """
        read the csv file
        :param file_path: the path of the csv file
        :return: the list of the csv file
        """
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
        return lines

    def to_path(self, i):
        path = r"C:\Users\lucyc\Desktop\face_loc_d" + "\\" + str(i) + ".jpg"
        return path
    
        
def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """反标准化PyTorch tensor图像"""
    for t, m, s in zip(tensor, mean, std):  # 对每个通道进行操作
        t.mul_(s).add_(m)  # 对应于 (x * std) + mean
    return tensor

def visualize_transformed_image(tensor_image, bbox, landmark, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    反标准化并可视化PyTorch tensor图像。
    
    Parameters:
    - tensor_image: PyTorch tensor，代表经过transform变换的图像。
    - mean: 标准化时使用的均值，应与transform操作中的均值相对应。
    - std: 标准化时使用的标准差，应与transform操作中的标准差相对应。
    """
    # 克隆图像tensor以避免修改原始数据，并进行反标准化处理
    unnormalized_image = unnormalize(tensor_image.clone().detach(), mean, std)
    
    # 将tensor图像转换为NumPy数组，并调整形状为HxWxC以适应matplotlib
    img = unnormalized_image.numpy().transpose((1, 2, 0))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    b = bbox
    # 在图片上画矩形框，参数分别是：图片、左上角坐标、右下角坐标、颜色（BGR格式）、线条厚度
    img = cv2.rectangle(img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), (255, 0, 0), 1)

    if landmark != None:
        for x, y in [(landmark[i],landmark[i+1]) for i in range(0,len(landmark),2)]:
            img = cv2.rectangle(img, (x,y), (x,y), (0, 0, 255), 1)
            
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')  # 不显示坐标轴
    plt.show()



