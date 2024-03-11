from torch.utils.data import Dataset
from PIL import Image
import scipy.io
import os
import csv
import random
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math



class CELEBADriver:
    
    #['image_id', 'lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y', 'nose_x', 'nose_y','leftmouth_x', 'leftmouth_y', 'rightmouth_x', 'rightmouth_y']
    #['000001.jpg', '69', '109', '106', '113', '77', '142', '73', '152', '108', '154']
    #['image_id', 'x_1', 'y_1', 'width', 'height']
    #['000001.jpg', '95', '71', '226', '313']

    def __init__(self, bbox_path, landmarks_path, imgs_path):
        self.bbox_path = bbox_path
        self.landmarks_path = landmarks_path
        self.imgs_path = imgs_path
        self.data = {}
        self.dataset_index = []  # 样本汇总
        self.init_dataset()

    def init_dataset(self):
        with open(self.bbox_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            count = 0
            for row in csv_reader:
                if count == 0:
                    count += 1
                    continue
                self.data[count] = {"name":row[0]}
                self.data[count]["bbox"] = [int(x) for x in row[1:]]
                count += 1
                
        with open(self.landmarks_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            count = 0
            for row in csv_reader:
                if count == 0:
                    count += 1
                    continue
                self.data[count]["ldmk"] = [int(x) for x in row[1:]]
                count += 1

        self.dataset_index = [x for x in range(count)]

    def get_file_path(self, index):
        # index 获取一个图片文件路径
        return self.imgs_path + "\\" + self.data[index]["name"]

    def get_face_bbx_list(self, index):
        return [self.data[index]["bbox"]]

    def get_face_ldmk_list(self, index):
        return [self.data[index]["ldmk"]]

    def random_init(self):
        if input("!!! SERIOUS RANDOM INIT DATASET ALARM !!! type ‘y’ to continue... ") == 'y':
            print("RANDOM SET")
            random.shuffle(self.dataset_index)

    def get_data(self, i):
        i = self.dataset_index[i]
        return (self.get_file_path(i),self.get_face_bbx_list(i),self.get_face_ldmk_list(i))

class WFDriver:

    def __init__(self, mat_path, clas_root_path):
        self.data = scipy.io.loadmat(mat_path) # 读取 mat 文件到 内存中
        self.r_path = clas_root_path # 类别文件 系统根目录
        self.clas_i_map = {} # 我们利用文件命名中的数字编号 作为索引
        # 建立clas_i_map 将文件命名中的编号 和 mat数据中索引进行关联
        self.clas_map = {} # 将 文件命名中的编号 和 类别文件夹的全名
        self.dataset_index = [] # 样本汇总

        self.init_clas_map() 
        
    def init_clas_map(self):
        # 初始化 类别标签
        # 我们利用文件命名中的数组编号 作为索引
        # 建立clas_i_map 将文件命名中的编号 和 数据中随机索引进行关联
        for i in range(len(self.data["file_list"])):
            j = int(self.data["file_list"][i][0][0][0][0].split("_")[0])
            self.clas_i_map[j] = i

        # 生成clas_map，将类别编号 和其 名称绑定
        for _, dirs, files in os.walk(self.r_path):
            for dir in dirs:

                clas_i = int(dir.split("--")[0]) # 根据目录名字 生成类别索引
                self.clas_map[clas_i] = dir      # 将类别索引 与 目录名字 建立连接

                num_smp = len(self.data["file_list"][self.clas_i_map[clas_i]][0]) # 求出一个类别有多少个样本
                self.dataset_index += [(clas_i, x) for x in range(num_smp)] # 并入 样本数组

            
    def get_file_path(self, clas, index):
        # 通过 clas 和 index 获取一个图片文件信息
        return self.r_path + "\\" + self.clas_map[clas]\
                + "\\" + self.data["file_list"][self.clas_i_map[clas]][0][index][0][0] + ".jpg"

    def get_face_bbx_list(self, clas, index):
        return self.data["face_bbx_list"][self.clas_i_map[clas]][0][index][0]

    def random_init(self):
        if input("!!! SERIOUS RANDOM INIT DATASET ALARM !!! type ‘y’ to continue... ") == 'y':
            print("RANDOM SET")
            random.shuffle(self.dataset_index)

    def get_data(self, i):
        assert i < len(self.dataset_index), "Index out of range WF {} {}".format(i, len(self.dataset_index))
        i,j = self.dataset_index[i]
        return (self.get_file_path(i,j),self.get_face_bbx_list(i,j),None)
            

class MTCDataSet(Dataset):
    
    def __init__(self, wfdd, ceba, split = [] ,transform = None):
        self.wfdd = wfdd
        self.ceba = ceba
        self.wfd_index = wfdd.dataset_index
        self.ceb_index = ceba.dataset_index
        self.max_index = len(self.wfd_index)
        self.transform = transform
        
        self.shift = 0
        self.limit = 0
        if split:
            self.shift = split[0]
            self.limit = split[1]

    def _2index(self, index):
        return index + self.shift

    def __len__(self):
        if self.shift == 0 and self.limit == 0:
            return self.max_index
        else:
            return self.limit - self.shift

    def __getitem__(self, idx):

        idx = self._2index(idx)

        data_trip = [] # [(img, bbox, landmark), (sample2),  (sample3)]
        
        wfd_imgp, wfd_bbox, _ = self.wfdd.get_data(idx)
        ceb_imgp, _, ceb_landmark = self.ceba.get_data(idx)
        
        # 读取文件内容
        #cv2.imread(path2,1)
        #image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        wfd_img = cv2.imread(wfd_imgp,1)
        ceb_img = cv2.imread(ceb_imgp,1)

        # p-positive 正样本，用于人脸识别 和 边框检测训练
        # m-mixed 混合样本，用于人脸识别 和 边框检测训练
        # n-negative 负样本，用于人脸识别
        # a-landmark 特征点样本，用于人脸识别 和 特征点检测
        for type_of_sample in ["p", "m", "n"]:
            for img, bbox in self.wfd_random_cut(wfd_img, wfd_bbox, type_of_sample):
                bbox = self.resize_bbox(bbox, len(img))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = self.transform(img) # 需要指定使用缩放金字塔进行处理
                data_trip.append((type_of_sample, img, bbox, None))

        for img, bbox, landmark in self.ceb_random_cut_s(ceb_img, ceb_landmark):
            bbox = self.resize_bbox(bbox, len(img))
            landmark = self.resize_bbox(landmark, len(img))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = self.transform(img) # 需要指定使用缩放金字塔进行处理
            data_trip.append(("a", img, bbox, landmark))
            
        # BUG 等比例 缩放 Bbox ，因为 transform 缩放至 12像素时，其他的也要等比例缩放
            
        if data_trip == [] or data_trip == [(None, None)] or data_trip == [(None, None, None)]: # 如果没有数据
            return self.__getitem__(random.randint(self.shift, self.limit))
        
        return data_trip

    def cal_iou(self, boxA, boxB):
        # 计算两个边界框的坐标
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
    
        # 计算交集的面积
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
        # 计算两个边界框的面积
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
        # 计算并集的面积
        iou = interArea / float(boxAArea + boxBArea - interArea)
    
        # 返回计算出的IoU值
        return iou

    def resize_bbox(self, bbox, img_size):
        n_bbox = []
        for loc in bbox:
            n_bbox.append((loc * 12)//img_size)
        return n_bbox
        
    def wfd_random_cut(self, img, bboxs, type):
        # 结果 边框
        imgs_bboxs = []
        # 从所有 标记数据中 选择 人脸边框
        for bbox in bboxs:
    
            # 确保 所有的已标记边框 长度 大于12像素
            if sum(bbox[2:]) < 24:
                continue
            height, width = img.shape[:2]
    
            # 设置三种裁剪模式 用于优化执行速度
            # p-positive 正样本模式，移动距离最小，最靠近原有边框
            # m-mixed 混合模式，移动距离中等，裁取一半原有边框
            # n-negative 负样本模式，移动距离最大，采取不相干的样本
            if type == "p":
                change_step = max(bbox[2],bbox[3])//2
            elif type == "m":
                change_step = max(bbox[2],bbox[3])*2
            else:
                change_step = height//3
    
            # 无限循环保险
            safe_counter = 0
            while safe_counter < 100:
                safe_counter += 1
                # 无限循环保险
                sec_safe_counter = 0
                while sec_safe_counter < 100:
                    sec_safe_counter += 1
    
                    # 先随机生成两个轴向的偏移量
                    nx_shift = random.randint(0,change_step)
                    ny_shift = random.randint(0,change_step)
                    nx = bbox[0] + nx_shift
                    ny = bbox[1] + ny_shift
                    # 通过随机数 进行 正负偏移
                    if random.randint(0,1):
                        nx = nx - 2*nx_shift
                    if random.randint(0,1):
                        ny = ny - 2*ny_shift
    
                    # 裁剪安全检查
                    if nx>=0 and ny>=0 and nx<=width and ny<=height:
                        break
                
                if sec_safe_counter >= 100:
                    continue #search fail break the loop
    
                # 设置最大剪裁边框，三个神经网络的输入都是正方形，两边同长
                max_length = max([bbox[2],bbox[3]])
                nh = max_length
                nw = max_length
    
                boxA = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
                boxB = [nx, ny, nw+nx, nh+ny]
                iou = self.cal_iou(boxA,boxB) # 计算 iou 来决策是否是我们要的样本
    
                # 移动原有的边框空间位置，让其作用于新的空间
                nbx = bbox[0] - nx
                nby = bbox[1] - ny
                nbw = bbox[2]
                nbh = bbox[3]
    
                # 通过不同的 iou 划分不同模式下要的样本
                if type == "p" and iou > 0.6:
                    imgn = img[ny:ny+nh, nx:nx+nw, :]
                    imgs_bboxs.append((imgn, (nbx,nby,nbw,nbh)))
                    break
                if type == "n" and iou < 0.2:
                    imgn = img[ny:ny+nh, nx:nx+nw, :]
                    imgs_bboxs.append((imgn, (nbx,nby,nbw,nbh)))
                    break
                if type == "m" and iou > 0.2 and iou < 0.55 :
                    imgn = img[ny:ny+nh, nx:nx+nw, :]
                    imgs_bboxs.append((imgn, (nbx,nby,nbw,nbh)))
                    break

        if imgs_bboxs == []:
            return [(None, None)]
        return imgs_bboxs

    def ceb_random_cut(self, img, bbox, landmark, num):
        # 结果 边框
        imgs_bboxs_landmark = []
        
        # 从所有 标记数据中 选择 人脸边框
        for _ in range(num):
            height, width = img.shape[:2]
            change_step = max(bbox[2],bbox[3])//2

            # 无限循环保险
            safe_counter = 0
            while safe_counter < 100:
                safe_counter += 1
                # 无限循环保险
                sec_safe_counter = 0
                while sec_safe_counter < 100:
                    sec_safe_counter += 1
    
                    # 先随机生成两个轴向的偏移量
                    nx_shift = random.randint(0,change_step)
                    ny_shift = random.randint(0,change_step)
                    nx = bbox[0] + nx_shift
                    ny = bbox[1] + ny_shift
                    # 通过随机数 进行 正负偏移
                    if random.randint(0,1):
                        nx = nx - 2*nx_shift
                    if random.randint(0,1):
                        ny = ny - 2*ny_shift
    
                    # 裁剪安全检查
                    if nx>=0 and ny>=0 and nx<=width and ny<=height:
                        break
                
                if sec_safe_counter >= 100:
                    break #search fail break the loop
    
                # 设置最大剪裁边框，三个神经网络的输入都是正方形，两边同长
                max_length = max([bbox[2],bbox[3]])
                nh = max_length
                nw = max_length
    
                boxA = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
                boxB = [nx, ny, nw+nx, nh+ny]
                iou = self.cal_iou(boxA,boxB) # 计算 iou 来决策是否是我们要的样本
    
                # 移动原有的边框空间位置，让其作用于新的空间
                nbx = bbox[0] - nx
                nby = bbox[1] - ny
                nbw = bbox[2]
                nbh = bbox[3]

                landmarkn = landmark[:]
                for i in range(1,len(landmark),2):
                    landmarkn[i-1] -= nx
                    landmarkn[i] -= ny
    
                if iou > 0.6:
                    imgn = img[ny:ny+nh, nx:nx+nw, :]
                    imgs_bboxs_landmark.append((imgn, (nbx,nby,nbw,nbh), landmarkn))
                    break

        if imgs_bboxs_landmark == []:
            return [(None, None, None)]
        return imgs_bboxs_landmark

 
    def ceb_random_cut_s(self, ceb_img, ceb_landmark, num = 3):
        
        ceb_landmark = ceb_landmark[0]
        
        # 结果 标记点
        imgs_landmark = []

        # 定位左右眼
        leye = ceb_landmark[:2]
        reye = ceb_landmark[2:4]

        # 求解双目距离
        dis = math.sqrt((abs(leye[0] - reye[0])**2) + (abs(leye[1] - reye[1])**2))
        dis = dis//1

        # 求解 嘴部图像最低点
        lower = min(ceb_landmark[-1], ceb_landmark[-3])

        # 硬编码 面部关系
        x = leye[0] - dis//1.5
        y = leye[1] - dis
        w = dis*2.5
        h = lower - y + dis
        
        bbox = [int(x) for x in [x,y,w,h]]
        #show_img(ceb_img, [bbox], None)

        ceb_random_cut = self.ceb_random_cut(ceb_img, bbox, ceb_landmark, num)
        
        return ceb_random_cut
        
        
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


def get_trp_dataset():
    # 定义转换操作
    transform = transforms.Compose([
        transforms.Resize(12),
        transforms.CenterCrop(12),
        transforms.ToTensor(),  # 将PIL图像或NumPy ndarray转换为FloatTensor。
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化，使用ImageNet的均值和标准差
                             std=[0.229, 0.224, 0.225])
    ])

    bbox_path = r"C:\Users\lucyc\Desktop\celebA\list_bbox_celeba.csv"
    ldmk_path = r"C:\Users\lucyc\Desktop\celebA\list_landmarks_align_celeba.csv"
    basic_path = r"C:\Users\lucyc\Desktop\celebA\img_align_celeba\img_align_celeba"
    
    cead = CELEBADriver(bbox_path, ldmk_path, basic_path)
    
    mat_path = r"C:\Users\lucyc\Desktop\faces\WIDER_train\WIDER_train\images"
    clas_root_path = r"C:\Users\lucyc\Desktop\faces\wider_face_split\wider_face_split\wider_face_train.mat"
    
    wfd = WFDriver(clas_root_path, mat_path)

    cead.random_init()
    wfd.random_init()

    # 创建数据集
    test_dataset = MTCDataSet(wfd, cead, [0,7728], transform)
    val_dataset = MTCDataSet(wfd, cead, [7728,10304], transform)
    train_dataset = MTCDataSet(wfd, cead, [10304,12870], transform)

    return test_dataset, val_dataset, train_dataset