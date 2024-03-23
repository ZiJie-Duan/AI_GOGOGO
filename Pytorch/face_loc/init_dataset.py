import os
import csv
import random
import cv2
import math
import scipy.io


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
        random.shuffle(self.dataset_index)

    def get_data(self, i):
        assert i < len(self.dataset_index), "Index out of range WF {} {}".format(i, len(self.dataset_index))
        i,j = self.dataset_index[i]
        return (self.get_file_path(i,j),self.get_face_bbx_list(i,j),None)
            

class ImgTransform:
    """
    Transforms for images
    We use cv2 for image processing
    1. Generate face bounding box via landmarks
    2. Random crop face bounding box 
        (iou>0.6 for positive, iou<0.2 for negative, 0.2<iou<0.55 for mixed)
    3. Modify landmarks and bounding box according to the crop

    sample type:
        l-landmark, p-positive, m-mixed, n-negative
    """

    def __init__(self, img_path, bbox=None, landmark=None):
        self.img_path = img_path
        self.img = cv2.imread(img_path,1)
        self.bbox = bbox        # [[x,y,w,h]]
        self.landmark = landmark # [[x1,y1,x2,y2,...]]
        self.hight, self.width = self.img.shape[:2]

        if self.landmark != None:
            self.update_bbox_via_landmark()
            self.datatype = "Full Sample" # landmark
        else:
            self.datatype = "Bbox Sample" # only bbox

    def update_bbox_via_landmark(self):
        ceb_landmark = self.landmark[0]
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
        self.bbox = [bbox]
        #show_img(ceb_img, [bbox], None)

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
    
    def random_cut(self, sample_type, bbox):
        """
        sample type here only has three types: p, m, n
        because Landmark not considered here
        p - positive, m - mixed, n - negative
        iou threshold: p>0.6, n<0.2, m-0.2~0.55
        """

        if sample_type == "p":
            change_step = max(bbox[2],bbox[3])//2
        elif sample_type == "m":
            change_step = max(bbox[2],bbox[3])*2
        else:
            change_step = self.hight//3

        count = 0
        while True:
            count += 1
            if count > 1000:
                raise Exception("1000 times search fail")
                
            # 先随机生成两个轴向的偏移量
            nx_shift = random.randint(-1,1) * random.randint(0,change_step)
            ny_shift = random.randint(-1,1) * random.randint(0,change_step)
            nx = bbox[0] + nx_shift
            ny = bbox[1] + ny_shift

            wh_max = max([bbox[2],bbox[3]])\
                + random.randint(-1,1) * random.randint(0,10)
            nh = wh_max # make sure the crop is square
            nw = wh_max

            # 裁剪安全检查
            if nx>=0 and ny>=0 and\
                (nx+nw)<=self.width and (ny+nh)<=self.hight:

                iou = self.cal_iou(
                    [bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]],
                    [nx,ny,nx+nw,ny+nh])

                if sample_type == "p" and iou > 0.6:
                    return [nx,ny,nw,nh]
                if sample_type == "n" and iou < 0.2:
                    return [nx,ny,nw,nh]
                if sample_type == "m" and iou > 0.2 and iou < 0.55:
                    return [nx,ny,nw,nh]
    

    def redefine_bbox(self, crop, bbox):
        # 通过裁剪后的图片和原始bbox，重新定义bbox
        x = bbox[0] - crop[0]
        y = bbox[1] - crop[1]
        w = bbox[2]
        h = bbox[3]
        return [x,y,w,h]
    

    def img_size_check(self, bbox):
        # 检查裁剪后的边框是否符合要求
        if bbox[2] < 24 or bbox[3] < 24:
            return False
        return True

    def generate_wfd_sample(self):

        sample_list = [] # 生成的样本列表
        """
        [
            [(img,[nbx,nby,nbw,nbh]), xxx, xxx],
            [positive, mixed, negative],
            ...
        ]
        """

        for bbox in self.bbox:

            sample_trip = [] # 生成的样本列表
            if self.img_size_check(bbox) == False:
                continue

            # 生成三种样本
            # p-positive 正样本，用于人脸识别 和 边框检测训练
            # m-mixed 混合样本，用于人脸识别 和 边框检测训练
            # n-negative 负样本，用于人脸识别
            try:
                for type_of_sample in ["p", "m", "n"]:
                    nimgb = self.random_cut(type_of_sample, bbox)
                    nbbox = self.redefine_bbox(nimgb, bbox)

                    nimg = self.img[nimgb[1]:nimgb[1]+nimgb[3],
                                   nimgb[0]:nimgb[0]+nimgb[2], :]
                    
                    sample_trip.append((nimg, nbbox))

            except Exception as e:
                print(e)
                continue

            sample_list.append(sample_trip)
        
        return sample_list # allow empty list
    

    def redefine_landmark(self, crop, landmark):
        # 通过裁剪后的图片和原始landmark，重新定义landmark
        nlandmark = landmark[:]
        for i in range(1,len(landmark),2):
            nlandmark[i-1] -= crop[0]
            nlandmark[i] -= crop[1]
        return nlandmark
    

    def generate_ceb_sample(self):
        
        bbox = self.bbox[0] # only one bbox

        if self.img_size_check(bbox) == False:
            return [] # allow empty list
        
        try:
            nimgb = self.random_cut("p", bbox)
        except Exception as e:
            print(e)
            return [] # allow empty list
        
        # only one landmark here for ceb, so landmark[0]
        nlandmark = self.redefine_landmark(nimgb, self.landmark[0])

        nimg = self.img[nimgb[1]:nimgb[1]+nimgb[3],
                        nimgb[0]:nimgb[0]+nimgb[2], :]
        
        return [nimg, nlandmark] # different from wfd sample
    

class BuildDataset:
    
    def __init__(self, wfdd, ceba, imgs_path, csv_path):
        self.wfdd = wfdd
        self.ceba = ceba
        self.wfd_index = wfdd.dataset_index
        self.ceb_index = ceba.dataset_index

        self.imgs_path = imgs_path
        self.csv_path = csv_path

        self.sample_index = 0


    def write_csv(self, line_elements):

        for i in range(len(line_elements)):
            if type(line_elements[i]) == list:
                line_elements[i] = " ".join([str(x) for x in line_elements[i]])
            else:
                line_elements[i] = str(line_elements[i])

        with open(self.csv_path, mode='a', encoding='utf-8', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(line_elements)

    def save_img(self, img):
        path = self.imgs_path + "\\" + str(self.sample_index) + ".jpg"
        cv2.imwrite(path, img)
    

    def generate_dataset(self):

        j = 0
        for i in range(len(self.wfd_index)):

            print("Processing {}/{}".format(i, len(self.wfd_index)))

            wfd_imgp, wfd_bbox, _ = self.wfdd.get_data(i)

            # 生成样本
            img_samples = ImgTransform(wfd_imgp, wfd_bbox).generate_wfd_sample()
            if len(img_samples) == 0:
                print("No sample generated for {}".format(wfd_imgp))
                continue
            
            # ！！！！！！！！！！！！十分重要！！！！！！！！！！！！！
            # 利用 wfd的数据结构 [positive, mixed, negative] “img_samples”
            # 在ceb中生成样本 加入到 [positive, mixed, negative, landmark] 中

            count = len(img_samples)
            while count > 0:
                ceb_imgp, _, ceb_landmark = self.ceba.get_data(j)
                j += 1

                ceb_samples = ImgTransform(ceb_imgp, None, ceb_landmark)\
                    .generate_ceb_sample()
                if len(ceb_samples) == 0:
                    print("No sample generated for {}".format(ceb_imgp))
                    continue
                
                img_samples[count-1].append(ceb_samples)
                count -= 1
            

            # 保存样本
            # [positive, mixed, negative, landmark]
            for sample_group in img_samples:

                csv_line = []
                for sample in sample_group:
                    csv_line += [self.sample_index, sample[1]]
                    self.save_img(sample[0])
                    self.sample_index += 1

                self.write_csv(csv_line)




bbox_path = r"C:\Users\lucyc\Desktop\celebA\list_bbox_celeba.csv"
ldmk_path = r"C:\Users\lucyc\Desktop\celebA\list_landmarks_align_celeba.csv"
basic_path = r"C:\Users\lucyc\Desktop\celebA\img_align_celeba\img_align_celeba"

cead = CELEBADriver(bbox_path, ldmk_path, basic_path)

mat_path = r"C:\Users\lucyc\Desktop\faces\WIDER_train\WIDER_train\images"
clas_root_path = r"C:\Users\lucyc\Desktop\faces\wider_face_split\wider_face_split\wider_face_train.mat"

wfd = WFDriver(clas_root_path, mat_path)

cead.random_init()
wfd.random_init()


bds = BuildDataset(wfd, cead, r"C:\Users\lucyc\Desktop\ttds", r"C:\Users\lucyc\Desktop\dataset.csv")
bds.generate_dataset()