from torch.utils.data import Dataset
from PIL import Image

class TrpDataSet(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        #anchor, positive, negative = (1,2,3)
        # # 读取文件内容
        anchor = Image.open(file_path[0])
        positive = Image.open(file_path[1])
        negative = Image.open(file_path[2])

        # 这里可以添加任何自定义的转换，例如将文本转换为张量等
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
            
        return anchor, positive, negative