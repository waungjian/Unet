from torch.utils.data import Dataset
import torch
from PIL import Image


class Mydataset(Dataset):
    def __init__(self,img_paths,anno_paths,transform):
        super(Mydataset, self).__init__()
        self.img_paths = img_paths
        self.anno_paths = anno_paths
        self.transform = transform
    def __getitem__(self, item):
        pil_img = Image.open(self.img_paths[item]).convert("RGB")
        tensor_img = self.transform(pil_img)
        pil_anno = Image.open(self.anno_paths[item])
        tensor_anno = self.transform(pil_anno)
        tensor_anno[tensor_anno>0] = 1
        tensor_anno = torch.squeeze(tensor_anno).type(torch.long)
        return tensor_img,tensor_anno
    def __len__(self):
        return len(self.img_paths)

