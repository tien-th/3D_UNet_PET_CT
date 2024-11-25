from torch.utils.data import Dataset
import numpy as np
import os

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, max_pixel=32767, flip=False, to_normal=False):
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.max_pixel = float(max_pixel)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = None
        try:
            np_image = np.load(img_path, allow_pickle=True)
            np_image = np_image / float(self.max_pixel)

        except BaseException as e:
            print(img_path)    
        np_image = (np_image - 0.5) * 2.
        # image_name = Path(img_path).stem
        np_image = np.expand_dims(np_image, axis=0)
        return np_image

def get_image_paths_from_dir(fdir):
    flist = os.listdir(fdir)
    flist.sort()
    image_paths = []
    for i in range(0, len(flist)):
        fpath = os.path.join(fdir, flist[i])
        if os.path.isdir(fpath):
            image_paths.extend(get_image_paths_from_dir(fpath))
        else:
            image_paths.append(fpath)
    return image_paths

class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_path , stage='train'):
        super().__init__()

        image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_path, f'{stage}/A'))
        image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_path, f'{stage}/B'))

        self.imgs_cond = ImagePathDataset(image_paths_cond) 
        self.imgs_ori = ImagePathDataset(image_paths_ori)  


    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        return self.imgs_ori[i], self.imgs_cond[i]