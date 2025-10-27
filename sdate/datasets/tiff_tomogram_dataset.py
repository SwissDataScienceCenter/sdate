import h5py
import torch
from PIL import Image
import math
import numpy as np
import os
from sdate.datasets.base_dataset import BaseImageDataset

Image.MAX_IMAGE_PIXELS = None

class tiff_wrapper():
    def __init__(self, path, im_size=512):
        if os.path.isdir(path):
            folder = [
                os.path.join(root, filename)
                for root, _, files in os.walk(path)
                for filename in files
                if filename.endswith(('.tiff', '.tif'))
            ]
        else:
            folder = [os.path.basename(path)]
            path = os.path.dirname(path)

        self.path = path
        self.folder = folder
        self.im_size = im_size
        sizes = []
        idx_to_file_list = []
        global_index_to_local_list = []
        coordinates_list = []

        for i, filename in enumerate(folder):
            image = Image.open(os.path.join(path, filename))
            w, h = image.size
            w_ = (w - im_size) / (math.ceil(w / im_size) - 1) if im_size != w else w
            h_ = (h - im_size) / (math.ceil(h / im_size) - 1) if im_size != h else h
            num_images = math.ceil(w / im_size) * math.ceil(h / im_size)
            row = torch.arange(num_images) // math.ceil(w / im_size)
            col = torch.arange(num_images) % math.ceil(w / im_size)
            x = col * w_
            y = row * h_

            coordinates_list.append(torch.stack([x, y], -1))
            idx_to_file_list.append(i * torch.ones(num_images).int())
            global_index_to_local_list.append(torch.arange(num_images))

        self.idx_to_file = torch.cat(idx_to_file_list)
        self.global_index_to_local = torch.cat(global_index_to_local_list)
        self.coordinates = torch.cat(coordinates_list)

    def __getitem__(self, idx):
        file_id = self.idx_to_file[idx]
        local_index = self.global_index_to_local[idx]
        coors = self.coordinates[idx].int().numpy()

        filename = self.folder[file_id]
        image = Image.open(os.path.join(self.path, filename))
        cropped_image = image.crop((coors[0], coors[1], coors[0] + self.im_size, coors[1] + self.im_size))
        return cropped_image

    def __len__(self):
        return len(self.idx_to_file)

class TIFFDataset(BaseImageDataset):

    def __init__(self, path, im_size,
                 rescale=None, clip_range=None, normalize_range=False, rotation_angle=None,
                 contrast=None, train_transform=False, crop=None):
        super().__init__(
            rescale=rescale, clip_range=clip_range, normalize_range=normalize_range, rotation_angle=rotation_angle,
            contrast=contrast, train_transform=train_transform, crop=crop
        )

        self.im_size = im_size
        self.images = tiff_wrapper(path, im_size)

    def __getitem__(self, idx):
        hr_image = torch.tensor(np.array(self.images[idx])).float()

        if len(hr_image.shape) == 2:
            hr_image = hr_image.unsqueeze(0)

        hr_image = self.transform(hr_image)

        return hr_image
    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    import lovely_tensors as lt
    import os

    lt.monkey_patch()

    im_size = 512

    kwargs = {
        "path": "/Users/lfbarba/GitHub/deep-anomaly-detection/data/downsampled/rec_16bit_phase_00043",
        "im_size": im_size,
        "train_transform": True,
        "rescale": im_size,
        "normalize_range": False,
    }
    trainSet = TIFFDataset(**kwargs)