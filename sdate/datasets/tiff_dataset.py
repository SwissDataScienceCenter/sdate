import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile
import numpy as np
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor


class TiffDataset(Dataset):
    def __init__(self, source_path, transform=None, rescale=-1, clip_range=None, max_workers=8, normalize=False):
        """
        Preloads all TIFF images from a folder or multi-frame TIFF file into memory using parallel loading.

        Args:
            source_path (str): Either a directory containing TIFF images or a path to a multi-frame TIFF file.
            transform (callable, optional): Transform to apply to each image.
            rescale (int or tuple, optional): If not -1, rescale the images to the given size.
            clip_range (tuple, optional): A tuple specifying the clip range.
            max_workers (int, optional): Number of threads for parallel image loading.
        """
        self.transform = transform
        self.images = []
        self.clip_range = clip_range
        self.normalize = normalize

        if os.path.isdir(source_path):
            # Collect all TIFF file paths in the directory
            image_paths = sorted(
                glob.glob(os.path.join(source_path, '*.tif')) +
                glob.glob(os.path.join(source_path, '*.tiff'))
            )

            # Helper function to load and process each image file.
            def load_and_process(path):
                image = tifffile.imread(path)
                if not isinstance(image, np.ndarray):
                    image = np.array(image)
                # If the loaded image has multiple frames, process each frame.
                if image.ndim > 2:
                    return [self._process_frame(frame, rescale) for frame in image]
                else:
                    return self._process_frame(image, rescale)

            # Use ThreadPoolExecutor to parallelize the loading process.
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(load_and_process, image_paths))

            # Flatten the results since some entries may be lists of frames.
            for result in results:
                if isinstance(result, list):
                    self.images.extend(result)
                else:
                    self.images.append(result)

        elif os.path.isfile(source_path):
            image = tifffile.imread(source_path)
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            if image.ndim > 2:
                for frame in image:
                    self.images.append(self._process_frame(frame, rescale))
            else:
                self.images.append(self._process_frame(image, rescale))
        else:
            raise ValueError("source_path does not exist or is not a valid file/directory.")

    def _process_frame(self, frame, rescale):
        """Convert a numpy frame to a torch tensor and apply rescaling if needed."""
        img_tensor = torch.from_numpy(frame.astype(np.float32))
        if rescale != -1:
            # Add batch and channel dimensions for F.interpolate.
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0).unsqueeze(0),
                size=rescale,
                mode='bilinear'
            ).squeeze()
        return img_tensor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        # Add a channel dimension if needed.
        if self.normalize:
            image = (image - image.min()) / (image.max() - image.min())
            
        return image.unsqueeze(0) if self.clip_range is None else image.unsqueeze(0).clip(self.clip_range[0],
                                                                                          self.clip_range[1])


# Example usage:
if __name__ == '__main__':
    source_path = '/path/to/your/tiff_directory_or_file'
    dataset = TiffDataset(source_path=source_path, rescale=(256, 256))
    print(f"Dataset length: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=True)
    for batch in loader:
        print(batch.shape)