from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision
import random
import os
import re


IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.JPG', '.PNG', '.JPEG')


def augment(hazy, clean):
    augmentation_method = random.choice([0, 1, 2, 3, 4, 5])
    rotate_degree = random.choice([90, 180, 270])

    if augmentation_method == 0:
        hazy = transforms.functional.rotate(hazy, rotate_degree)
        clean = transforms.functional.rotate(clean, rotate_degree)
        return hazy, clean

    if augmentation_method == 1:
        vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)
        hazy = vertical_flip(hazy)
        clean = vertical_flip(clean)
        return hazy, clean

    if augmentation_method == 2:
        horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
        hazy = horizontal_flip(hazy)
        clean = horizontal_flip(clean)
        return hazy, clean

    return hazy, clean


def is_image_file(filename):
    return filename.endswith(IMG_EXTENSIONS)


def extract_pair_key(filename):
    """
    Extract a dataset-independent pairing key.

    Works for your datasets:
      Dense-Haze / NH-Haze:
        hazy: 01_hazy.png   -> key '1'
        gt:   01_GT.png     -> key '1'

      HD-NH-Haze:
        001.JPG            -> key '1'

      NH-Haze2:
        041.png            -> key '41'

      SOTS:
        hazy: 1400_10.png  -> key '1400'
        gt:   1400.png     -> key '1400'
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r'^(\d+)', stem)
    if m is None:
        raise ValueError(f"Cannot extract pairing key from filename: {filename}")
    return str(int(m.group(1)))   # remove leading zeros safely


def build_file_map(folder):
    files = [f for f in os.listdir(folder) if is_image_file(f)]
    mapping = {}
    for f in files:
        key = extract_pair_key(f)
        if key in mapping:
            raise RuntimeError(f"Duplicate key '{key}' found in folder: {folder}")
        mapping[key] = f
    return mapping


class PairedDehazeDataset(Dataset):
    """
    Universal paired dehazing dataset.

    Expected structure:
        dataset_root/
            train/
                hazy/
                gt/
            test/
                hazy/
                gt/

    Usage:
        train_set = PairedDehazeDataset(dataset_root, split='train', crop_size=384, augment_data=True)
        test_set  = PairedDehazeDataset(dataset_root, split='test',  crop_size=None, augment_data=False)
    """
    def __init__(self, dataset_root, split='train', crop_size=384, augment_data=True):
        super().__init__()

        assert split in ['train', 'test'], "split must be 'train' or 'test'"

        self.dataset_root = dataset_root
        self.split = split
        self.crop_size = crop_size if split == 'train' else None
        self.augment_data = augment_data if split == 'train' else False
        self.transform = transforms.ToTensor()

        self.hazy_dir = os.path.join(dataset_root, split, 'hazy')
        self.gt_dir = os.path.join(dataset_root, split, 'gt')

        if not os.path.isdir(self.hazy_dir):
            raise FileNotFoundError(f"Hazy directory not found: {self.hazy_dir}")
        if not os.path.isdir(self.gt_dir):
            raise FileNotFoundError(f"GT directory not found: {self.gt_dir}")

        hazy_map = build_file_map(self.hazy_dir)
        gt_map = build_file_map(self.gt_dir)

        common_keys = sorted(set(hazy_map.keys()) & set(gt_map.keys()), key=lambda x: int(x))

        missing_in_gt = sorted(set(hazy_map.keys()) - set(gt_map.keys()), key=lambda x: int(x))
        missing_in_hazy = sorted(set(gt_map.keys()) - set(hazy_map.keys()), key=lambda x: int(x))

        if missing_in_gt:
            raise RuntimeError(f"Missing GT files for keys: {missing_in_gt}")
        if missing_in_hazy:
            raise RuntimeError(f"Missing hazy files for keys: {missing_in_hazy}")
        if len(common_keys) == 0:
            raise RuntimeError(f"No matched hazy/gt pairs found in {dataset_root}/{split}")

        self.samples = []
        for key in common_keys:
            hazy_path = os.path.join(self.hazy_dir, hazy_map[key])
            gt_path = os.path.join(self.gt_dir, gt_map[key])
            self.samples.append((hazy_path, gt_path, key))

        print(f"[PairedDehazeDataset] Loaded {len(self.samples)} pairs from {dataset_root} ({split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        hazy_path, gt_path, key = self.samples[index]

        hazy = Image.open(hazy_path).convert('RGB')
        clean = Image.open(gt_path).convert('RGB')

        if hazy.size != clean.size:
            raise RuntimeError(
                f"Size mismatch for key {key}: hazy={hazy.size}, gt={clean.size}, "
                f"hazy_path={hazy_path}, gt_path={gt_path}"
            )

        if self.split == 'train' and self.crop_size is not None:
            w, h = hazy.size
            if h < self.crop_size or w < self.crop_size:
                raise RuntimeError(
                    f"Image too small for crop_size={self.crop_size}. "
                    f"Got image size {w}x{h} for key {key}."
                )

            i, j, hh, ww = transforms.RandomCrop.get_params(
                hazy, output_size=(self.crop_size, self.crop_size)
            )
            hazy = TF.crop(hazy, i, j, hh, ww)
            clean = TF.crop(clean, i, j, hh, ww)

            if self.augment_data:
                hazy, clean = augment(hazy, clean)

        hazy = self.transform(hazy)
        clean = self.transform(clean)

        return hazy, clean