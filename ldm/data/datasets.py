import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import cv2


class DatasetsBase(Dataset):
    """CVC Dataset Base
    Notes:
        - `segmentation` is for the diffusion training stage (range binary -1 and 1)
        - `image` is for conditional signal to guided final seg-map (range -1 to 1)
    """
    def __init__(self, data_root, size=512, interpolation="nearest", mode=None,
                 num_classes=2, use_distance_transform=False, transform_enhance=False):
        self.data_root = data_root
        self.mode = mode
        self.use_distance_transform = use_distance_transform
        # assert mode in ["train", "val", "test"]
        self.data_paths = self._parse_data_list()
        self._length = len(self.data_paths)
        self.labels = dict(file_path_=[path for path in self.data_paths])
        self.size = size
        self.interpolation = dict(nearest=PIL.Image.NEAREST)[interpolation]   # for segmentation slice
        if transform_enhance:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomRotation(degrees=30),  # 随机旋转
                transforms.RandomAffine(degrees=0,
                                        translate=(0.1, 0.1),  # 平移
                                        scale=(0.9, 1.1)),  # 缩放
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])

        print(f"[Dataset]: 2 classes, in {self.mode} mode")

    def __getitem__(self, i):
        # read segmentation and images
        example = dict((k, self.labels[k][i]) for k in self.labels)
        # segmentation = Image.open(example["file_path_"].replace("Original", "GroundTruth")).convert("RGB")
        # image = Image.open(example["file_path_"]).convert("RGB")    # same name, different postfix
        # segmentation = Image.fromarray(cv2.cvtColor(cv2.imread(example["file_path_"].replace("images", "masks")),cv2.COLOR_BGR2RGB))
        # image = Image.fromarray(cv2.cvtColor(cv2.imread(example["file_path_"]), cv2.COLOR_BGR2RGB))
        segmentation = Image.fromarray(cv2.imread(example["file_path_"].replace("images", "masks"),
                                                  cv2.IMREAD_GRAYSCALE))
        image = Image.fromarray(cv2.imread(example["file_path_"], cv2.IMREAD_GRAYSCALE))

        if self.size is not None:
            segmentation = segmentation.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            image = image.resize((self.size, self.size), resample=PIL.Image.BILINEAR)

        if self.mode == "train":
            segmentation, image = self._utilize_transformation(segmentation, image, self.transform)

        segmentation = (np.array(segmentation)[..., None] > 128).astype(np.uint8)

        if self.use_distance_transform:
            # ------------------------------------------------------------------------------------------------
            # If use_distance_transform is True, build a 3‑channel mask:
            #   channel 0 = original binary mask (-1, +1)
            #   channel 1 = normalized foreground distance (-1, +1)
            #   channel 2 = normalized background distance (-1, +1)
            # ------------------------------------------------------------------------------------------------
            # compute Euclidean distance for foreground pixels to nearest background
            fg_dist = cv2.distanceTransform(segmentation, distanceType=cv2.DIST_L2, maskSize=5)
            # compute Euclidean distance for background pixels to nearest foreground
            bg_dist = cv2.distanceTransform(1 - segmentation, distanceType=cv2.DIST_L2, maskSize=5)

            # normalize distances to [0,1]
            fg_dist = fg_dist / (fg_dist.max() + 1e-8)
            bg_dist = bg_dist / (bg_dist.max() + 1e-8)

            # map all channels into [-1,1]
            bin_chan = (segmentation.astype(np.float32) * 2) - 1
            fg_chan = (fg_dist.astype(np.float32) * 2) - 1
            bg_chan = (bg_dist.astype(np.float32) * 2) - 1

            # stack into H×W×3
            seg_out = np.stack([bin_chan, fg_chan, bg_chan], axis=-1)
        else:
            # ------------------------------------------------------------------------------------------------
            # Otherwise, just keep the single‐channel binary mask in [-1, +1]
            # ------------------------------------------------------------------------------------------------
            seg_bin = segmentation.astype(np.float32)
            seg_out = (seg_bin * 2) - 1

        # store segmentation
        example["segmentation"] = seg_out

        image = np.array(image).astype(np.float32) / 255.
        example["gray_seg"] = (seg_bin * image) * 2. - 1.
        image = (image * 2.) - 1.  # range from -1 to 1, np.float32
        example["image"] = image
        example["class_id"] = np.array([-1])  # doesn't matter for binary seg

        assert np.max(segmentation) <= 1. and np.min(segmentation) >= -1.
        assert np.max(image) <= 1. and np.min(image) >= -1.
        return example

    def __len__(self):
        return self._length

    def _parse_data_list(self):
        # all_imgs = glob.glob(os.path.join(self.data_root, "*.png"))
        # train_imgs, val_imgs, test_imgs = all_imgs[:800], all_imgs[800:], all_imgs[800:]

        split_root = self.data_root.replace("images", "split")

        def read_image_list(txt_file, ext=".png"):  # 如果是jpg就改成".jpg"
            with open(txt_file, 'r') as f:
                img_names = [line.strip() + ext for line in f.readlines()]
            return [os.path.join(self.data_root, name) for name in img_names]

        train_file = os.path.join(split_root, 'train.txt')
        test_file = os.path.join(split_root, 'test.txt')

        train_imgs = read_image_list(train_file)
        test_imgs = read_image_list(test_file)

        if self.mode == "train":
            return train_imgs
        elif self.mode == "val":
            return test_imgs  # val 和 test 都用 test.txt 的内容
        elif self.mode == "test":
            return test_imgs
        else:
            raise NotImplementedError(f"Only support dataset split: train, val, test!")

    @staticmethod
    def _utilize_transformation(segmentation, image, func):
        state = torch.get_rng_state()
        segmentation = func(segmentation)
        torch.set_rng_state(state)
        image = func(image)
        return segmentation, image


class DataTrain(DatasetsBase):
    def __init__(self, name, **kwargs):
        super().__init__(data_root=f"data/{name}/images", mode="train", **kwargs)


class DataValidation(DatasetsBase):
    def __init__(self, name, **kwargs):
        super().__init__(data_root=f"data/{name}/images", mode="val", **kwargs)


class DataTest(DatasetsBase):
    def __init__(self, name, **kwargs):
        super().__init__(data_root=f"data/{name}/images", mode="test", **kwargs)
