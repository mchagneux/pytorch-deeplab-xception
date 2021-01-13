from torchvision.datasets import CocoDetection
from torchvision import transforms
from PIL import Image
import os 
from pycocotools import mask
import numpy as np 
import torchvision.transforms.functional as TF
import random

import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2



class SurfriderDetection(CocoDetection):

    def __init__(self, split='val', root='/home/mathis/Documents/datasets/surfrider_data/subset_of_images/', no_transform = False):
        
        self.split = split
        annFile = root + 'annotations_' + self.split + '.json'

        if no_transform: transform = None
        else:
            transform = {
                'train': A.Compose(
                    [
                        A.RandomCrop(512, 512),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ToTensorV2()
                    ]
                ),
                'val': A.Compose(
                    [
                        A.Resize(512, 512),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ToTensorV2()
                    ]
                )
            }

        super(SurfriderDetection, self).__init__(root, annFile, transforms=transform[split])
        

        self.mask =  mask
    
        self.CAT_LIST = self.coco.getCatIds()

        self.NUM_CLASSES = len(self.CAT_LIST)

    def __getitem__(self, index):

        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_target = coco.loadAnns(ann_ids)

        coco_img  = coco.loadImgs(img_id)[0]
        path = coco_img['file_name']

        img = np.asarray(Image.open(os.path.join(self.root, path)).convert('RGB'))
        target = self._gen_seg_mask(
            coco_target, coco_img['height'], coco_img['width'])

        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=target)
            img=transformed['image']
            target=transformed['mask']
        sample = {'image':img,'label':target}
        return sample

    def _gen_seg_mask(self, target, h, w):
        coco_mask = self.mask
        mask = np.zeros((h, w), dtype=np.uint8)
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
                
        return mask