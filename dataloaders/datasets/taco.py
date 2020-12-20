import numpy as np
import torch
from torch.utils.data import Dataset
from mypath import Path
from tqdm import trange
import os
from pycocotools.coco import COCO
from pycocotools import mask
from torchvision import transforms
from dataloaders import custom_transforms as tr
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class TACOSegmentation(Dataset):
    NUM_CLASSES = 60

    def __init__(self, args=None, base_dir=Path.db_root_dir('taco'), split="train"):
        super().__init__()

        round = 0 
        ann_file = os.path.join(base_dir , 'annotations')
        ann_file += "_" + str(round) + "_" + split + ".json"
        ids_file = os.path.join(base_dir, 'annotations_{}_ids_{}.pth'.format(split, round))
        self.img_dir = base_dir
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)
        self.args = args

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}


        # return transforms.Compose([tr.FixScaleCrop(crop_size=self.args.crop_size),tr.ToTensor()])(sample)
        
        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        

    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        img_id = self.ids[index]

        _img, img_metadata = self._load_image(img_id)
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        _target = Image.fromarray(self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width']))

        return _img, _target

    def _load_image(self, img_id):
        # Load image. TODO: do this with opencv to avoid need to correct orientation
        img_metadata = self.coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        image = Image.open(os.path.join(self.img_dir, path))
        img_shape = np.shape(image)

        # load metadata
        exif = image._getexif()
        if exif:
            exif = dict(exif.items())
            # Rotate portrait images if necessary (274 is the orientation tag code)
            if 274 in exif:
                if exif[274] == 3:
                    image = image.rotate(180, expand=True)
                if exif[274] == 6:
                    image = image.rotate(270, expand=True)
                if exif[274] == 8:
                    image = image.rotate(90, expand=True)

        # If has an alpha channel, remove it for consistency
        if img_shape[-1] == 4:
            image = image[..., :3]

        return image.convert('RGB'), img_metadata #np.array(image)

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while. " + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            # if cat in self.CAT_LIST:
            #     c = self.CAT_LIST.index(cat)
            # else:
            #     continue
            c = cat
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.498, 0.470, 0.415), std=(0.234, 0.220, 0.220)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.498, 0.470, 0.415), std=(0.234, 0.220, 0.220)),
            tr.ToTensor()])

        return composed_transforms(sample)


    def __len__(self):
        return len(self.ids)


# if __name__ == "__main__":
#     from dataloaders import custom_transforms as tr
#     from dataloaders.utils import decode_segmap
#     from torch.utils.data import DataLoader
#     from torchvision import transforms
#     import matplotlib.pyplot as plt
#     import argparse
#     import torch
#     from tqdm import tqdm



#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#     args.base_size = 513
#     args.crop_size = 513

#     coco_val = TACOSegmentation(args,split='train')

#     dataloader = DataLoader(coco_val, batch_size=64, shuffle=True, num_workers=0)

#     def get_mean_std(loader):
#         # var[X] = E[X**2] - E[X]**2
#         channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

#         for data in tqdm(loader):
#             data = data['image']
#             channels_sum += torch.mean(data, dim=[0, 2, 3])
#             channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
#             num_batches += 1

#         mean = channels_sum / num_batches
#         std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

#         return mean/255., std/255.


#     mean, std = get_mean_std(dataloader)
#     print(mean)
#     print(std)




if __name__ == "__main__":
    from dataloaders import custom_transforms as tr
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    coco_val = TACOSegmentation(args,split='val')

    dataloader = DataLoader(coco_val, batch_size=4, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='taco')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


