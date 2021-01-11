class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/s'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'taco':
            return '/home/mathis/Documents/datasets/TACO/data'
        elif dataset =='surfrider':
            return '/home/mathis/Documents/datasets/surfrider_data/subset_of_images'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
