import numpy as np
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
from albumentations.augmentations.transforms import Cutout

class MyImageDataGenerator(Sequence):
    def __init__(self, 
                 images,
                 y,
                 batch_size, 
                 image_generator_kwargs,
                 random_erasing_kwargs=None,
                 mixup_alpha=None,
                 cutout_kwargs=None,
                 ):
        self.IMAGES = images
        self.Y = y

        self.BATCH_SIZE = batch_size

        self.IMAGE_GENERATOR_ARGS = image_generator_kwargs

        # {'erasing_prob':, 'area_rate_low':, 'area_rate_high':, 'aspect_rate_low':, 'aspect_rate_high':}
        self.RANDOM_ERASING_KWARGS = random_erasing_kwargs
        self.MIXUP_ALPHA = mixup_alpha
        # {'num_holes':1, 'max_h_size':16, 'max_w_size':16, 'always_apply':False, 'p':0.3}
        self.CUTOUT_KWARGS = cutout_kwargs

        self.__initialize()

        return

    def __getitem__(self, idx):
        return self.__get_images(idx)

    def __len__(self):
        return self.STEP_NUM

    def on_epoch_end(self):
        self.idxes = self.__get_indexes(self.SAMPLE_NUM, do_shuffle=True)
        return

    def __get_images(self, idx):
        batch_idxes = self.idxes[idx * self.BATCH_SIZE : (idx + 1) * self.BATCH_SIZE]

        # y
        ys = self.Y[batch_idxes]
        
        # get images
        if self.IMAGE_GENERATOR_ARGS is not None:
            images = next(self.image_datagen.flow(self.IMAGES[batch_idxes], batch_size=self.BATCH_SIZE, shuffle=False))
        else:
            images = self.IMAGES[batch_idxes]

        # mixup
        if self.MIXUP_ALPHA is not None:
            images, ys = self.__mixup(images, ys)

        # random erasing
        if self.RANDOM_ERASING_KWARGS is not None:
            images = self.__random_erasing(images, **self.RANDOM_ERASING_KWARGS)

        # cutout
        if self.CUTOUT_KWARGS is not None:
            images = self.__cutout(images, **self.CUTOUT_KWARGS)

        return images, ys

    def __initialize(self):
        self.SAMPLE_NUM = len(self.IMAGES)
        self.STEP_NUM = int(np.ceil(self.SAMPLE_NUM / self.BATCH_SIZE))

        self.idxes = self.__get_indexes(self.SAMPLE_NUM, do_shuffle=True)

        if self.IMAGE_GENERATOR_ARGS is not None:
            self.image_datagen = ImageDataGenerator(**self.IMAGE_GENERATOR_ARGS)

        return

    def __get_indexes(self, sample_num, do_shuffle=True):
        '''
        return shuffled indexes.
        '''
        indexes = np.arange(sample_num)
        if do_shuffle:
            indexes = np.random.permutation(indexes)
        return indexes

    def __random_erasing(self, images, erasing_prob, area_rate_low, area_rate_high, aspect_rate_low, aspect_rate_high):
        # https://qiita.com/takurooo/items/a3cba475a3db2c7272fe

        def _rand_erase(_img):
            target_image = _img.copy()

            if np.random.rand() > erasing_prob:
                # RandomErasingを実行しない
                return target_image 

            H, W, C = target_image.shape
            S = H * W

            while True:
                Se = np.random.uniform(area_rate_low, area_rate_high) * S # 画像に重畳する矩形の面積
                re = np.random.uniform(aspect_rate_low, aspect_rate_high) # 画像に重畳する矩形のアスペクト比

                He = int(np.sqrt(Se * re)) # 画像に重畳する矩形のHeight
                We = int(np.sqrt(Se / re)) # 画像に重畳する矩形のWidth

                xe = np.random.randint(0, W) # 画像に重畳する矩形のx座標
                ye = np.random.randint(0, H) # 画像に重畳する矩形のy座標

                if xe + We <= W and ye + He <= H:
                    # 画像に重畳する矩形が画像からはみ出していなければbreak
                    break

            #mask = np.random.randint(np.min(target_image), np.max(target_image), (He, We, C)) # 矩形がを生成 矩形内の値はランダム値
            mask = np.random.rand(He, We, C) # 矩形がを生成 矩形内の値はランダム値
            mask = mask * np.max(target_image) + (1 - mask) * np.min(target_image)
            
            target_image[ye:ye + He, xe:xe + We, :] = mask # 画像に矩形を重畳

            return target_image

        erased_imgs = []
        for img in images:
            erased_imgs.append(_rand_erase(img))
        erased_imgs = np.array(erased_imgs)

        return erased_imgs

    def __mixup(self, xs, ys):
        mixup_rate = np.random.beta(self.MIXUP_ALPHA, self.MIXUP_ALPHA, len(xs))
        mixup_idx = self.__get_indexes(len(xs), do_shuffle=True)

        def __mix(_x):
            _rate_shape = list(_x.shape)        
            for i in range(len(_rate_shape)):
                if i > 0:
                    _rate_shape[i] = 1
            _rate_shape = tuple(_rate_shape)
            _re_rate = mixup_rate.reshape(_rate_shape)
            return _re_rate * _x + (1.0 - _re_rate) * _x[mixup_idx]

        mix_xs = __mix(xs)
        mix_ys = __mix(ys)
        return mix_xs, mix_ys

    def __cutout(self, images, num_holes=1, max_h_size=16, max_w_size=16, always_apply=False, p=0.3):
        aug = Cutout(num_holes, max_h_size, max_w_size, always_apply, p)
        #auged_images = aug(image=images)['image']
        auged_images = []
        for img in images:
            auged_images.append(aug(image=img)['image'])
        auged_images = np.array(auged_images)

        return auged_images