import numpy as np
from server.py_rmpe_data_iterator import RawDataIterator
import keras

import six
if six.PY3:
  buffer_ = memoryview
else:
  buffer_ = buffer  # noqa

class DataIteratorBase:

    def __init__(self, batch_size = 10, num_out=4, segmentation=True, visualize=False):

        self.batch_size = batch_size
        self.heat_num = 18
        self.num_out = num_out
        self.keypoints = [None]*self.batch_size #this is not passed to NN, will be accessed by accuracy calculation
        self.segmentation = segmentation
        self.visualize = visualize

    def gen_raw(self): # this function used for test purposes in py_rmpe_server

        while True:
            yield tuple(self._recv_arrays())

    def preprocess_input(self, image):
        '''
        :param image: image in RGB format
        :return: image normalized by imagenet means
        '''
        image = image.astype(keras.backend.floatx())
        image[:, :, 0] -= 103.939
        image[:, :, 1] -= 116.779
        image[:, :, 2] -= 123.68
        return image

    def gen(self):
        batches_x  = [None]*self.batch_size
        batches_x2 = [None]*self.batch_size
        batches_y2 = [None]*self.batch_size

        sample_idx = 0

        for foo in self.gen_raw():

            if len(foo)==5:
                data_img, mask_img, mask_all, label, kpts = foo
            else:
                data_img, mask_img, mask_all, label = foo
                kpts = None

            # order = np.array([0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10])
            # label[order]

            dta_img = np.transpose(data_img, (1, 2, 0))[:,:,::-1]
            if not self.visualize:
                dta_img = self.preprocess_input(dta_img)
            label = np.transpose(label, (1, 2, 0))
            mask_miss = np.repeat(mask_img[:, :, np.newaxis], self.heat_num, axis=-1)

            if self.segmentation:
                label = np.dstack((label, np.expand_dims(mask_all, axis=-1)))
                mask_miss = np.dstack((mask_miss, np.ones((mask_miss.shape[0], mask_miss.shape[1], 1))))

            # print('mask_all max: %s' % mask_all.max())
            # print('mask_img max: %s' % mask_img.max())
            # print('mask_all min: %s' % mask_all.min())
            # print('mask_img min: %s' % mask_img.min())

            batches_x[sample_idx]=dta_img[np.newaxis, ...]
            batches_x2[sample_idx] = mask_miss[np.newaxis, ...]
            batches_y2[sample_idx] = label[np.newaxis, ...]

            self.keypoints[sample_idx] = kpts

            sample_idx += 1

            if sample_idx == self.batch_size:
                sample_idx = 0

                batch_x = np.concatenate(batches_x)
                batch_x2 = np.concatenate(batches_x2)

                batch_y2 = np.concatenate(batches_y2)


                yield [batch_x] + [batch_x2], [batch_y2]*self.num_out

                self.keypoints = [None] * self.batch_size

    def keypoints(self):
        return self.keypoints

class DataIterator(DataIteratorBase):

    def __init__(self, file, config, shuffle=True, augment=True, batch_size=10, num_out=4, limit=None,
                 segmentation=True, visualize=False):

        super(DataIterator, self).__init__(batch_size, num_out, segmentation, visualize)

        self.limit = limit
        self.records = 0

        self.raw_data_iterator = RawDataIterator(file, config, shuffle=shuffle, augment=augment)
        self.generator = self.raw_data_iterator.gen()


    def _recv_arrays(self):

        while True:

            if self.limit is not None and self.records > self.limit:
                raise StopIteration

            tpl = next(self.generator, None)
            if tpl is not None:
                self.records += 1
                return tpl

            if self.limit is None or self.records < self.limit:
                print("Starting next generator loop cycle")
                self.generator = self.raw_data_iterator.gen()
            else:
                raise StopIteration
