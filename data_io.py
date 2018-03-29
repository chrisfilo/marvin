from glob import glob
import tensorflow as tf
import os
import nibabel as nb
import numpy as np
from nilearn import image
import pandas as pd


def _get_resize_arg(target_shape):
    mni_shape_mm = np.array([148.0, 184.0, 156.0])
    target_resolution_mm = np.ceil(
        mni_shape_mm / np.array(target_shape)).astype(
        np.int32)
    target_affine = np.array([[4., 0., 0., -75.],
                              [0., 4., 0., -105.],
                              [0., 0., 4., -70.],
                              [0., 0., 0., 1.]])
    target_affine[0, 0] = target_resolution_mm[0]
    target_affine[1, 1] = target_resolution_mm[1]
    target_affine[2, 2] = target_resolution_mm[2]
    return target_affine, list(target_shape)


def _get_data(nthreads, batch_size, src_folder, n_epochs, cache_prefix,
              shuffle, target_shape):
    in_images = glob(os.path.join(src_folder, "*.nii*"))
    assert len(in_images) != 0

    paths = tf.constant(in_images)
    paths_ds = tf.data.Dataset.from_tensor_slices((paths,))

    target_affine, target_shape = _get_resize_arg(target_shape)

    target_affine_tf = tf.constant(target_affine)
    target_shape_tf = tf.constant(target_shape)

    def _read_and_resample(path, target_affine, target_shape):
        path_str = path.decode('utf-8')
        nii = nb.load(path_str)
        data = nii.get_data()
        data[np.isnan(data)] = 0
        nii = nb.Nifti1Image(data, nii.affine)
        nii = image.resample_img(nii,
                                 target_affine=target_affine,
                                 target_shape=target_shape)
        data = nii.get_data().astype(np.float32)

        return data

    data_ds = paths_ds.map(
        lambda path: tuple(tf.py_func(_read_and_resample,
                                      [path, target_affine_tf,
                                       target_shape_tf],
                                      [tf.float32])),
        num_parallel_calls=nthreads)

    def _reshape(data):
        return tf.reshape(data, target_shape)

    data_ds = data_ds.map(_reshape, num_parallel_calls=nthreads)

    def _get_label(path):
        path_str = path.decode('utf-8')
        df = pd.read_csv(
            "D:/data/PAC_Data/PAC_Data/PAC2018_Covariates_Upload.csv")
        PAC_ID = path_str.split(os.sep)[-1].split('.')[0]
        label = int(df[df.PAC_ID == PAC_ID]['Label'])
        return label

    labels_ds = paths_ds.map(
        lambda path: tuple(tf.py_func(_get_label,
                                      [path],
                                      [tf.int32])),
        num_parallel_calls=nthreads)

    dataset = tf.data.Dataset.zip((data_ds, labels_ds))

    dataset = dataset.cache(cache_prefix)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(n_epochs)
    return dataset


class InputFnFactory:

    def __init__(self, target_shape, batch_size,
                 n_epochs, train_src_folder,
                 train_cache_prefix, eval_src_folder, eval_cache_prefix,
                 nthreads=None):
        self.target_shape = target_shape
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.train_src_folder = train_src_folder
        self.train_cache_prefix = train_cache_prefix
        self.eval_src_folder = eval_src_folder
        self.eval_cache_prefix = eval_cache_prefix
        if nthreads is None:
            import multiprocessing
            self.nthreads = multiprocessing.cpu_count()
        else:
            self.nthreads = nthreads

    def _get_iterator(self, cache_prefix, src_folder, shuffle, n_epochs):
        training_dataset = _get_data(nthreads=self.nthreads,
                                     batch_size=self.batch_size,
                                     src_folder=src_folder,
                                     n_epochs=n_epochs,
                                     cache_prefix=cache_prefix,
                                     shuffle=shuffle,
                                     target_shape=self.target_shape)
        # You can use feedable iterators with a variety of different kinds of iterator
        # (such as one-shot and initializable iterators).
        training_iterator = training_dataset.make_one_shot_iterator()
        return training_iterator.get_next()

    def train_input_fn(self):
        return self._get_iterator(cache_prefix=self.train_cache_prefix,
                                  src_folder=self.train_src_folder,
                                  shuffle=True,
                                  n_epochs=self.n_epochs)

    def eval_input_fn(self):
        return self._get_iterator(cache_prefix=self.eval_cache_prefix,
                                  src_folder=self.eval_src_folder,
                                  shuffle=False,
                                  n_epochs=1)
