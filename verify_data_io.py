import tensorflow as tf

from data_io import _get_data
import numpy as np

with tf.Session() as sess:
    validation_dataset = _get_data(nthreads=8,
                                   batch_size=10,
                                   src_folder="D:/data/PAC_Data/PAC_Data/train",
                                   n_epochs=3,
                                   cache_prefix=None,
                                   shuffle=True,
                                   target_shape=(32, 32, 32))

    validation_iterator = validation_dataset.make_one_shot_iterator()
    filenames = []
    count = 0

    import time

    start = time.time()
    print("hello")

    while True:
        try:
            features, labels = sess.run(validation_iterator.get_next())
            print(labels, count)
            print(np.amax(features, axis=(1,2,3)))
            count += 1
        except tf.errors.OutOfRangeError:
            break
    end = time.time()
    print(end - start)
    print(count)