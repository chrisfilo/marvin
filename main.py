import os
import tensorflow as tf
import models.basic_cnn as model
from data_io import InputFnFactory
import datetime

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    log_dir = "logs"
    current_run_subdir = os.path.join(
        "run_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    model_dir = os.path.join(log_dir, model.name, "72x72")#current_run_subdir)

    run_config = tf.estimator.RunConfig(model_dir=model_dir)

    params = tf.contrib.training.HParams(
        target_shape=(72, 72, 72),
        model_dir=model_dir
    )

    ds = InputFnFactory(target_shape=params.target_shape,
                        n_epochs=100,
                        train_src_folder="D:/data/PAC_Data/PAC_data/train",
                        train_cache_prefix="D:/drive/workspace/marvin/cache_train",
                        eval_src_folder="D:/data/PAC_Data/PAC_data/eval",
                        eval_cache_prefix="D:/drive/workspace/marvin/cache_eval",
                        batch_size=25
                        )

    train_spec = tf.estimator.TrainSpec(input_fn=ds.train_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=ds.eval_input_fn,
                                      steps=None,
                                      start_delay_secs=0,
                                      throttle_secs=1200)

    estimator = tf.estimator.Estimator(model_fn=model.model_fn,
                                       params=params,
                                       config=run_config)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
