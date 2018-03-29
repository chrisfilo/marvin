import os
import tensorflow as tf
import models.basic_cnn as model
from data_io import InputFnFactory
import datetime
import models.utils as utils

from tensorflow.contrib.learn.python.learn import learn_runner

tf.logging.set_verbosity(tf.logging.INFO)


def experiment_fn(run_config, params):
    """Train MNIST for a number of steps."""

    ds = InputFnFactory(target_shape=params.target_shape,
                        n_epochs=params.n_epochs,
                        train_src_folder=params.train_src_folder,
                        train_cache_prefix=params.train_cache_prefix,
                        eval_src_folder=params.eval_src_folder,
                        eval_cache_prefix=params.eval_cache_prefix,
                        batch_size=params.batch_size
                        )

    estimator = utils.get_estimator(model.model_fn, run_config, params)

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,  # Estimator
        train_input_fn=ds.train_input_fn,  # First-class function
        eval_input_fn=ds.eval_input_fn,  # First-class function
        train_steps=params.train_steps,  # Minibatch steps
        eval_steps=None  # Use evaluation feeder until its empty
    )

    return experiment


if __name__ == '__main__':
    log_dir = "logs"
    current_run_subdir = os.path.join(
        "run_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    model_dir = os.path.join(log_dir, model.name, current_run_subdir)

    session_config = tf.ConfigProto()
    session_config.inter_op_parallelism_threads = 8
    run_config = tf.contrib.learn.RunConfig(
        save_checkpoints_secs=600,
        model_dir=model_dir,
        save_summary_steps=100,
        log_step_count_steps=100,
        session_config=session_config)

    params = tf.contrib.training.HParams(
        learning_rate=0.0002,
        train_steps=2000000,
        target_shape=(32, 32, 32),
        train_src_folder="D:/data/PAC_Data/PAC_data/train",
        train_cache_prefix="D:/drive/workspace/PAC/cache_train",
        eval_src_folder="D:/data/PAC_Data/PAC_data/eval",
        eval_cache_prefix="D:/drive/workspace/PAC/cache_eval",
        batch_size=40,
        n_epochs=100,
        model_dir=model_dir,
    )

    learn_runner.run(
        experiment_fn=experiment_fn,  # First-class function
        run_config=run_config,  # RunConfig
        schedule="train_and_evaluate",  # What to run
        hparams=params  # HParams
    )
