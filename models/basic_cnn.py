import tensorflow as tf

name = "basic_cnn"

def model_fn(features, labels, mode, params):

    features = tf.expand_dims(features, -1)

    layer_conv1 = tf.layers.conv3d(inputs=features,
                                   filters=32,
                                   kernel_size=3,
                                   padding='same',
                                   bias_initializer=tf.random_normal_initializer(0, 0.02),
                                   use_bias=True,
                                   kernel_initializer=tf.random_normal_initializer(0, 0.02))

    layer_maxpool1 = tf.layers.max_pooling3d(inputs=layer_conv1,
                                             pool_size=2,
                                             strides=2,
                                             padding='same')
    layer_relu1 = tf.nn.relu(layer_maxpool1)

    layer_flat = tf.layers.flatten(layer_relu1)
    layer_fc1 = tf.contrib.layers.fully_connected(layer_flat, num_outputs=2)

    y_pred = tf.nn.softmax(layer_fc1, name="y_pred")
    predictions = tf.argmax(y_pred, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )
    else:
        labels = tf.Print(labels, [labels], message="This is labels: ")
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=layer_fc1,
            labels=tf.one_hot(indices=labels, depth=2))
        loss = tf.reduce_mean(cross_entropy)

        # Add a scalar summary for the snapshot loss.
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(loss,
                                          global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            # eval_metric_ops={
            #     "accuracy": tf.metrics.accuracy(
            #         labels=labels, predictions=predictions)}
        )