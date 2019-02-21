import tensorflow as tf
conv2d=tf.layers.conv2d
bilinear = tf.image.ResizeMethod.BILINEAR
activation = tf.nn.leaky_relu


def log_images(stack_name, image_stack):
    image_unstack = tf.unstack(image_stack, axis=3)
    for i, image in enumerate(image_unstack):
        tf.summary.image(stack_name+'_'+str(i), tf.expand_dims(image, axis=3))
def conv_resize_dropout_layer(x,filters,kernel_size,padding='valid',resize=None,l2_scale=0.0,rate=0.0,name=None,mode=tf.estimator.ModeKeys.PREDICT):
    regularizer = tf.contrib.layers.l2_regularizer(scale=l2_scale)
    conv=conv2d(
        x, filters=filters,kernel_size=kernel_size, padding=padding, activation=activation,name=name, kernel_regularizer=regularizer)
    if resize:
        resized=tf.image.resize_images(conv,resize,method=bilinear)
    else:
        resized=conv
    if name:
        log_images(name,resized)
    return tf.layers.dropout(resized, rate=rate, training=mode==tf.estimator.ModeKeys.TRAIN)