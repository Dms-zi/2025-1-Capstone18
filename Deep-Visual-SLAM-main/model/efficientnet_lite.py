import tensorflow as tf
from tensorflow import keras as tf_keras
from copy import deepcopy

DEFAULT_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 2
    if tf_keras.backend.image_data_format() == "channels_last":
        img_dim = 1
    input_size = tf_keras.backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

def block(inputs, activation_fn=tf.nn.swish, drop_rate=0., name='',
          filters_in=32, filters_out=16, kernel_size=3, strides=1,
          expand_ratio=1, se_ratio=0., id_skip=True):
    """A mobile inverted residual block.
    # Arguments
        inputs: input tensor.
        activation_fn: activation function.
        drop_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.
        se_ratio: float between 0 and 1, fraction to squeeze the input filters.
        id_skip: boolean.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if tf_keras.backend.image_data_format() == 'channels_last' else 1

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = tf_keras.layers.Conv2D(filters, 1,
                          padding='same',
                          use_bias=False,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name=name + 'expand_conv')(inputs)
        x = tf_keras.layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
        x = tf_keras.layers.Activation(activation_fn, name=name + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        x = tf_keras.layers.ZeroPadding2D(padding=correct_pad(x, kernel_size),
                                 name=name + 'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    x = tf_keras.layers.DepthwiseConv2D(kernel_size,
                               strides=strides,
                               padding=conv_pad,
                               use_bias=False,
                               depthwise_initializer=CONV_KERNEL_INITIALIZER,
                               name=name + 'dwconv')(x)
    x = tf_keras.layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    x = tf_keras.layers.Activation(activation_fn, name=name + 'activation')(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = tf_keras.layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
        if bn_axis == 1:
            se = tf_keras.layers.Reshape((filters, 1, 1), name=name + 'se_reshape')(se)
        else:
            se = tf_keras.layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
        se = tf_keras.layers.Conv2D(filters_se, 1,
                                    padding='same',
                                    activation=activation_fn,
                                    kernel_initializer=CONV_KERNEL_INITIALIZER,
                                    name=name + 'se_reduce')(se)
        se = tf_keras.layers.Conv2D(filters, 1,
                                    padding='same',
                                    activation='sigmoid',
                                    kernel_initializer=CONV_KERNEL_INITIALIZER,
                                    name=name + 'se_expand')(se)
        if tf_keras.backend.backend() == 'theano':
            # For the Theano backend, we have to explicitly make
            # the excitation weights broadcastable.
            se = tf_keras.layers.Lambda(
                lambda x: tf_keras.backend.pattern_broadcast(x, [True, True, True, False]),
                output_shape=lambda input_shape: input_shape,
                name=name + 'se_broadcast')(se)
        x = tf_keras.layers.multiply([x, se], name=name + 'se_excite')

    # Output phase
    x = tf_keras.layers.Conv2D(filters_out, 1,
                               padding='same',
                               use_bias=False,
                               kernel_initializer=CONV_KERNEL_INITIALIZER,
                               name=name + 'project_conv')(x)
    x = tf_keras.layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
    if (id_skip is True and strides == 1 and filters_in == filters_out):
        if drop_rate > 0:
            x = tf_keras.layers.Dropout(drop_rate,
                               noise_shape=(None, 1, 1, 1),
                               name=name + 'drop')(x)
        x = tf_keras.layers.add([x, inputs], name=name + 'add')

    return x

def efficientnet_lite(width_coefficient,
                      depth_coefficient,
                      default_size,
                      dropout_rate=0.2,
                      drop_connect_rate=0.2,
                      depth_divisor=8,
                      activation_fn=tf.nn.relu6,
                      blocks_args=[{k:v if "se_ratio" not in k else 0. for k, v in arg.items()} for arg in DEFAULT_BLOCKS_ARGS],
                      block = block,
                      conv_kernel_initializer = CONV_KERNEL_INITIALIZER,
                      dense_kernel_initializer = DENSE_KERNEL_INITIALIZER,
                      include_top=True,
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      classes=1000,
                      weights = None,
                      **kwargs):
    #https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py
    def correct_pad(inputs, kernel_size):
        """Returns a tuple for zero-padding for 2D convolution with downsampling.
        # Arguments
            input_size: An integer or tuple/list of 2 integers.
            kernel_size: An integer or tuple/list of 2 integers.
        # Returns
            A tuple.
        """
        img_dim = 2
        if tf_keras.backend.image_data_format() == "channels_last":
            img_dim = 1
        input_size = tf_keras.backend.int_shape(inputs)[img_dim:(img_dim + 2)]

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if input_size[0] is None:
            adjust = (1, 1)
        else:
            adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

        correct = (kernel_size[0] // 2, kernel_size[1] // 2)

        return ((correct[0] - adjust[0], correct[0]),
                (correct[1] - adjust[1], correct[1]))
    
    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(tf.math.ceil(tf.cast(depth_coefficient * repeats, tf.float32)))
    
    if input_tensor is None:
        img_input = tf_keras.layers.Input(shape = input_shape)
    else:
        if not tf_keras.backend.is_keras_tensor(input_tensor):
            img_input = tf_keras.layers.Input(tensor = input_tensor, shape = input_shape)
        else:
            img_input = input_tensor

    # Build stem
    x = img_input
    x = tf_keras.layers.ZeroPadding2D(padding=correct_pad(x, 3),
                                      name='stem_conv_pad')(x)
    #filters = round_filters(32) #efficientnet lite > fixed feature
    x = tf_keras.layers.Conv2D(32, 3,
                               strides=2,
                               padding='valid',
                               use_bias=False,
                               kernel_initializer=conv_kernel_initializer,
                               name='stem_conv')(x)
    x = tf_keras.layers.BatchNormalization(axis=-1, name='stem_bn')(x)
    x = tf_keras.layers.Activation(activation_fn, name='stem_activation')(x)

    # Build blocks
    blocks_args = deepcopy(blocks_args)

    b = 0
    blocks = float(sum(args['repeats'] for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        # Update block input and output filters based on depth multiplier.
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])
        
        #repeats = round_repeats(args.pop('repeats')) #efficientnet lite > repeats condition add
        repeats = args.pop("repeats") if (i == 0 or i == (len(blocks_args) - 1)) else round_repeats(args.pop("repeats"))

        for j in range(repeats):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            x = block(x, activation_fn, drop_connect_rate * b / blocks,
                      name='block{}{}_'.format(i + 1, chr(j + 97)), **args)
            b += 1
    
    # Build top
    #filters = round_filters(1280) #efficientnet lite > fixed feature
    x = tf_keras.layers.Conv2D(1280, 1,
                               padding='same',
                               use_bias=False,
                               kernel_initializer=conv_kernel_initializer,
                               name='top_conv')(x)
    x = tf_keras.layers.BatchNormalization(axis=-1, name='top_bn')(x)
    x = tf_keras.layers.Activation(activation_fn, name='top_activation')(x)
    if include_top:
        x = tf_keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate > 0:
            x = tf_keras.layers.Dropout(dropout_rate, name='top_dropout')(x)
        x = tf_keras.layers.Dense(classes,
                         activation='softmax',
                         kernel_initializer=dense_kernel_initializer,
                         name='probs')(x)
    else:
        if pooling == 'avg':
            x = tf_keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = tf_keras.layers.GlobalMaxPooling2D(name='max_pool')(x)
    
    model = tf_keras.Model(img_input, x)
    if weights is not None:
        model.load_weights(weights)
    return model

effnet_lite_urls = {
    "effnet_lite_b0":"https://tfhub.dev/tensorflow/efficientnet/lite0/classification/2",
    "effnet_lite_b1":"https://tfhub.dev/tensorflow/efficientnet/lite1/classification/2",
    "effnet_lite_b2":"https://tfhub.dev/tensorflow/efficientnet/lite2/classification/2",
    "effnet_lite_b3":"https://tfhub.dev/tensorflow/efficientnet/lite3/classification/2",
    "effnet_lite_b4":"https://tfhub.dev/tensorflow/efficientnet/lite4/classification/2"
}

def load_weight(model, url):
    """
    https://tfhub.dev/tensorflow/efficientnet/lite0/classification/2
    https://tfhub.dev/tensorflow/efficientnet/lite1/classification/2
    https://tfhub.dev/tensorflow/efficientnet/lite2/classification/2
    https://tfhub.dev/tensorflow/efficientnet/lite3/classification/2
    https://tfhub.dev/tensorflow/efficientnet/lite4/classification/2
    """
    try:
        import tensorflow_hub as hub
        with tf.device("/cpu:0"):
            mod = hub.load(url)
    except:
        print("If you want to use 'EfficientNet-Lite Weight', please install 'tensorflow_hub'")
        return model
    for w, new_w in zip(model.weights, mod.variables):
        try:
            tf_keras.backend.set_value(w, new_w.numpy())
        except:
            print('Could not transfer weight for', w.name)
    return model

def efficientnet_lite_b0(input_tensor = None, input_shape = None, classes = 1000, include_top = True, weights = "imagenet"):
    hub_weight = False
    if weights == "imagenet":
        hub_weight = True
        weights = None
    model = efficientnet_lite(1.0, 1.0, 224, 0.2, input_tensor = input_tensor, input_shape = input_shape, classes = classes, include_top = include_top, weights = weights)
    if hub_weight:
        model = load_weight(model, effnet_lite_urls["effnet_lite_b0"])
    return model

def efficientnet_lite_b1(input_tensor = None, input_shape = None, classes = 1000, include_top = True, weights = "imagenet"):
    hub_weight = False
    if weights == "imagenet":
        hub_weight = True
        weights = None
    model = efficientnet_lite(1.0, 1.1, 240, 0.2, input_tensor = input_tensor, input_shape = input_shape, classes = classes, include_top = include_top, weights = weights)
    if hub_weight:
        model = load_weight(model, effnet_lite_urls["effnet_lite_b1"])
    return model

def efficientnet_lite_b2(input_tensor = None, input_shape = None, classes = 1000, include_top = True, weights = "imagenet"):
    hub_weight = False
    if weights == "imagenet":
        hub_weight = True
        weights = None
    model = efficientnet_lite(1.1, 1.2, 260, 0.3, input_tensor = input_tensor, input_shape = input_shape, classes = classes, include_top = include_top, weights = weights)
    if hub_weight:
        model = load_weight(model, effnet_lite_urls["effnet_lite_b2"])
    return model

def efficientnet_lite_b3(input_tensor = None, input_shape = None, classes = 1000, include_top = True, weights = "imagenet"):
    hub_weight = False
    if weights == "imagenet":
        hub_weight = True
        weights = None
    model = efficientnet_lite(1.2, 1.4, 280, 0.3, input_tensor = input_tensor, input_shape = input_shape, classes = classes, include_top = include_top, weights = weights)
    if hub_weight:
        model = load_weight(model, effnet_lite_urls["effnet_lite_b3"])
    return model

def efficientnet_lite_b4(input_tensor = None, input_shape = None, classes = 1000, include_top = True, weights = "imagenet"):
    hub_weight = False
    if weights == "imagenet":
        hub_weight = True
        weights = None
    model = efficientnet_lite(1.4, 1.8, 300, 0.3, input_tensor = input_tensor, input_shape = input_shape, classes = classes, include_top = include_top, weights = weights)
    if hub_weight:
        model = load_weight(model, effnet_lite_urls["effnet_lite_b4"])
    return model

if __name__ == '__main__':
    model = efficientnet_lite_b0(input_shape = (224, 224, 3), classes = 1000, include_top = True, weights = "imagenet")