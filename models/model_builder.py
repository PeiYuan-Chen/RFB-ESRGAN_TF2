import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, UpSampling2D, Flatten, Dense, add
from tensorflow.keras.models import Model
from utils.conv2D_args import k3n64s1, k3n3s1, k3n256s1
from models.backbone.RRFDB import residual_of_receptive_field_dense_block
from models.backbone.RFB import receptive_field_block
from models.backbone.RRDB import residual_in_residual_dense_block


# def generator(kernel_initializer=tf.keras.initializers.GlorotNormal()):
#     inputs = Input(shape=(None, None, 3))
#     # pre-process
#     x = tf.keras.layers.Rescaling(scale=1.0 / 255)(inputs)
#     # shallow extraction
#     x = Conv2D(kernel_initializer=kernel_initializer, **k3n64s1)(x)
#     # trunk
#     lsc = x
#     for _ in range(23):
#         x = residual_of_receptive_field_dense_block(
#             x, kernel_initializer=kernel_initializer)
#     x = add([x, lsc])
#     # upsample nearest
#     x = Conv2D(kernel_initializer=kernel_initializer, **k3n64s1)(x)
#     x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
#     x = Conv2D(kernel_initializer=kernel_initializer, **k3n64s1)(x)
#     x = LeakyReLU(alpha=0.2)(x)

#     # reconstruct
#     x = Conv2D(kernel_initializer=kernel_initializer, **k3n64s1)(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Conv2D(kernel_initializer=kernel_initializer, **k3n3s1)(x)

#     # post-process
#     outputs = tf.keras.layers.Rescaling(scale=255)(x)
#     model = Model(inputs=inputs, outputs=outputs)
#     print(model.summary())
#     return model


def generator_x4(kernel_initializer=tf.keras.initializers.GlorotNormal()):
    inputs = Input(shape=(None, None, 3))
    # pre-process
    x = tf.keras.layers.Rescaling(scale=1.0 / 255)(inputs)
    # shallow extraction
    x = Conv2D(kernel_initializer=kernel_initializer, **k3n64s1)(x)
    # trunk
    lsc = x
    for _ in range(16):
        x = residual_in_residual_dense_block(
            x, kernel_initializer=kernel_initializer)
    for _ in range(8):
        x = residual_of_receptive_field_dense_block(
            x, kernel_initializer=kernel_initializer)
    x = add([x, lsc])
    x = receptive_field_block(
        x, input_channels=64, output_channels=64, kernel_initializer=kernel_initializer)
    # upsample nearest
    x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    x = receptive_field_block(
        x, input_channels=64, output_channels=64, kernel_initializer=kernel_initializer)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(kernel_initializer=kernel_initializer, **k3n256s1)(x)
    x = tf.nn.depth_to_space(x, block_size=2)
    x = receptive_field_block(
        x, input_channels=64, output_channels=64, kernel_initializer=kernel_initializer)
    x = LeakyReLU(alpha=0.2)(x)

    # reconstruct
    x = Conv2D(kernel_initializer=kernel_initializer, **k3n64s1)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(kernel_initializer=kernel_initializer, **k3n3s1)(x)

    # post-process
    outputs = tf.keras.layers.Rescaling(scale=255)(x)
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model


def discriminator(inputs, filter_num=64):
    # spatial_size=(128,128)
    assert inputs.shape[-2] == 128 and inputs.shape[-3] == 128
    # shallow extraction
    x = Conv2D(filters=filter_num,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same', )(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    # downsample
    x = Conv2D(filters=filter_num,
               kernel_size=(4, 4),
               strides=(2, 2),
               padding='same',
               use_bias=False)(x)  # output size /2
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=filter_num * 2,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               use_bias=False)(x)  # output feature map * 2
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=filter_num * 2,
               kernel_size=(4, 4),
               strides=(2, 2),
               padding='same',
               use_bias=False)(x)  # output size /4
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=filter_num * 4,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               use_bias=False)(x)  # output feature map * 4
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=filter_num * 4,
               kernel_size=(4, 4),
               strides=(2, 2),
               padding='same',
               use_bias=False)(x)  # output size /8
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=filter_num * 8,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               use_bias=False)(x)  # output feature map * 8
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=filter_num * 8,
               kernel_size=(4, 4),
               strides=(2, 2),
               padding='same',
               use_bias=False)(x)  # output size /16
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=filter_num * 16,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               use_bias=False)(x)  # output feature map * 16
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=filter_num * 16,
               kernel_size=(4, 4),
               strides=(2, 2),
               padding='same',
               use_bias=False)(x)  # output size /32
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # spatial size=(4,4)
    x = Flatten()(x)
    x = Dense(units=100)(x)
    x = LeakyReLU(alpha=0.2)(x)
    outputs = Dense(units=1)(x)
    # model
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model
