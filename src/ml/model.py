import glob
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Concatenate, MaxPooling2D, Conv2DTranspose, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Dropout, BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.python.client import device_lib

from src.ml.loss import dice_coef_loss, hausdorff_loss


# model construction


def conv_block(m, dim, bn, res, do=0):
    n = BatchNormalization()(m) if bn else m
    n = Dropout(do)(n) if do else m
    # inception
    tower_1 = Conv2D(dim, 1, padding='same', activation='linear')(n)
    tower_1 = LeakyReLU(alpha=0.1)(tower_1)
    tower_1 = Conv2D(dim, (3, 3), padding='same', activation='linear')(tower_1)
    tower_1 = LeakyReLU(alpha=0.1)(tower_1)
    tower_2 = Conv2D(dim, (1, 1), padding='same', activation='linear')(n)
    tower_2 = LeakyReLU(alpha=0.1)(tower_2)
    tower_2 = Conv2D(dim, (5, 5), padding='same', activation='linear')(tower_2)
    tower_2 = LeakyReLU(alpha=0.1)(tower_2)
    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(n)
    tower_3 = LeakyReLU(alpha=0.1)(tower_3)
    tower_3 = Conv2D(dim, (1, 1), padding='same', activation='linear')(tower_3)
    tower_3 = LeakyReLU(alpha=0.1)(tower_3)
    n = Concatenate()([tower_1, tower_2, tower_3])

    return Concatenate()([m, n]) if res else n


def level_block(m, dim, depth, inc, do, bn, mp, up, res):
    if depth > 0:
        n = conv_block(m, dim, bn, res)
        m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
        m = level_block(m, int(inc * dim), depth - 1, inc, do, bn, mp, up, res)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation='linear', padding='same')(m)
            m = LeakyReLU(alpha=0.1)(m)
        else:
            m = Conv2DTranspose(dim, 3, strides=2, activation='linear', padding='same')(m)
            m = LeakyReLU(alpha=0.1)(m)
        n = Concatenate()([n, m])
        m = conv_block(n, dim, bn, res)
    else:
        m = conv_block(m, dim, bn, res, do)
    return m


def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2.,
         dropout=0.5, batchnorm=True, maxpool=True, upconv=True, residual=True):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(out_ch, 1, activation='sigmoid')(o)
    return Model(inputs=i, outputs=o)


def create_model(params):
    model_params = params['model']
    PATCH_SHAPE = model_params['patch_shape']
    START_CH = model_params['start_ch']
    NUM_LAYERS = model_params['layers']
    DROPOUT = model_params['dropout']

    model = UNet(
        (PATCH_SHAPE[1], PATCH_SHAPE[2], PATCH_SHAPE[0]),  # bc tf expects (W, H, D)
        start_ch=START_CH,
        depth=NUM_LAYERS,
        out_ch=PATCH_SHAPE[0],
        dropout=DROPOUT,
        residual=True,
        upconv=False
    )

    return model


def load_latest_model(model_name):
    if model_name.endswith("hdf5"):
        model = load_model(model_name, custom_objects={'dice_coef_loss': dice_coef_loss})
        return model, -1

    model_filename = glob.escape(model_name) + "*.hdf5"
    files = glob.glob(model_filename)
    epoch = 0
    if len(files) != 0:
        for file in files:
            epoch0 = int(file.rsplit('.', 1)[0].rsplit('__', 1)[-1])
            if epoch0 > epoch:
                epoch = epoch0
                filename = file

        print("loading model: " + filename)
        model = load_model(filename, custom_objects={'dice_coef_loss': dice_coef_loss})
    else:
        model = []

    #print(model.summary())

    return model, epoch


def get_model(params, data_loader):
    model_params = params['model']
    model_name = data_loader.get_name()
    INIT_LEARNING_RATE = model_params['init_learning_rate']

    #tf.device('/gpu:0')
    print(device_lib.list_local_devices())

    model, epoch = load_latest_model(model_name)
    if epoch == 0:
        model = create_model(params)

    model.compile(
        optimizer=tf.train.AdamOptimizer(learning_rate=INIT_LEARNING_RATE),
        loss=dice_coef_loss
    )

    """
    TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']  # get TPU address
    tf.logging.set_verbosity(tf.logging.INFO)

    strategy = tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)
    )

    tpu_model = tf.contrib.tpu.keras_to_tpu_model(
        model,
        strategy=strategy,
    )

    tpu_model.fit_generator(
    """

    #print(model.summary())

    return model, epoch
