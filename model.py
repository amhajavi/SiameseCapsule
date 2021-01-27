from __future__ import print_function
from __future__ import absolute_import
import keras


weight_decay = 1e-4

class ModelMGPU(keras.Model):
    def __init__(self, ser_model, gpus):
        pmodel = keras.utils.multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

def ghostvlad_model_resnet(input_dim=(257, 250, 1), num_class=8631, mode='train'):

    from models.thin_resnet import thin_resnet

    inputs, x = thin_resnet(input_dim=input_dim, mode=mode)

    y = keras.layers.Dense(num_class, activation='softmax',
                               kernel_initializer='orthogonal',
                               use_bias=False, trainable=True,
                               kernel_regularizer=keras.regularizers.l2(weight_decay),
                               bias_regularizer=keras.regularizers.l2(weight_decay),
                               name='classification')(x)

    if mode == 'eval':
        y = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, 1))(x)

    model = keras.models.Model(inputs, y)

    # mgpu = len(keras.backend.tensorflow_backend._get_available_gpus())

    if mode == 'train':
        # if mgpu > 1:
        #     model = ModelMGPU(model, gpus=mgpu)
        # set up optimizer.
        opt = keras.optimizers.Adam(lr=1e-3)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    return model


def end_to_end_mlp_model(input_dim=(2, 4096)):

    inputs = keras.layers.Input(shape=input_dim, name='input')
    reshaped = keras.layers.Reshape(target_shape=(-1,))(inputs)

    fc_1 = keras.layers.BatchNormalization()(reshaped)
    fc_1 = keras.layers.Dense(2048, kernel_initializer='orthogonal',
                              use_bias=False, trainable=True,
                              name='fc_1')(fc_1)
    fc_1 = keras.layers.LeakyReLU()(fc_1)

    fc_2 = keras.layers.BatchNormalization()(fc_1)
    fc_2 = keras.layers.Dense(1024, kernel_initializer='orthogonal',
                              use_bias=False, trainable=True,
                              name='fc_2')(fc_2)
    fc_2 = keras.layers.LeakyReLU()(fc_2)

    fc_3 = keras.layers.BatchNormalization()(fc_2)
    fc_3 = keras.layers.Dense(512, activation='tanh',
                              kernel_initializer='orthogonal',
                              use_bias=False, trainable=True,
                              name='fc_3')(fc_3)

    binary_classification = keras.layers.Dense(1, activation='sigmoid',
                                               kernel_initializer='orthogonal',
                                               use_bias=False, trainable=True,
                                               name='classification')(fc_3)
    model = keras.models.Model(inputs=inputs, outputs=binary_classification)
    opt = keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
    return model


def end_to_end_capsule_model(input_dim=(2, 4096)):
    from models import capsule
    from end2end.models import custom_activation
    inputs = keras.layers.Input(shape=input_dim, name='input')
    dim_added = keras.layers.Reshape(target_shape=input_dim+(1,))(inputs)
    utter_1 = keras.layers.Lambda(lambda x: x[:, 0])(dim_added)
    utter_2 = keras.layers.Lambda(lambda x: x[:, 1])(dim_added)
    concatenated = keras.layers.Concatenate(axis=-1)([utter_1, utter_2])
    reshaped = keras.layers.Reshape(target_shape=(64, 64, 2))(concatenated)
    conv = keras.layers.Conv2D(64, (3, 3),
                   kernel_initializer='orthogonal',
                   use_bias=False,
                   kernel_regularizer=keras.regularizers.l2(1e-4),
                   name='conv_1')(reshaped)

    conv = keras.layers.Conv2D(128, (3, 3),
                                kernel_initializer='orthogonal',
                                use_bias=False,
                                kernel_regularizer=keras.regularizers.l2(1e-4),
                                name='conv_2')(conv)
    conv = keras.layers.Conv2D(256, (3, 3),
                                kernel_initializer='orthogonal',
                                use_bias=False,
                                kernel_regularizer=keras.regularizers.l2(1e-4),
                                name='conv_3')(conv)

    # batched = keras.layers.BatchNormalization()(conv)

    primary = capsule.PrimaryCap(concatenated, dim_capsule=2, n_channels=16, kernel_size=9, strides=1, padding='valid')

    capsulated = capsule.CapsuleLayer(4, 256)(concatenated)

    flat = keras.layers.Flatten()(capsulated)
    # binary_classification = custom_activation.CosFace(2)([flat, inputs])
    binary_classification =  keras.layers.Dense(1, activation='sigmoid',
                                               # kernel_initializer='orthogonal',
                                               name='classification')(flat)
    model = keras.models.Model(inputs=inputs, outputs=binary_classification)
    opt = keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
    return model