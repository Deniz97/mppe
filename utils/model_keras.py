from __future__ import print_function

import keras
import keras_resnet
import keras_resnet.models
from keras.layers import Input, Activation, Conv2D, Conv2DTranspose, Concatenate, Add, UpSampling2D, \
    Lambda, Maximum, Average, Multiply, Cropping2D

from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model
from utils.cpn_model import conv, stage1_block, stageT_block, apply_mask
import os
import tensorflow
from keras import layers

weight_decay = 5e-4
weights_path101 = '/home/muhammed/.keras/models/ResNet-101-model.keras.h5'
weights_path50 = '/home/muhammed/.keras/models/ResNet-50-model.keras.h5'

options = {
    'padding': 'same',
    'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
    'bias_initializer': 'zeros',
    'kernel_regularizer': keras.regularizers.l2(weight_decay)
}


def relu(x):
    return Activation(activation='relu')(x)

def resize_images(*args, **kwargs):
    return tensorflow.image.resize_images(*args, **kwargs)

class UpsampleLike(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        return resize_images(source, (target_shape[1], target_shape[2]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)

def cascaded_pyramid_features(C2, C3, C4, C5, feature_size=256, trainable=True):
    pyra_opt = {
        'padding': 'same',
        'trainable': trainable
    }
    P5           = Conv2D(feature_size, kernel_size=1, strides=1, name='P5', **pyra_opt)(C5)
    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, C4])
    P5_upsampled = Conv2D(feature_size, kernel_size=1, strides=1, name='P5_1', **pyra_opt)(P5_upsampled)

    # add P5 elementwise to C4
    P4           = Conv2D(feature_size, kernel_size=1, strides=1, name='C4_reduced', **pyra_opt)(C4)
    P4           = Add(name='P4_merged')([P5_upsampled, P4])
    P4           = Conv2D(feature_size, kernel_size=3, strides=1, name='P4', **pyra_opt)(P4)
    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, C3])
    P4_upsampled = Conv2D(feature_size, kernel_size=1, strides=1, name='P4_1', **pyra_opt)(P4_upsampled)

    # add P4 elementwise to C3
    P3 = Conv2D(feature_size, kernel_size=1, strides=1, name='C3_reduced', **pyra_opt)(C3)
    P3 = Add(name='P3_merged')([P4_upsampled, P3])
    P3 = Conv2D(feature_size, kernel_size=3, strides=1, name='P3', **pyra_opt)(P3)
    P3_upsampled = UpsampleLike(name='P3_upsampled')([P3, C2])
    P3_upsampled = Conv2D(feature_size, kernel_size=1, strides=1, name='P3_1', **pyra_opt)(P3_upsampled)

    P2 = Conv2D(feature_size, kernel_size=1, strides=1, name='C2_reduced', **pyra_opt)(C2)
    P2 = Add(name='P2_merged')([P3_upsampled, P2])
    P2 = Conv2D(feature_size, kernel_size=3, strides=1, name='P2', **pyra_opt)(P2)

    return P2, P3, P4, P5

def create_pyramid_features(C2, C3, C4, C5, feature_size=256):
    # upsample C5 to get P5 from the FPN paper
    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
    P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3           = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3_upsampled = layers.UpsampleLike(name='P3_upsampled')([P3, C3])
    P3           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # add P4 elementwise to C3
    P2 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced')(C2)
    P2 = keras.layers.Add(name='P2_merged')([P3_upsampled, P2])
    P2 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P2')(P2)

    return P2, P3, P4, P5

def dense_pyramid_features(C2, C3, C4, C5, feature_size=256):
    ks = 3
    F = Add

    P5           = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(C5)
    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, C4])

    # add P5 elementwise to C4
    P4           = Conv2D(feature_size, kernel_size=ks, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = F(name='P4_merged')([P5_upsampled, P4])
    P4           = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)
    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, C3])

    # add P4 elementwise to C3
    P3           = Conv2D(feature_size, kernel_size=ks, strides=1, padding='same', name='C3_reduced')(C3)
    P3           = F(name='P3_merged')([P4_upsampled, P3])
    P3           = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)
    P3_upsampled = UpsampleLike(name='P3_upsampled')([P3, C2])

    P2           = Conv2D(feature_size, kernel_size=ks, strides=1, padding='same', name='C2_reduced')(C2)
    P2           = F(name='P2_merged')([P3_upsampled, P2])
    P2           = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P2')(P2)

    return P2, P3, P4, P5

def ups(inp, level=0, final=False):
    nb = 256
    step = 1
    ks = 3

    out = inp
    for i in range(step):
        out = Conv2D(nb, kernel_size=ks, **options)(out)
        out = Activation('relu')(out)

    if not final:
        out = Add()([out,inp])
        out = UpSampling2D()(out)
        return out
    else:
        int = out
        out = Conv2D(128, kernel_size=1, **options)(out)
        out = Activation('relu')(out)
        out = Conv2D(17, kernel_size=1, name='ana_%d'%(level), **options)(out)
        return out, int

def bottleneck(features, up=ups):
    up_feats = []
    P2, P3, P4, P5 = features

    k = 1

    x = up(P5)
    x = up(x)
    x = up(x)
    out_5, int_5 = up(x, level=5, final=True)

    x = up(P4)
    x = up(x)
    out_4, int_4 = up(x, level=4, final=True)

    x = up(P3)
    out_3, int_3 = up(x, level=3, final=True)

    x = Conv2D(256, kernel_size=3, **options)(P2)
    int_2 = Activation('relu')(x)
    x = Conv2D(128, kernel_size=1, **options)(int_2)
    x = Activation('relu')(x)
    out_2 = Conv2D(17, kernel_size=1, name='ana_%d'%(2), **options)(x)

    outputs = [out_2, out_3, out_4, out_5]
    inters = [int_2, int_3, int_4, int_5]

    return inters, outputs

def init_features(features, upsample=False):
    outs = []
    for i,f in enumerate(features):
        x = f

        if upsample:
            for j in range(i):
                x = UpSampling2D()(x)
                x = Conv2D(256, kernel_size=7, **options)(x)

        x = Conv2D(128, kernel_size=1, **options)(x)
        x = Activation('relu')(x)
        x = Conv2D(17, kernel_size=1, name='yan_%d'%(i+2), **options)(x)
        outs.append(x)
    return outs

def init_features_multi(features, mask, upsample=False):
    num_parts = 18
    outs = []
    test_outs = []
    for i,f in enumerate(features):
        x = f
        if upsample:
            for j in range(i):
                x = UpSampling2D()(x)
                x = Conv2D(256, kernel_size=5, **options)(x)
                # x = Conv2D(256, kernel_size=1, **options)(x)
                # x = Conv2D(256, kernel_size=1, **options)(x)
                if i == 3 and j == i-1:
                    x = Cropping2D(cropping=((2,2), (2,2)))(x)
        x = Conv2D(128, kernel_size=1, **options)(x)
        x = Activation('relu')(x)
        x = Conv2D(num_parts, kernel_size=1, **options)(x)
        test_outs.append(x)
        x = Multiply(name='yan_%d'%(i+2))([x,mask])
        outs.append(x)
    return outs, test_outs

def init_dilated_multi(features, mask):
    num_parts = 18
    outs = []
    test_outs = []
    for i,f in enumerate(features):
        x = f
        if i > 0:
            x = UpSampling2D()(x)
            x = Conv2D(256, kernel_size=1, **options)(x)

        x = Conv2D(128, kernel_size=1, **options)(x)
        x = Activation('relu')(x)
        x = Conv2D(num_parts, kernel_size=1, **options)(x)
        test_outs.append(x)
        x = Multiply(name='yan_%d'%(i+2))([x,mask])
        outs.append(x)
    return outs, test_outs

def init_super_features(features, mask):
    num_parts = 18
    outs = []
    cont = []
    test_outs = []
    for i,f in enumerate(features):
        x = f
        for j in range(i):
            x = UpSampling2D()(x)
            x = Conv2D(256, kernel_size=1, **options)(x)
            # x = Conv2D(256, kernel_size=1, **options)(x)
            # x = Conv2D(256, kernel_size=1, **options)(x)
            # if i == 3 and j == i-1:
            #     x = Cropping2D(cropping=((2,2), (2,2)))(x)

        cont.append(x)
        x = Conv2D(128, kernel_size=1, **options)(x)
        x = Activation('relu')(x)
        x = Conv2D(num_parts, kernel_size=1, **options)(x)
        test_outs.append(x)
        x = Multiply(name='yan_%d'%(i+2))([x,mask])
        outs.append(x)
    return outs, cont, test_outs

def build_models():

    input = Input((None, None, 3))
    backbone = keras_resnet.models.ResNet101(input, include_top=False, freeze_bn=True)

    print('Loading weights from %s' % weights_path101)
    backbone.load_weights(weights_path101, by_name=True)

    C2, C3, C4, C5 = backbone.outputs
    features = cascaded_pyramid_features(C2, C3, C4, C5)

    outputs = []
    outs = init_features(features)
    outputs = outputs + outs

    inters, outs = bottleneck(features)
    outputs = outputs + outs

    last_steps = 3
    x = Concatenate()(inters)
    for i in range(last_steps):
        x = Conv2D(256, kernel_size=3, **options)(x)
        x = Activation('relu')(x)
    x = Conv2D(128, kernel_size=1, **options)(x)
    x = Activation('relu')(x)
    x = Conv2D(17, kernel_size=1, name='son', **options)(x)

    outputs = outputs + [x]

    train_model = Model(inputs=input, outputs=outputs)
    test_model = Model(inputs=input, outputs=x)

    return train_model, test_model

def build_global_net():
    weights_path = '/home/muhammed/.keras/models/ResNet-101-model.keras.h5'

    input = Input((None, None, 3))
    backbone = keras_resnet.models.ResNet101(input, include_top=False, freeze_bn=True)

    print('Loading weights from %s' % weights_path)
    backbone.load_weights(weights_path, by_name=True)

    C2, C3, C4, C5 = backbone.outputs
    features = cascaded_pyramid_features(C2, C3, C4, C5)

    outputs = init_features(features, upsample=True)

    train_model = Model(inputs=input, outputs=outputs)
    test_model = Model(inputs=input, outputs=outputs)

    return train_model, test_model

def build_multi_global_net(upsample=True, dilated=True):
    num_parts = 18

    input = Input((None, None, 3))
    mask = Input((None, None, num_parts))

    inputs = [input, mask]

    if dilated:
        import resnet.models
        backbone = resnet.models.ResNet101(input, include_top=False, freeze_bn=True)
    else:
        backbone = keras_resnet.models.ResNet101(input, include_top=False, freeze_bn=True)

    print('Loading weights from %s' % weights_path101)
    backbone.load_weights(weights_path101, by_name=True)

    C2, C3, C4, C5 = backbone.outputs
    features = cascaded_pyramid_features(C2, C3, C4, C5)

    if dilated:
        tr_outs, test_outs = init_dilated_multi(features, mask)
    else:
        tr_outs, test_outs = init_features_multi(features, mask, upsample=upsample)
    P2 = features[0]


    train_model = Model(inputs=inputs, outputs=tr_outs)
    test_model = Model(inputs=inputs, outputs=test_outs)

    return train_model, test_model

def build_res50(upsample=True, dilated=True):
    num_parts = 18

    input = Input((None, None, 3))
    mask = Input((None, None, num_parts))

    inputs = [input, mask]

    if dilated:
        import resnet.models
        backbone = resnet.models.ResNet50(input, include_top=False, freeze_bn=True)
    else:
        backbone = keras_resnet.models.ResNet50(input, include_top=False, freeze_bn=True)

    print('Loading weights from %s' % weights_path50)
    backbone.load_weights(weights_path50, by_name=True)

    C2, C3, C4, C5 = backbone.outputs
    features = cascaded_pyramid_features(C2, C3, C4, C5)

    if dilated:
        tr_outs, test_outs = init_dilated_multi(features, mask)
    else:
        tr_outs, test_outs = init_features_multi(features, mask, upsample=upsample)

    train_model = Model(inputs=inputs, outputs=tr_outs)
    test_model = Model(inputs=inputs, outputs=test_outs)

    return train_model, test_model

def build_super_net():
    num_parts = 18

    input = Input((None, None, 3))
    mask = Input((None, None, num_parts))

    outputs = []
    inputs = [input, mask]
    test_outs = []
    backbone = keras_resnet.models.ResNet101(input, include_top=False, freeze_bn=True)

    print('Loading weights from %s' % weights_path101)
    backbone.load_weights(weights_path101, by_name=True)

    C2, C3, C4, C5 = backbone.outputs
    features = cascaded_pyramid_features(C2, C3, C4, C5)

    outs, cont, test_out = init_super_features(features, mask)

    test_outs = test_outs + test_out
    outputs = outputs + outs

    x = Concatenate()(cont)

    # Additional non rpn layers
    x = conv(x, 512, 3, "conv4_3_CPM", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv4_4_CPM", (weight_decay, 0))
    feats = relu(x)

    stages = 2

    stage1_out = stage1_block(feats, num_parts, 1, weight_decay)
    test_outs.append(stage1_out)
    w = apply_mask(stage1_out, mask, 1)

    x = Concatenate()([stage1_out, feats])

    outputs.append(w)

    for sn in range(2, stages + 1):
        stageT_out = stageT_block(x, num_parts, sn, 1, weight_decay)
        test_outs.append(stageT_out)
        w = apply_mask(stageT_out, mask, sn)

        outputs.append(w)

        if sn < stages:
            x = Concatenate()([stageT_out, feats])


    train_model = Model(inputs=inputs, outputs=outputs)
    test_model = Model(inputs=input, outputs=test_outs[-1])

    return train_model, test_model

def build_last_net(upsample=True, dilated=True):
    num_parts = 18

    input = Input((None, None, 3))
    mask = Input((None, None, num_parts))

    inputs = [input, mask]
    outputs = []

    if dilated:
        import resnet.models
        backbone = resnet.models.ResNet101(input, include_top=False, freeze_bn=True)
    else:
        backbone = keras_resnet.models.ResNet101(input, include_top=False, freeze_bn=True)

    # cpm_weights = '/home/muhammed/DEV_LIBS/developments/keypoint-detection/fpn-pack/exp/meg_dil_480/cpm-weights.h5'
    res_weights = '/home/muhammed/DEV_LIBS/developments/keypoint-detection/fpn-pack/exp/meg_dil_480/weights.08.h5'

    for l in backbone.layers:
        l.trainable = False

    C2, C3, C4, C5 = backbone.outputs
    features = cascaded_pyramid_features(C2, C3, C4, C5, trainable=False)

    # if dilated:
    #     tr_outs, test_outs = init_dilated_multi(features, mask)
    # else:
    #     tr_outs, test_outs = init_features_multi(features, mask, upsample=upsample)

    # outputs = outputs + tr_outs

    x = features[0]
    x = conv(x, 128, 3, "conv4_1_CPM", (weight_decay, 0))
    feats = relu(x)

    stages = 4

    stage1_out = stage1_block(feats, num_parts, 1, weight_decay)
    #test_outs.append(stage1_out)
    w = apply_mask(stage1_out, mask, 1)

    x = Concatenate()([stage1_out, feats])

    outputs.append(w)

    for sn in range(2, stages + 1):
        stageT_out = stageT_block(x, num_parts, sn, 1, weight_decay)
        #test_outs.append(stageT_out)
        w = apply_mask(stageT_out, mask, sn)
        outputs.append(w)
        if sn < stages:
            x = Concatenate()([stageT_out, feats])

    train_model = Model(inputs=inputs, outputs=outputs)
    # print('Loading weights from %s' % cpm_weights)
    # train_model.load_weights(cpm_weights, by_name=True)
    print('Loading weights from %s' % res_weights)
    train_model.load_weights(res_weights, by_name=True)

    return train_model

def build_sharp_net():
    input = Input((None, None, 3))
    backbone = keras_resnet.models.ResNet101(input, include_top=False, freeze_bn=True)

    print('Loading weights from %s' % weights_path101)
    backbone.load_weights(weights_path101, by_name=True)

    C2, C3, C4, C5 = backbone.outputs
    features = cascaded_pyramid_features(C2, C3, C4, C5)
    x = features[0]
    # outputs = init_features(features)
    x = Conv2D(128, kernel_size=1, **options)(features[0])
    x = Activation('relu')(x)
    output = Conv2D(17, kernel_size=1, name='out', **options)(x)

    train_model = Model(inputs=input, outputs=output)
    test_model = Model(inputs=input, outputs=output)

    return train_model, test_model

def build_multi_sharp_net():
    input = Input((None, None, 3))
    mask = Input((None, None, 19))

    backbone = keras_resnet.models.ResNet101(input, include_top=False, freeze_bn=True)

    print('Loading weights from %s' % weights_path101)
    backbone.load_weights(weights_path101, by_name=True)

    C2, C3, C4, C5 = backbone.outputs
    features = cascaded_pyramid_features(C2, C3, C4, C5)
    x = features[0]
    # outputs = init_features(features)
    x = Conv2D(128, kernel_size=1, **options)(features[0])
    x = Activation('relu')(x)
    x = Conv2D(19, kernel_size=1, **options)(x)
    output = Multiply(name='out')([x, mask])

    train_model = Model(inputs=[input, mask], outputs=output)
    test_model = Model(inputs=input, outputs=x)

    return train_model, test_model

def build_global_cpn():
    input = Input((None, None, 3))

    backbone = keras_resnet.models.ResNet101(input, include_top=False, freeze_bn=True)

    print('Loading weights from %s' % weights_path101)
    backbone.load_weights(weights_path101, by_name=True)

    C2, C3, C4, C5 = backbone.outputs
    features = cascaded_pyramid_features(C2, C3, C4, C5)

    outputs = init_features(features)

    from cpn_model import stage1_block, stageT_block, single_upsample_block, meg_features
    from cpn_model import conv, relu

    x = meg_features(features, single_upsample_block, weight_decay)

    stages = 6
    np = 17

    # Additional non rpn layers
    x = conv(x, 512, 3, "conv4_3_CPM", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv4_4_CPM", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv4_5_CPM", (weight_decay, 0))
    x = relu(x)
    w = conv(x, np, 1, "weight_stage%d" % (0), (weight_decay, 0))
    outputs.append(w)

    feats = x

    # stage 1
    stage1_out = stage1_block(feats, np, 1, weight_decay)


    w_name = "weight_stage%d" % (1)
    w = Activation(activation='linear', name=w_name)(stage1_out)

    x = Concatenate()([stage1_out, feats])
    outputs.append(w)

    for sn in range(2, stages + 1):
        stageT_out = stageT_block(x, np, sn, 1, weight_decay)

        w_name = "weight_stage%d" % (sn)
        w = Activation(activation='linear', name=w_name)(stageT_out)

        outputs.append(w)

        if sn < stages:
            x = Concatenate()([stageT_out, feats])

    train_model = Model(inputs=input, outputs=outputs)
    test_model = Model(inputs=input, outputs=outputs[-1])

    return train_model, test_model

def build_global_refine():
    input = Input((None, None, 3))
    backbone = keras_resnet.models.ResNet101(input, include_top=False, freeze_bn=True)

    print('Loading weights from %s' % weights_path101)
    backbone.load_weights(weights_path101, by_name=True)

    C2, C3, C4, C5 = backbone.outputs
    features = cascaded_pyramid_features(C2, C3, C4, C5)

    outputs = init_features(features)

    from cpn_model import stage1_block, stageT_block, single_upsample_block, meg_features
    from cpn_model import conv, relu

    x = meg_features(features, single_upsample_block, weight_decay)

    stages = 6
    np = 17

    # Additional non rpn layers
    fs = 7
    x = conv(x, 512, fs, "conv4_3_CPM", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, fs, "conv4_4_CPM", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, fs, "conv4_5_CPM", (weight_decay, 0))
    x = relu(x)
    w = conv(x, np, 1, "weight_stage%d" % (0), (weight_decay, 0))
    outputs.append(w)

    train_model = Model(inputs=input, outputs=outputs)
    test_model = Model(inputs=input, outputs=outputs)

    return train_model, test_model

def build_stuff_net():
    num_parts = 19

    input = Input((None, None, 3))
    mask = Input((None, None, num_parts))

    inputs = [input, mask]

    backbone = keras_resnet.models.ResNet101(input, include_top=False, freeze_bn=False)

    print('Loading weights from %s' % weights_path101)
    backbone.load_weights(weights_path101, by_name=True)

    C2, C3, C4, C5 = backbone.outputs
    features = cascaded_pyramid_features(C2, C3, C4, C5)

    down = []
    for f in features:
        x = f
        x = Conv2D(128, kernel_size=3, **options)(x)
        x = relu(x)
        x = Conv2D(128, kernel_size=3, **options)(x)
        x = relu(x)
        down.append(x)

    up = []
    for i,d in enumerate(down):
        x = d
        for j in range(i):
            x = UpSampling2D()(x)
            x = Conv2D(128, kernel_size=1, **options)(x)
        up.append(x)

    x = Concatenate()(up)
    x = Conv2D(128, kernel_size=3, **options)(x)
    x = Conv2D(num_parts, kernel_size=1, **options)(x)
    x = Multiply()([x, mask])

    model = Model(inputs=inputs, outputs=x)

    return model

def build_stuff_br_net(inter=True):
    num_parts = 19

    input = Input((None, None, 3))
    mask = Input((None, None, num_parts))

    inputs = [input, mask]

    backbone = keras_resnet.models.ResNet101(input, include_top=False, freeze_bn=False)

    print('Loading weights from %s' % weights_path101)
    backbone.load_weights(weights_path101, by_name=True)

    C2, C3, C4, C5 = backbone.outputs
    features = cascaded_pyramid_features(C2, C3, C4, C5)

    down = []
    for f in features:
        x = f
        x = Conv2D(128, kernel_size=7, **options)(x)
        x = relu(x)
        x = Conv2D(128, kernel_size=7, **options)(x)
        x = relu(x)
        down.append(x)

    up = []
    if inter:
        outputs = []
    for i,d in enumerate(down):
        x = d
        # c = 0
        for j in range(i):
            r = range(0,i)[::-1]
            x = UpSampling2D()(x)
            x = Add()([x,down[r[j]]])
            x = Conv2D(128, kernel_size=1, **options)(x)
            x = relu(x)
            # c += 1
        up.append(x)
        if inter:
            x = Conv2D(num_parts, kernel_size=1, **options)(x)
            x = Multiply()([x, mask])
            outputs.append(x)

    x = Concatenate()(up)
    x = Conv2D(256, kernel_size=7, **options)(x)
    x = relu(x)
    x = Conv2D(256, kernel_size=7, **options)(x)
    x = relu(x)
    x = Conv2D(num_parts, kernel_size=1, **options)(x)
    x = Multiply()([x, mask])

    if not inter:
        outputs = x
    else:
        outputs.append(x)
    model = Model(inputs=inputs, outputs=outputs)

    return model

def build_hg():
    num_parts = 19

    input = Input((None, None, 3))
    mask = Input((None, None, num_parts))

    inputs = [input, mask]

    backbone = keras_resnet.models.ResNet101(input, include_top=False, freeze_bn=False)

    print('Loading weights from %s' % weights_path101)
    backbone.load_weights(weights_path101, by_name=True)

    C2, C3, C4, C5 = backbone.outputs
    P2, P3, P4, P5 = create_pyramid_features(C2, C3, C4, C5)

    H5 = UpSampling2D()(P5)
    H5 = Add()([H5,P4])
    H5 = UpSampling2D()(H5)

    H4 = UpSampling2D()(P4)
    H4 = Add()([H4, H5, P3])


    model = Model(inputs=inputs, outputs=x)

    return model

import keras.backend as K

def eucl_loss(batch_size):
    def _eucl_loss(x, y):
        return K.sum(K.square(x - y)) / batch_size / 2
    return _eucl_loss

def save_model(model, exp_dir, filename):
    print ('Saving checkpoint')
    file_name = os.path.join(exp_dir, filename)
    model.save(filepath=file_name)

from utils.losses import eucl_loss

custom_objects = {'UpsampleLike': UpsampleLike,
                  'resize_images': resize_images,
                  'eucl_loss': eucl_loss}

custom_objects.update(keras_resnet.custom_objects)