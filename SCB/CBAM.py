import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from tensorflow.keras import layers


# 判断输入数据格式，是channels_first还是channels_last
channel_axis = 1 if K.image_data_format() == "channels_first" else 3

# CAM
def channel_attention(input_xs, reduction_ratio=0.125):
    # get channel
    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = layers.GlobalMaxPooling2D()(input_xs)
    maxpool_channel = layers.Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = layers.GlobalAvgPool2D()(input_xs)
    avgpool_channel = layers.Reshape((1, 1, channel))(avgpool_channel)
    Dense_One = layers.Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    Dense_Two = layers.Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    # max path
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = layers.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
    # avg path
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = layers.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    channel_attention_feature = layers.Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = layers.Activation('sigmoid')(channel_attention_feature)
    return layers.Multiply()([channel_attention_feature, input_xs])

# SAM
def spatial_attention(channel_refined_feature):
    maxpool_spatial = layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = layers.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return layers.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)


def cbam_module(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = layers.Multiply()([channel_refined_feature, spatial_attention_feature])
    return layers.Add()([refined_feature, input_xs])