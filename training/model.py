"""DS-CNN keyword-spotting model (after Zhang et al., "Hello Edge", 2017).

A standard 10x4 convolution followed by depthwise-separable blocks. Splitting
each 3x3 convolution into a depthwise 3x3 plus a pointwise 1x1 cuts its MACs
by ~8x at 64 channels, which is what makes a 1 s window cheap enough for a
microcontroller. Every layer maps to an int8 TFLite Micro kernel that ESP-NN
accelerates on the ESP32-S3: CONV_2D, DEPTHWISE_CONV_2D, AVERAGE_POOL_2D,
FULLY_CONNECTED, SOFTMAX (BatchNorm and ReLU are folded/fused at export).
"""
import keras
from keras import layers


def build_dscnn(input_shape=(49, 10, 1), num_classes=12, filters=64, blocks=4, dropout=0.2,
                batch_size=None):
    # Short layer names: TFLite tensor names are derived from them and are
    # stored in the flatbuffer, i.e. in the device's flash.
    inp = keras.Input(shape=input_shape, batch_size=batch_size, name="mfcc")
    x = layers.Conv2D(filters, (10, 4), strides=(2, 2), padding="same", use_bias=False, name="c0")(inp)
    x = layers.BatchNormalization(name="b0")(x)
    x = layers.ReLU(name="r0")(x)
    for i in range(1, blocks + 1):
        x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False, name=f"dw{i}")(x)
        x = layers.BatchNormalization(name=f"bd{i}")(x)
        x = layers.ReLU(name=f"rd{i}")(x)
        x = layers.Conv2D(filters, (1, 1), use_bias=False, name=f"pw{i}")(x)
        x = layers.BatchNormalization(name=f"bp{i}")(x)
        x = layers.ReLU(name=f"rp{i}")(x)
    x = layers.Dropout(dropout, name="drop")(x)
    x = layers.AveragePooling2D(pool_size=(x.shape[1], x.shape[2]), name="pool")(x)
    # Fixed-size reshape (not Flatten): converts to a single RESHAPE with a
    # constant shape instead of a SHAPE/STRIDED_SLICE/PACK chain.
    x = layers.Reshape((filters,), name="flat")(x)
    out = layers.Dense(num_classes, activation="softmax", name="fc")(x)
    return keras.Model(inp, out, name="dscnn")


def count_macs(model):
    """Multiply-accumulates per inference for the layer types used above."""
    macs = 0
    for layer in model.layers:
        if isinstance(layer, layers.DepthwiseConv2D):
            _, h, w, c = layer.output.shape
            kh, kw = layer.kernel_size
            macs += h * w * c * kh * kw
        elif isinstance(layer, layers.Conv2D):
            _, h, w, c = layer.output.shape
            kh, kw = layer.kernel_size
            macs += h * w * c * kh * kw * layer.input.shape[-1]
        elif isinstance(layer, layers.Dense):
            macs += layer.input.shape[-1] * layer.units
    return int(macs)
