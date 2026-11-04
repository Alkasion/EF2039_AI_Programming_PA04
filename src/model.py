"""Classifier definitions. All experiments share one architecture; only data changes."""
import tensorflow as tf


def build_model(cfg):
    num_classes = len(cfg["data"]["classes"])
    h, w = cfg["data"]["image_size"]
    backbone = cfg["train"]["backbone"]
    inputs = tf.keras.Input(shape=(h, w, 3))

    if backbone == "mobilenetv2":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        base = tf.keras.applications.MobileNetV2(
            input_shape=(h, w, 3), include_top=False, weights="imagenet", pooling="avg"
        )
        base.trainable = False
        x = base(x, training=False)
        x = tf.keras.layers.Dropout(0.2)(x)
    elif backbone == "small_cnn":
        x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
        for filters in (32, 64, 128):
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name=f"clf_{backbone}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["train"]["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
