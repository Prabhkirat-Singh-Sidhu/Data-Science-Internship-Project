import tensorflow as tf
import os

try:
    import tensorflowjs as tfjs
    print("Converting model to TensorFlow.js...")
    model = tf.keras.models.load_model("model.h5")
    tfjs.converters.save_keras_model(model, "tfjs_model")
    print("Successfully converted! Check the 'tfjs_model' directory.")
except ImportError:
    print("tensorflowjs package not found. Installing it now...")
    os.system("pip install tensorflowjs")
    import tensorflowjs as tfjs
    model = tf.keras.models.load_model("model.h5")
    tfjs.converters.save_keras_model(model, "tfjs_model")
    print("Successfully converted! Check the 'tfjs_model' directory.")
