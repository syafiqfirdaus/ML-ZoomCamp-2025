import tensorflow as tf
import tf2onnx
import onnx

model = tf.keras.models.load_model('best_model.keras')

# Compatibility patch for Keras 3 / tf2onnx
if not hasattr(model, 'output_names'):
    if hasattr(model, 'outputs'):
        # Keras 3 outputs are KerasTensors which have names
        model.output_names = [x.name for x in model.outputs]
    else:
        model.output_names = ['output_0']

input_signature = [tf.TensorSpec([None, 7], tf.float32, name='features')]

onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)
onnx.save(onnx_model, "best_model.onnx")
print("Converted to best_model.onnx")
