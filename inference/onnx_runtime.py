import onnx_runtime
import numpy as np
import cv2

# Load the ONNX model
session = onnx_runtime.InferenceSession("converters/model.onnx")

# Load the input image
image = cv2.imread("input.jpg")

# Preprocess the image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (640, 640))
image = image.transpose((2, 0, 1))
image = np.expand_dims(image, axis=0)

# Run inference on the image
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
output = session.run([output_name], {input_name: image})