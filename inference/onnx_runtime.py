import onnxruntime
import numpy as np
import cv2

def load_onnx_model(model_path):
    """
    Load an ONNX model from a file.

    Args:
    - model_path (str): Path to the ONNX model file.

    Returns:
    - session (onnxruntime.InferenceSession): An ONNX runtime inference session object.
    """
    session = onnxruntime.InferenceSession(model_path)
    return session

def preprocess_image(image, input_size):
    """
    Preprocess an input image for use with an ONNX model.

    Args:
    - image (numpy.ndarray): An input image as a NumPy array.
    - input_size (tuple): A tuple of (width, height) specifying the input size of the ONNX model.

    Returns:
    - image (numpy.ndarray): The preprocessed image as a NumPy array.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_size)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def run_onnx_inference(session, image):
    """
    Run inference on an input image using an ONNX model.

    Args:
    - session (onnxruntime.InferenceSession): An ONNX runtime inference session object.
    - image (numpy.ndarray): An input image as a NumPy array.

    Returns:
    - output (numpy.ndarray): The output of the ONNX model as a NumPy array.
    """
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: image})
    return output

if __name__ == '__main__':
    # Define input and model paths
    input_path = "/home/ubuntu/fall-detect/inference/person.jpg"
    model_path = "converters/model.onnx"

    # Load the ONNX model
    session = load_onnx_model(model_path)

    # Load the input image
    image = cv2.imread(input_path)

    # Preprocess the image
    input_size = (640, 640)
    image = preprocess_image(image, input_size)

    # Run inference on the image
    output = run_onnx_inference(session, image)