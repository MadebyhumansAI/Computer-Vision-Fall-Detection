import numpy as np
import tensorflow as tf
from PIL import Image

def preprocess_image(image_path, input_size=(640, 640)):
    """Preprocess an image for YOLOv5 model."""
    image = Image.open(image_path)
    image = image.resize(input_size)
    image_array = np.array(image, dtype=np.uint8)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def postprocess_output(raw_output, score_threshold=0.3, iou_threshold=0.5):
    """Post-process the raw output."""
    boxes = raw_output[..., :4]
    objectness = raw_output[..., 4:5]
    class_probs = raw_output[..., 5:]

    scores = objectness * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold
    )
    return boxes, scores, classes, valid_detections

def run_inference(tflite_model_path, image_path):

    """
    Run inference on an image with given TFLite model.

    Args:
    - tflite_model_path (str): Path to TFLite model.
    - image_path (str): Path to input image.

    Returns:
    - boxes (tensor): Predicted bounding boxes.
    - scores (tensor): Predicted bounding box scores.
    - classes (tensor): Predicted class indices.
    - valid_detections (tensor): Number of valid detections.

    """

    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the input image
    input_data = preprocess_image(image_path)

    # Set input tensor data
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the raw output
    raw_output = interpreter.get_tensor(output_details[0]['index'])

    # Post-process the raw output
    boxes, scores, classes, valid_detections = postprocess_output(raw_output)
    return boxes, scores, classes, valid_detections

# Example usage:
image_path = "/home/luis/fall-detect/inference/person.jpg"
tflite_model_path = "/home/luis/fall-detect/inference/yolo5s.tflite"
boxes, scores, classes, valid_detections = run_inference(tflite_model_path, image_path)

print(f"Boxes: {boxes}")
print(f"Scores: {scores}")
print(f"Classes: {classes}")
print(f"Valid Detections: {valid_detections}")
