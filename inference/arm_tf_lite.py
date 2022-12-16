import numpy as np
import tensorflow as tf
from PIL import Image

def preprocess_image(image_path, input_size=(640, 640)):
    """
    Preprocess an image for input to a YOLOv5 model.

    Args:
    - image_path (str): Path to the image file to preprocess.
    - input_size (tuple): A tuple of two integers representing the desired input size of the model.
                          Default is (640, 640).

    Returns:
    - image_array (numpy.ndarray): A NumPy array representing the preprocessed image.
                                    The array has shape (1, height, width, channels),
                                    where height and width are the input size of the model,
                                    and channels is 3 for RGB images.
                                    The array has dtype np.uint8.

    Raises:
    - FileNotFoundError: If the image file at image_path does not exist.
    - ValueError: If the input_size argument is not a tuple of two integers.

    """

    image = Image.open(image_path)
    image = image.resize(input_size)
    image_array = np.array(image, dtype=np.uint8)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def postprocess_output(raw_output, score_threshold=0.3, iou_threshold=0.5):

    
    """
    Post-process the raw output of a TensorFlow Lite object detection model.

    Args:
    - raw_output (numpy.ndarray): A NumPy array representing the raw output of the model.
                                  The array has shape (batch_size, num_anchors, num_classes + 5),
                                  where batch_size is the number of images in the batch,
                                  num_anchors is the number of anchor boxes per image,
                                  num_classes is the number of object classes,
                                  and the last dimension contains the box coordinates,
                                  objectness scores, and class probabilities.
    - score_threshold (float): A float representing the minimum score threshold for object detection.
                               Default is 0.3.
    - iou_threshold (float): A float representing the intersection over union (IoU) threshold for non-max suppression.
                             Default is 0.5.

    Returns:
    - boxes (numpy.ndarray): A NumPy array representing the bounding boxes of the detected objects.
                             The array has shape (batch_size, max_detections, 4),
                             where max_detections is the maximum number of detections per image.
                             The last dimension contains the box coordinates in the format (ymin, xmin, ymax, xmax).
    - scores (numpy.ndarray): A NumPy array representing the scores of the detected objects.
                              The array has shape (batch_size, max_detections).
    - classes (numpy.ndarray): A NumPy array representing the classes of the detected objects.
                               The array has shape (batch_size, max_detections).
    - valid_detections (numpy.ndarray): A NumPy array representing the number of valid detections per image.
                                         The array has shape (batch_size,).

    This function performs non-max suppression on the raw output of a TensorFlow Lite object detection model.
    It first extracts the box coordinates, objectness scores, and class probabilities from the raw output.
    It then computes the scores of the detected objects as the product of the objectness scores and class probabilities.
    It applies non-max suppression to the scores and box coordinates using the combined_non_max_suppression function
    from the TensorFlow image module. The function returns the bounding boxes, scores, classes, and valid detections
    of the detected objects.
    """
    
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

# test plaatje
image_path = "/home/ubuntu/fall-detect/inference/person.jpg"
tflite_model_path = "/home/ubuntu/fall-detect/inference/yolo5s.tflite"
boxes, scores, classes, valid_detections = run_inference(tflite_model_path, image_path)

print(f"Boxes: {boxes}")
print(f"Scores: {scores}")
print(f"Classes: {classes}")
print(f"Valid Detections: {valid_detections}")
