import cv2
import numpy as np
import subprocess
import json

def get_joint_positions(image_path):
    # path to the OpenPose binary
    openpose_binary = "/home/luis/openpose/build/openpose.bin"
    
    # output file name with joint positions
    output_file = "output.json"

    # execute OpenPose. 
    cmd = [
        openpose_binary,
        "--image_dir", image_path,
        "--write_json", output_file,
        "--display", "0",
        "--render_pose", "0"
    ]
    
    subprocess.check_call(cmd)
    
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    # Extracting joint positions from the output
    keypoints = data[0]['people'][0]['pose_keypoints_2d']
    
    # Reshape to get (x, y, confidence) triplets
    keypoints = np.array(keypoints).reshape((-1, 3))
    
    return keypoints


def calculate_descent_velocity(yt2, yt1, t2, t1):
    """
    This function calculates the descent velocity of the person 
    using two consecutive frames
    :param yt2: y-coordinate of the person's hip joint at time t2
    :param yt1: y-coordinate of the person's hip joint at time t1
    :param t2: time 2
    :param t1: time 1
    :return: velocity of the person's descent
    
    """
    
    delta_t = t2 - t1
    v = (yt2 - yt1) / delta_t
    return v

def is_fall(v):
    """
    This function checks if the person's descent velocity is greater than the critical speed
    :param v: velocity of the person's descent
    :return: True if the person's descent velocity is greater than the critical speed, False otherwise
    """
    critical_speed = 0.009  # m/s
    return v >= critical_speed


def main():
    # Load your image
    img = cv2.imread("/home/luis/Downloads/deaa7a7d-18c3-43ab-950e-57809defa36e.png")

    # Extract joint positions (specifically hip joints 11 and 12 for this example)
    joint_positions = get_joint_positions(img)

    yt1 = joint_positions["joint_11"][1]  # y-coordinate of joint 11
    yt2 = joint_positions["joint_12"][1]  # y-coordinate of joint 12

    t1, t2 = 0, 1  # Assuming two consecutive frames, hence 1 second difference

    v = calculate_descent_velocity(yt2, yt1, t2, t1)

    if is_fall(v):
        print("Fall detected!")
    else:
        print("No fall detected.")

if __name__ == "__main__":
    main()