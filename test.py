import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

from recognition_lib.FaceRecognition import FaceRecognition


import os

cuda_path_cudnn = os.path.join(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDNN\8.9\bin")
print(cuda_path_cudnn)
cuda_1_4 = os.path.join(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin")
print(cuda_1_4)

def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports,non_working_ports


def calculate_disparity(keypoints1, keypoints2, focal_length, baseline_width):
    if len(keypoints1) != 5 or len(keypoints2) != 5:
        raise ValueError("You need exactly 5 corresponding keypoints.")

    disparities = []
    for i in range(5):
        x1, y1 = keypoints1[i]
        x2, y2 = keypoints2[i]

        # Calculate the distance between corresponding points
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # Calculate disparity using the formula
        disparity = (baseline_width * focal_length) / distance
        disparities.append(disparity)

    return disparities

#list_ports()

fce = FaceRecognition('onnx')


cap_1= cv2.VideoCapture(2+cv2.CAP_DSHOW)
cap_2 = cv2.VideoCapture(1+cv2.CAP_DSHOW)

cap_1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cap_2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


cap_1.set(cv2.CAP_PROP_BUFFERSIZE, 2)
cap_2.set(cv2.CAP_PROP_BUFFERSIZE, 2)

def draw(image, items):
    for item in items:

        roi = item['roi']
        keypoints = item['keypoints']
        score = item['score']

        # Extract coordinates
        x1, y1= roi
        keypoints = [(int(x), int(y)) for x, y in keypoints]

        # Draw the rectangle
        image_with_rect = cv2.rectangle(image, x1, y1, (0, 255, 0), 2)

        # Draw keypoints
        for keypoint in keypoints:
            cv2.circle(image_with_rect, keypoint, 5, (0, 0, 255), -1)

        # Add text with score
        cv2.putText(image_with_rect, f'Score: {score:.2f}', (x1[0], x1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def visualize_depth_map(disparity_map):
    # Normalize the disparity map to the 0-255 range for visualization
    min_disp = disparity_map.min()
    max_disp = disparity_map.max()
    normalized_disp = (disparity_map - min_disp) / (max_disp - min_disp) * 255

    # Convert the normalized disparity map to an 8-bit image
    depth_map = normalized_disp.astype(np.uint8)

    # Apply a colormap to make it visually informative (e.g., 'COLORMAP_JET')
    depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

    return depth_map_colored

def merge_2d_disparity_to_3d(keypoints_2d_left, keypoints_2d_right, disparities, baseline_width, focal_length):
    keypoints_3d = []

    for i in range(len(keypoints_2d_left)):
        x_left, y_left = keypoints_2d_left[i]
        x_right, y_right = keypoints_2d_right[i]
        disparity = disparities[i]

        # Calculate the depth (Z-coordinate) from disparity
        depth = (baseline_width * focal_length) / disparity

        # Calculate the X and Y coordinates in 3D
        x_3d = (x_left + x_right) / 2
        y_3d = (y_left + y_right) / 2

        keypoints_3d.append((x_3d, y_3d, depth))

    return keypoints_3d

def visualize_3d_points(keypoints_3d, title="3D Keypoints Visualization"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_coords = [keypoint[0] for keypoint in keypoints_3d]
    y_coords = [keypoint[1] for keypoint in keypoints_3d]
    z_coords = [keypoint[2] for keypoint in keypoints_3d]

    # Plot the 3D keypoints
    ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o', label='3D Keypoints')

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the title of the plot
    plt.title(title)

    # Add a legend
    ax.legend()

    # Show the 3D plot
    plt.show()

def fov_to_focal_length(image_width, fov_angle):
    fov_radians = math.radians(fov_angle)
    focal_length = image_width / (2 * math.tan(fov_radians / 2))
    return focal_length

def normalize_points_with_numpy(points):
    points_array = np.array(points)
    
    min_vals = points_array.min(axis=0)
    max_vals = points_array.max(axis=0)
    
    # Perform element-wise normalization
    normalized_points = (points_array - min_vals) / (max_vals - min_vals)
    
    return normalized_points.tolist()


# Example usage:
#image_width = 1280  # Example image width in pixels
image_width = 5.856
fov_angle = 100  # Example FOV angle in degrees

#stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)
#focal_length = 1000  # Example focal length in pixels
baseline_width = 900  # Example baseline width in centimeters

focal_length = fov_to_focal_length(image_width, fov_angle)
print("Focal Length:", focal_length)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#sc = ax.scatter([], [], [], c='b', marker='o', label='3D Keypoints')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("Live 3D Keypoints Visualization")
ax.legend()


while True:
    ret_1, frame_1 = cap_1.read()
    ret_2, frame_2 = cap_2.read()


    if ret_1 != None and ret_2 != None:
        frame_1 = cv2.resize(frame_1, (640, 480))
        frame_2 = cv2.resize(frame_2, (640, 480))


        print(fce.verify_one(frame_1,frame_2))

        # now, you can do this either vertical (one over the other):
        print(frame_1.shape)
        print(frame_2.shape)

        results1 = fce.face_detection(frame_1)
        draw(frame_1, results1)

        results2 = fce.face_detection(frame_2)
        draw(frame_2, results2)

        kp1 = results1[0]['keypoints']
        kp2 = results2[0]['keypoints']

        disparities = calculate_disparity(kp1, kp2, focal_length, baseline_width)
        kpts = merge_2d_disparity_to_3d(kp1, kp2, disparities, baseline_width, focal_length)
        #visualize_3d_points(kpts, "3D Keypoints Visualization")
        #print(kpts)

        #gray_image1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
        #gray_image2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

        #disparity = stereo.compute(gray_image1, gray_image2)
        #depth_image = visualize_depth_map(disparity)
        #update_plot(kpts)
        kpts = normalize_points_with_numpy(kpts)

        x_coords = [keypoint[0] for keypoint in kpts]
        y_coords = [keypoint[1] for keypoint in kpts]
        z_coords = [keypoint[2] for keypoint in kpts]

        # Plot the 3D keypoints
        ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o', label='3D Keypoints')
        # Pause to allow the plot to update (adjust the interval as needed)
        plt.draw()
        plt.pause(0.02)
        ax.cla()
        final = cv2.vconcat([frame_1, frame_2])
        #cv2.imshow("II", depth_image)

        cv2.imshow("I", final)
        cv2.waitKey(10)