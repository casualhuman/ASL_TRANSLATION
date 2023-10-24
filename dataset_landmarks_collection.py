import multiprocessing
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
from IPython import display
import PIL
from PIL import Image
import pickle
import os



MODEL_PATH = "..\handlandmarker_model\hand_landmarker.task" #mediapipe hand landmarker



"""Colab-specific patches for functions."""

__all__ = ['cv2_imshow', 'cv_imshow']


def cv2_imshow(a):
  """A replacement for cv2.imshow() for use in Jupyter notebooks.

  Args:
    a: np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. For
      example, a shape of (N, M, 3) is an NxM BGR color image, and a shape of
      (N, M, 4) is an NxM BGRA color image.
  """
  a = a.clip(0, 255).astype('uint8')
  # cv2 stores colors as BGR; convert to RGB
  if a.ndim == 3:
    if a.shape[2] == 4:
      a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
    else:
      a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
  display.display(PIL.Image.fromarray(a))


# cv_imshow = cv2_imshow


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx, hand_landmarks in enumerate(hand_landmarks_list):
        # hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          hand_landmarks_proto,
          solutions.hands.HAND_CONNECTIONS,
          solutions.drawing_styles.get_default_hand_landmarks_style(),
          solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


# Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Load the input image.
# image_path = "C:\\Users\\abdul\\Downloads\\American\\r\\hand1_r_dif_seg_1_cropped.jpeg"# Save the resized image to a file
# image_path = "C:\\Users\\abdul\\Downloads\\American\\s\\hand1_s_left_seg_4_cropped.jpeg"
# image_path = "C:\\Users\\abdul\\Downloads\\American\\j\\hand1_j_dif_seg_5_cropped.jpeg"
# image_path = "C:\\Users\\abdul\\Downloads\\American\\q\\hand1_q_dif_seg_3_cropped.jpeg"

image_path = r"C:\Users\abdul\Downloads\dataset5\B\x\color_23_0237.png"

# TODO: Create a function that takes the image path and annotates it



def annotate_input_image(input_data):
    """
    Annotate an input image/frame with landmarks.

    Parameters:
    - input_data: either an image path (string) or an image frame (numpy array)
    
    Returns:
    - detection_result: result of the detector
    - annotated_image: image with landmarks drawn
    """

    if isinstance(input_data, str):
        image = cv2.imread(input_data)
        image = mp.Image.create_from_file(input_data)

    elif isinstance(input_data, np.ndarray):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=input_data)
    
    else:
        raise TypeError("Invalid input_data type. ")
    
    detection_result = detector.detect(image)

    #process the classification result by visualization here 
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    return detection_result, annotated_image 


# annotate_input_image(image_path)


def extract_keypoints(results):
    """ This extract the keypoints for each image and returns it in the hand_landmarks"""

    hand_landmarks = np.array([[result.x, result.y, result.z] for result in results.hand_landmarks[0] if results])
    hand_landmarks = hand_landmarks.flatten() if results.hand_landmarks else np.zeros(63)
    return hand_landmarks


def main():
   # Specify the parent folder path containing subfolders with images
    PARENT_FOLDER_PATHS = [r"C:\Users\abdul\Downloads\dataset5\B", r"C:\Users\abdul\Desktop\Dissertation\American"]
    OUTPUT_FOLDER = r"C:\Users\abdul\Desktop\Dissertation\NumpyFolder"
    CHECKPOINT_FILE = 'checkpoint.pkl'

    # Create an output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)



    #load checkpoint if exists 
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            landmark_data, labels, file_paths, processed_count, succesful_annotations, \
            failed_annotations, successful_paths, last_processed_dir, last_processed_file = pickle.load(f)

    else:
        processed_count, succesful_annotations, failed_annotations = 0, 0, 0
        successful_paths = {}
        landmark_data = []
        labels = []
        file_paths = []
        last_processed_dir = None
        last_processed_file = None

    START_PROCESSING_FILE = False
    start_processing_dir = last_processed_dir is None 

    for PARENT_FOLDER_PATH in PARENT_FOLDER_PATHS:
        for root, dirs, files in os.walk(PARENT_FOLDER_PATH):
            if root == last_processed_dir:
                start_processing_dir = True
                START_PROCESSING_FILE = last_processed_file is None # if there is no last processed file, start processing immediately

            # if we are past the last processed directory, process all files
            elif start_processing_dir:
                START_PROCESSING_FILE = True

            # no = 0
            # break_outer = False 

            for file_name in files:
                if not START_PROCESSING_FILE and file_name == last_processed_file:
                    START_PROCESSING_FILE = True
                    continue

                if START_PROCESSING_FILE and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    full_file_path = os.path.join(root, file_name)

                    try:
                        detection_result, annotated_image = annotate_input_image(full_file_path)

                        if detection_result.hand_landmarks:
                            succesful_annotations += 1
                            
                            successful_paths.setdefault(root, 0)
                            successful_paths[root] += 1

                            # Append the landmarks, labels, and file paths to the lists
                            # landmark_data.append(detection_result.hand_landmarks)
                            landmark_data.append(extract_keypoints(detection_result))
                            
                            print(f'Succesfullt processed {full_file_path}')
                            labels.append(root.split(os.sep)[-1])  # label is the name of the directory
                            file_paths.append(full_file_path)
                        else:
                            failed_annotations += 1

                    except Exception as e: 
                        print(f"Error processing {full_file_path}: {e}")
                        continue

                    processed_count += 1
                    # no += 1

                    # Update the last processed directory and file
                    last_processed_dir = root
                    last_processed_file = file_name

                    if processed_count % 100 == 0:
                        with open(CHECKPOINT_FILE, 'wb') as f: 
                            pickle.dump((landmark_data, labels, file_paths, processed_count, succesful_annotations, 
                                        failed_annotations, successful_paths, last_processed_dir, last_processed_file), f)
                            
            start_processing = False

    print(processed_count)
    print(f'succesful annotations: {succesful_annotations}, unsuccesful annotations: {failed_annotations}')
    print(successful_paths)

    # # Convert the lists to NumPy arrays
    landmark_data_np = np.array(landmark_data)
    labels_np = np.array(labels)
    file_paths_np = np.array(file_paths)

    # print(landmark_data_np[:10])
    # print(labels_np[:50])
    # print(file_paths_np[:50])


    # # Save the NumPy arrays to the output folder
    np.save(os.path.join(OUTPUT_FOLDER, 'landmark_data_newtest.npy'), landmark_data_np)
    np.save(os.path.join(OUTPUT_FOLDER, 'labels_new_test.npy'), labels_np)
    # np.save(os.path.join(OUTPUT_FOLDER, 'file_paths.npy'), file_paths_np)

if __name__ == "__main__":
    main()