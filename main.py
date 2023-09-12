import cv2
from utils import *
from PIL import Image
import torch

def main():
    # default hyperparameters
    history = 0  # the minimum number of frames that a pixel must persist to be considered for an object
    varThreshold = 100  # increasing detects less false positives
    area_threshold = 400  # the contour area required to be considered for an object (minimum contour area)
    classifier_threshold = 0.05  # minimum confidence probability required to classify/filter image as a bee/moth
    num_classes = 2
    device = 'cpu'
    task_name = 'moth'
    video_filepath = f"./{task_name}_task/deployment_data/deployment_moth.mp4" # file path for video to predict on
    generate_chips_dir = None#f"./{task_name}_task/training_data/unlabeled_chips" #None if not generating chips
    weights_path = 'moth_task_2-class_resnet18.pth'

    cv2.namedWindow('Your Window Name', cv2.WINDOW_NORMAL)  # Create a resizable window
    cv2.resizeWindow('Your Window Name', 1080, 480)  # Set the desired width and height

    model, _, _ = get_model(num_classes, device, weights_path)
    transform = get_transforms()
    print(f"Loading {video_filepath}")
    cap = cv2.VideoCapture(video_filepath)  # initializes a video capture object for accessing video frames from either a video file

    object_detector = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold)  # separates foreground objects from background objects

    frame_counter = 0
    while True:
        frame_counter = frame_counter + 1
        ret, frame = cap.read()  # gets each frame from video capture
        mask = object_detector.apply(frame)  # applies the background subtractor and creates a binary mask for the current frame
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # finds contours from the binary image mask
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)  # Calculate area and remove small elements
            if area > area_threshold:
                x, y, w, h = cv2.boundingRect(cnt)  # creates a bounding box from the limits of the contour polygon
                #if not generate_chips_dir:  # uncomment this line and the line below for all motion tracking boxes
                    #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3) #for motion tracking (red boxes)
                # a bounding box is defined by an x,y pixel coordinate and its width and height
                input_image = frame[y:y + h, x:x + w]  # crops the frame to the coordinates given by cv2.boundingRect()
                if generate_chips_dir:
                    print(f"saving to {generate_chips_dir}\{frame_counter}_{i}.jpg")
                    cv2.imwrite(f'{generate_chips_dir}\{frame_counter}_{i}.jpg', input_image)
                input_image = Image.fromarray(input_image)
                input_tensor = transform(input_image).unsqueeze(0)

                model.eval()
                with torch.no_grad():
                    outputs = model(input_tensor)
                _, predicted_class = outputs.max(1)

                if not generate_chips_dir:
                    if torch.sigmoid(outputs)[0][0] > classifier_threshold:
                        #if more than threshhold% sure its the variable draw a box aka detect
                        #print(predicted_class)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
                    #if predicted_class == 0: #is it the most likely class, then draw a box
                        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)


        cv2.imshow('Your Window Name', frame)

        key = cv2.waitKey(30)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindow()
    return

if __name__ == "__main__":
    main()
