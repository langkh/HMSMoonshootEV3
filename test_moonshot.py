#A small script to test if the ai portion works without using the ev3 brick


import cv2  # used for video capture
from time import time  # used for time tracking
import threading
import pygame
import numpy as np
from datetime import datetime
import tempfile
import os
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

exit = 0
kepler_version_name = "Kepler AI V1.1"

ai_mode_enabled = False

img = r"C:\Users\schoo\Pictures\Screenshots\Screenshot 2024-07-22 190357.png"

frame = cv2.imread(img)
model_id = "moonshot-crsfj/1"
smooth_id = "tumor-instance-segmentation/2"
uv_model_id= "moonshot-syab2/2"

roboflow_api_key = "IPgBWwHgstLbpATOG57L"
smooth_api_key = "Xz0yz1dpWbJegCjOk4AN"
uv_api_key = "fCsLSHS3kDjh9m2yYlyU"


confidence_threshold = 0.6
iou_threshold = 0.6

config = InferenceConfiguration(confidence_threshold, iou_threshold)

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key=roboflow_api_key
)

smooth_client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key=smooth_api_key
)
uv_client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key=uv_api_key
)


client.configure(config)
client.select_model(model_id)



smooth_client.configure(config)
smooth_client.select_model(smooth_id)


uv_client.select_model(config)
uv_client.select_model(uv_model_id)
# Initialize the camera


def calculate_tumor_location(frame):
    # We are accessing the global variables tumorX, tumorY in this function
    global tumorX, tumorY

    if ai_mode_enabled:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Save the frame temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            cv2.imwrite(tmp_file.name, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

            prediction = client.infer(tmp_file.name)
            smooth_prediction = smooth_client.infer(tmp_file.name)
            uv_prediction = uv_client.infer(tmp_file.name)

            predictions_list = prediction['predictions']  # Access the 'predictions' key directly
            s_predictions_list = smooth_prediction['predictions']
            uv_prediction_list = uv_prediction['predictions']

            total_list = predictions_list + s_predictions_list 
       
            # Initialize sum variables and count
            sum_x = 0
            sum_y = 0
            count = 0

            for detection in total_list:
                points = detection['points']
                for point in points:
                    sum_x += int(point['x'])
                    sum_y += int(point['y'])
                    count += 1

                if detection['class'] == 'Blueberry-fat-attenuated':
                    brgcolor = (250, 58, 10)
                elif detection['class'] == 'grape - calcified':
                    brgcolor = (250, 10, 198)
                elif detection['class'] == 'Raspberry-cavitated':
                    brgcolor = (130, 10, 250)
                elif detection['class'] == 'Tumor':
                    brgcolor = (0, 100, 255)
                elif detection['class']=='medium':
                    brgcolor = (255,255,255)
                elif detection['class']=='high':
                    brgcolor = (255,255,255)
                elif detection['class']=='medium':
                    brgcolor = (255,255,255)

                contour = np.array([[int(point['x']), int(point['y'])] for point in points], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [contour], isClosed=True, color=brgcolor, thickness=2)
                cv2.putText(frame, detection['class'], (int(point['x']), int(point['y'])), cv2.FONT_HERSHEY_SIMPLEX, 1, brgcolor, 3)

            if count > 0:
                # Calculate midpoint
                midpoint_x = sum_x / count
                midpoint_y = sum_y / count

                # Update global variables
                tumorX = midpoint_x
                tumorY = midpoint_y

            # Ensure to delete the temp file after using it
        os.unlink(tmp_file.name)

    return frame

tryAI = input("Do you want to turn AI on? (y/n): ")

if tryAI.lower() == "y":
    ai_mode_enabled =True
    print("AI Mode Tracking On")
    calculate_tumor_location(frame)
    cv2.imshow(kepler_version_name,frame)
    cv2.waitKey(1000000000)
    
