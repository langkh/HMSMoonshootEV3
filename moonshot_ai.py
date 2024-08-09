#NOTE: Before running this file:

#1: run ./rpyc_server.sh in the SSH terminal of your EV3 brick, and main.py on your physical brick (ensure PS4 controller is connected to your brick)
#2: run the line: "inference server start" (no quotes) in your terminal / command prompt for the AI model to inference or start your docker container
#This code imports libraries needed to interact with the EV3 brick and perform object detection
import rpyc #used to communicate with the Ev3 Mindstorms brick
import cv2 #used for video capture
from time import time #used for time tracking
import threading
import pygame
import numpy as np
from datetime import datetime
import ftd2xx as ftd
import tempfile
import os
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
import time # Explicitly import time module
#import async 
#------SETTINGS------------------------------------------------------------------------------------------------------------------------------------------------#

#enable / disable recording
enable_recording = False

#run code without robot connected
disable_robot = False

#AI mode toggle, disabled by default (slows system down!)
ai_mode_enabled = False

#PS4 controller threshold to prevent system from responding to joystick commands that are close, but not equal to 0
controller_threshold = 0.1
joystick_moving = False
#PS4 controller mode enabled by default, disabled during mouse or tumor tracking. Press "s" to stop all motors and ensure controller mode is enabled.
controller_mode_enabled = True 

#Mouse control enabled by default, disabled during mouse or tumor tracking. Press "s" to stop all motors and ensure mouse control mode is enabled.
mouse_mode_enabled = True 

#Foot pedal to control robot
footpedal_pressed = False
                                                                                    
#Change the number to select the camera input port for the endoscope camera (0-1)
camera_capture_port = 0
#Change the IP address to match the IP address of the EV3 brick
ip_address = '192.168.137.1'
#ip_address = '0.0.0.0'

#These variables are used to tune the system for tumor detection
tumor_target_accuracy = 25 #threshold for tumor tracking accuracy in pixelsq
tumor_track_duty_cycle = 2 #speed at which the robot moves to track the tumor
hor_inst_ofst = 100 #horizontal instrument offset for the biopsy tool to reachq the tumor (negative is more left)
vert_inst_ofst = -100 #vertical instrument offset for the biopsy tool to reach the tumor (negative is more upwards)
debug_tumor_track = 0 #turn 1`this on to show the error in the x and y directions for tuqmor tracking

#Deep Learning Model Inputs

model_id = "moonshot-crsfj/1"
smooth_id = "tumor-instance-segmentation/2"
uv_model_id= "moonshot-syab2/2"

roboflow_api_key = "IPgBWwHgstLbpATOG57L"
smooth_api_key = "Xz0yz1dpWbJegCjOk4AN"
uv_api_key = "fCsLSHS3kDjh9m2yYlyU"


camera_capture_port = 0

confidence_threshold = 0.6
iou_threshold=0.6

#This line sets a constant value for the speed of the motors when controlled by the PS4
yaw_motor_duty_cycle = 75
pitch_motor_duty_cycle =75
insert_motor_duty_cycle = 75

#This line sets a constant value for the speed of the motors when controlled by the mouse
yaw_mouse_duty_cycle = 50
pitch_mouse_duty_cycle =35
insert_mouse_speed= 1000
insert_mouse_time = 200 #milliseconds

#variables to ensure robot moves when mouse buttons are held down
mouse_pressed=0
#timer to resume controller control after mouse is released
mouse_timer =0
mouse_timer_threshold = 2



#------RUN TIME CODE------------------------------------------------------------------------------------------------------------------------------------------------#
#breaking main loop variable
exit =0
kepler_version_name = "Kepler AI V1.1"

# Keep-alive function
def keep_alive(conn, interval=30):
    """
    Send periodic heartbeat messages to keep the connection alive.
    :param conn: The RPyC connection object.
    :param interval: Time interval in seconds between heartbeats.
    """
    while True:
        try:
            conn.ping()
            time.sleep(interval)
        except (EOFError, ConnectionError):
            print("Connection lost.")
            break

#This line connects to the EV3 brick using the given IP address
if not(disable_robot): 
    conn = rpyc.classic.connect(ip_address, keepalive=True)
    
    # Start the keep-alive thread
    keep_alive_thread = threading.Thread(target=keep_alive, args=(conn,))
    keep_alive_thread.daemon = True
    keep_alive_thread.start()

#initialize AI model
config = InferenceConfiguration(confidence_threshold, iou_threshold)



#added two new datasets
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
class_ids = {}
#initialize instrument control hub
device = ftd.open(0) # open ftd device 0 (the only one if using just 1 board)
device.setBitMode(0xff, 1) # set all 8 bits (only 4 used on this board) to 


if not(disable_robot):
    #These lines set up the various components of the EV3 brick, like motors, buttons, and displays
    #The modules needed to interact with these components are imported and stored as variables
    motor = conn.modules['ev3dev2.motor']
    button = conn.modules['ev3dev2.button']
    sound = conn.modules['ev3dev2.sound']
    display = conn.modules['ev3dev2.display']
    sensor = conn.modules['ev3dev2.sensor']
    sensor_lego = conn.modules['ev3dev2.sensor.lego']
    ev3Button = button.Button()
    ev3Sound = sound.Sound()
    ev3Display = display.Display()

    #These lines configure the EV3 brick to recognize the motors and sensors connected to it
    #The motors are assigned to specific ports on the brick. 
    #You will need to change the Motor Port settings to match the way you have wired your robot.

    yawMotor = motor.Motor(motor.OUTPUT_A)
    pitchMotor = motor.Motor(motor.OUTPUT_B)
    insertMotor = motor.Motor(motor.OUTPUT_C)

#This line creates an object to capture video using the camera
#The video will be fed into Kepler's deep learning algorithms
player = cv2.VideoCapture(camera_capture_port)
# Get the width and height of the frame from the capture device
frame_width = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))

if enable_recording:
    # Define the codec and create a VideoWriter object to save the video
    # The filename includes the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_filename = f"record_{current_time}.mp4"
    out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (frame_width, frame_height))

#The following lines initialize variables for object detection and tracking
#We are creating four variables: mouseX, mouseY, tumorX, and tumorY, which will hold the coordinates of different objects on the screen
mouseX = 0 # This variable will hold the X-coordinate of the mouse cursor
mouseY = 0 # This variable will hold the Y-coordinate of the mouse cursor
tumorX = 0 # This variable will hold the X-coordinate of the tumor
tumorY = 0 # This variable will hold the Y-coordinate of the tumor

#We are also creating two boolean variables: tumor_location_available and tracking_mode_enabled,
#which will be used to track if the tumor is present and if the tumor detection mode is turned on or off
tumor_location_available = False # This variable will be used to track if the tumor location is available or not
tracking_mode_enabled = False # This variable will be used to track if the tumor detection mode is enabled or not

# Initialize pygame
pygame.init()



# Initialize the joystick
joystick = pygame.joystick.Joystick(0)
pygame.joystick.init()

def stop_all_motors():
    if not(disable_robot):
        yawMotor.stop(stop_action="brake")
        pitchMotor.stop(stop_action="brake")
        insertMotor.stop(stop_action="brake")

#This is a function that controls a robot using mouse movements on the screen
def mouse_click_robot(event,x,y,flags,param):
    
    if not(disable_robot) and mouse_mode_enabled:
        global controller_mode_enabled, mouse_pressed,mouse_timer,mouse_timer_threshold,tracking_mode_enabled
        # if the left button on the mouse is clicked...
        if event == cv2.EVENT_LBUTTONDOWN or mouse_pressed==1:
            mouse_pressed = 1
            controller_mode_enabled=False

            # get the height and width of the frame (picture)
            (h, w) = frame.shape[:2]
            # divide height and width by 2 to find the center of the frame
            w=w/2
            h=h/2
            
            # get the x and y position of the mouse click
            mouseX = x
            mouseY = y
            
            # if the mouse is moved to the right of the center of the frame, move the yaw motor left
            if mouseX-w > 0:
                yawMotor.run_direct(duty_cycle_sp=-yaw_mouse_duty_cycle)
            # if the mouse is moved to the left of the center of the frame, move the yaw motor right
            else:
                yawMotor.run_direct(duty_cycle_sp=yaw_mouse_duty_cycle)
    
            # if the mouse is moved above the center of the frame, move the pitch motor forward
            if mouseY-h > 0:
                pitchMotor.run_direct(duty_cycle_sp=pitch_mouse_duty_cycle)
            # if the mouse is moved below the center of the frame, move the pitch motor backward
            else:
                pitchMotor.run_direct(duty_cycle_sp=-pitch_mouse_duty_cycle)
        
        if event==cv2.EVENT_MOUSEWHEEL: 
            if y>0:                                
                insertMotor.run_timed(speed_sp=insert_mouse_speed,time_sp=insert_mouse_time)
            else:                
                insertMotor.run_timed(speed_sp=-insert_mouse_speed,time_sp=insert_mouse_time)
            
        #logic to ensure robot still moves when mouse is pressed
        if event ==cv2.EVENT_LBUTTONUP:
            stop_all_motors()
            mouse_pressed=0
            mouse_timer = mouse_timer + 1    
    
'''
The following code defines a function called "calculate_tumor_location", which is used to update the qcoordinates of the tumor based on the output of a deep 
learning model that detects objects in an image or video frame. The function takes in three parameters: "frame", "model", and "results". frame" represents the current
image or video frame being processed, "model" represents the deep learning model being used to detect objects, and "results" represents the output of the model after
detecting objects in the frame. #We are declaring three variables as global variables: tumorX, tumorY, and tracking_mode_enabled. These variables will be used later in the
function.
'''

def calculate_tumor_location(frame):
    #We are accessing the global variables tumorX, tumorY in this function
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
            uv_prediction_list = uv_prediction['predictions']#make sure that you include the two new datasets

            total_list = predictions_list+s_predictions_list
            print(prediction)
            # Initialize sum variables and count
            sum_x = 0
            sum_y = 0
            count = 0
            
              #lableing and coloring the tumor outlines (MAKE SURE DATASETS ARE IN INSTANCE SEGMENTATION ON ROBOFLOW AND NOT OBJECT DETECTION - "points" is not a key in the object detection dict return
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
                elif detection['class']=='light':
                    brgcolor = (255,255,255)
                
                contour = np.array([[int(point['x']), int(point['y'])] for point in points], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [contour], isClosed=True, color=brgcolor, thickness=2)
                cv2.putText(frame, detection['class'],(int(point['x']),int(point['y'])),cv2.FONT_HERSHEY_SIMPLEX,1,brgcolor,3)#labels! :)
            
            if count > 0:
                # Calculate midpoint
                midpoint_x = sum_x / count
                midpoint_y = sum_y / count
                
                # Update global variables                    
                tumorX = midpoint_x
                tumorY = midpoint_y

            # Ensure to delete the temp file after using it
        os.unlink(tmp_file.name)        #IF ON WINDOWS YOU MUST HAVE THE UNLINK() OUTSIDE THE "WITH___AS" CLAUSE BC OF FILE HANDLING ISSUES
    
    return frame
            
'''
This function helps the robot move towards the tumor target and perform a biopsy
It takes a "frame" (picture) as input and outputs motor commands to move the robot
this function uses information about the tumor's location to move a robot towards the tumor and perform a biopsy. 
The robot adjusts its pitch (up and down) and yaw (left and right) motors to center the tumor in the image.
'''

def tumor_track(frame):
    # Declare some variables as "global", meaning they can be accessed from outside the function
    global tumorX,tumorY, tracking_mode_enabled, tumor_target_accuracy, controller_mode_enabled, debug_tumor_track

    # Get the height and width of the input picture
    (h, w) = frame.shape[:2] #w:image-width and h:image-height

    # Divide the height and width by 2 (to get the center of the image)
    h=h/2
    w=w/2    


    # If the tumor is found and tracking is enabled:
    if tumorX!=0 and tumorY!=0 and tracking_mode_enabled==True and mouse_pressed==0:
        # Print the distance between the tumor location and the center of the image
        if  debug_tumor_track:
            print ("error_x, error_y",abs(tumorX-w),abs(tumorY-h))
       
        # If the tumor is close enough to the center of the image:
        if abs(tumorY+vert_inst_ofst-h)<tumor_target_accuracy and abs(tumorX+hor_inst_ofst-w) < tumor_target_accuracy:
            # Turn off tracking mode and print a message
            print("Tumor Locked")            
            stop_tumor_track()
        else:
            # If the tumor is not close enough, adjust the pitch motor (up and down)
            if abs(tumorY+vert_inst_ofst-h)>tumor_target_accuracy:
                if tumorY+vert_inst_ofst-h > tumor_target_accuracy:
                    #pitchMotor.run_direct(duty_cycle_sp=tumor_track_duty_cycle)
                    pitchMotor.on(tumor_track_duty_cycle)
                elif tumorY+vert_inst_ofst-h < tumor_target_accuracy:
                    #pitchMotor.run_direct(duty_cycle_sp=-tumor_track_duty_cycle)
                    pitchMotor.on(-tumor_track_duty_cycle)
            else:
                # Stop the pitch motor if the tumor is in the correct vertical position
                pitchMotor.stop()

            # If the tumor is not close enough, adjust the yaw motor (left and right)  
            if abs(tumorX+hor_inst_ofst-w)>tumor_target_accuracy:    
                if tumorX+hor_inst_ofst-w > tumor_target_accuracy:
                    #yawMotor.run_direct(duty_cycle_sp=-tumor_track_duty_cycle)
                    yawMotor.on(-tumor_track_duty_cycle)
                elif tumorX+hor_inst_ofst-w < tumor_target_accuracy:
                    #yawMotor.run_direct(duty_cycle_sp=tumor_track_duty_cycle)
                    yawMotor.on(tumor_track_duty_cycle)
            else:
                # Stop the yaw motor if the tumor is in the correct horizontal position
                yawMotor.stop()

cv2.namedWindow(kepler_version_name)

# Create and start a thread for mouse click handling
mouse_thread = threading.Thread(target=cv2.setMouseCallback, args=(kepler_version_name, mouse_click_robot))
mouse_thread.daemon = True
mouse_thread.start()


#enables tumor tracking
def enable_tumor_track():
    global tumorX, tumorY, ai_mode_enabled, tracking_mode_enabled,controller_mode_enabled,mouse_mode_enabled
    print("Tumor Tracking On")
    print("Controller and Mouse Control Disabled - press s to resume contol")
    tumorY=0
    tumorX=0
    ai_mode_enabled=True
    tracking_mode_enabled= True
    controller_mode_enabled=False    
    mouse_mode_enabled = False

#stocks tumor tracking parameters
def stop_tumor_track():
    global tumorX, tumorY, ai_mode_enabled, tracking_mode_enabled,controller_mode_enabled,mouse_mode_enabled,mouse_pressed
    print("Tumor Tracking Off")
    print("Controller and Mouse Control Enabled - press t to resume tumor tracking")
    tumorY=0
    tumorX=0
    ai_mode_enabled=False
    tracking_mode_enabled= False
    controller_mode_enabled=True    
    mouse_mode_enabled = True
    mouse_pressed=0
    stop_all_motors()

# Initialize last known states to prevent redundant motor commands

while(True):    
    ret, frame = player.read()    
    if ret:    
        calculate_tumor_location(frame)        

    cv2.imshow(kepler_version_name,frame)
    
    # Write the frame into the file 'out'
    if enable_recording:
        out.write(frame)

       
    # Get the state of the PS4 controller
    pygame.event.pump()        

    for event in pygame.event.get():
        # Check if a key is pressed (None of these work on a Windows so I suggest using the xbox controller buttons
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_b:#should be the key for foot pedal                    
                footpedal_pressed = True # Turns on all instruments
                device.write(bytes([0b000001111]))  # Turns on all instruments    
            elif event.key == pygame.K_a:
                if ai_mode_enabled:
                    ai_mode_enabled = False
                    print("AI Mode Tracking Off")
                else:
                    ai_mode_enabled=True
                    print("AI Mode Tracking On")
            elif event.key == pygame.K_q:
                exit=1
            elif event.key == pygame.K_t:                   
                if not(disable_robot): enable_tumor_track()                        
            elif event.key == pygame.K_s: 
                if not(disable_robot): stop_tumor_track()                                                        
                
        elif event.type == pygame.KEYUP:            
            footpedal_pressed = False        
            stop_all_motors()
            device.write(bytes([0b000000000]))  # Turns off instruments         

    if controller_mode_enabled:
        pygame.event.pump()  # Process the event queue to get the latest events
        # Joystick axes for movement
        left_x, left_y, right_y= joystick.get_axis(0), joystick.get_axis(1), joystick.get_axis(3)                  

        # Joystick buttons for actions
        x_button = joystick.get_button(0)
        triangle_button = joystick.get_button(3)
        square_button = joystick.get_button(2)
        circle_button = joystick.get_button(1)

        if not(disable_robot):        
            if abs(left_x)>controller_threshold or abs(left_y)>controller_threshold or abs(right_y)>controller_threshold:
                joystick_moving = True           
                if abs(left_x) > controller_threshold:
                    yawMotor.on(-int(left_x * yaw_motor_duty_cycle)) 
                else:
                    yawMotor.stop(stop_action="brake")

                if abs(left_y) > controller_threshold:
                    pitchMotor.on(int(left_y * pitch_motor_duty_cycle))           
                else:
                    pitchMotor.stop(stop_action="brake")
                
                if abs(right_y) > controller_threshold:
                    insertMotor.on(int(-right_y * insert_motor_duty_cycle))         
                else:
                    insertMotor.stop(stop_action="brake") 
            else:
                if joystick_moving:
                    stop_all_motors()
                    joystick_moving=False
                        
        # Process button presses only on change
        if x_button:
            if not(disable_robot): stop_all_motors()
            tracking_mode_enable02d = False
            device.write(bytes([0b000000000]))  # Turns off all instruments

        if triangle_button:
            if not(disable_robot):enable_tumor_track()            

        if square_button:
            if ai_mode_enabled:
                ai_mode_enabled = False
                print("AI Mode Tracking Off")
            else:
                ai_mode_enabled=True
                print("AI Mode Tracking On")  # Example action for square button press        

        if circle_button:
            device.write(bytes([0b000000100]))  # Example action for circle button press            
        
    if tracking_mode_enabled:
        tumor_track(frame)

    if mouse_timer!=0 and mouse_timer < mouse_timer_threshold:
        mouse_timer= mouse_timer + 1
    elif mouse_timer>=mouse_timer_threshold:
        if tracking_mode_enabled!=True:
            controller_mode_enabled = True
        mouse_timer = 0
            
    # Get the state of the PS4 controller
    pygame.event.pump()           
    if exit:
        break    
    
            
       
# After the main loop, clean up and join the tumor tracking and mouse threads
cv2.destroyAllWindows()
pygame.quit()
player.release()
mouse_thread.join()
if enable_recording:
    out.release()
