# Subramanya Bhatt, Prabhanvi Technologies, TraffiCount - Vehicle counting and Classification system

# Import necessary packages

import cv2
import csv
import collections
import numpy as np
from trafficount_tracker import *
from random import randint
import sys, getopt
import math
from datetime import datetime

inputfile = ''
outputfile = ''
startframetime = ''
endframetime = ''
#print ((sys.argv))
args = []
args = (sys.argv)
inputfile = str(args[1])
if len(args) > 2:
    outputfile = args[2]
if len(args) > 3:
    startframetime = args[3]
if len(args) > 4:
    endframetime = args[4]
#print (inputfile.split('.')[0])
#print (outputfile)

#inputfile = 'Ch1Toll1.mp4'

# Traffic direction
horizontal_traffic = 'YES'

# Initialize Tracker
tracker = EuclideanDistTracker()

# Initialize the videocapture object
# cap = cv2.VideoCapture('video.mp4')
cap = cv2.VideoCapture(inputfile)
input_size = 320
frameSizeRatio = 0.4
iw = int((cap.get(3))*frameSizeRatio)
ih = int((cap.get(4))*frameSizeRatio)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
time_length = total_frame_count/fps


if endframetime == '':
    endframetime = time_length

speedscale = 1
framesGap = 5 #for detection skip frames
#framesGap=5600
frame_number =0

#add logo to the final image
logo = cv2.imread('prabhanvi/PrabhanviLogo.jpg')
logoScale = 0.5
logo_width = int(logo.shape[1]*logoScale)
logo_height = int(logo.shape[0]*logoScale)
logo_dim = (logo_width, logo_height)
logo_wm = cv2.resize(logo, logo_dim, interpolation=cv2.INTER_AREA)


outVideo = cv2.VideoWriter(inputfile.split('.')[0] +'_output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), int((fps/(framesGap+1))*speedscale), (iw,ih))

#print(str(fps) + 'fps | Total frames:' + str(total_frame_count) + ' | Length: ' + str(time_length) + 'sec |' + str(time_length/60) + 'min')
print(str(fps) + 'fps | Total frames:' + str(total_frame_count) + ' | Length: ' + str(round(time_length/60,2)) + ' min')     

# Detection confidence threshold
confThreshold =0.2
nmsThreshold= 0.15

font_color = (230, 200, 255)
font_size = 1.25*frameSizeRatio
font_thickness = 1
reportLocation_top = int(120*frameSizeRatio)
reportLineGap = int(40*frameSizeRatio)
reportLocation_columnTabHeading1 = int(0.58*iw*frameSizeRatio)
reportLocation_columnTabHeading2 = int(0.75*iw*frameSizeRatio)
reportLocation_left = int(0.1*iw*frameSizeRatio)
reportLocation_columnTab1 = int(0.58*iw*frameSizeRatio)
reportLocation_columnTab2 = int(0.75*iw*frameSizeRatio)
linenumber=0

# Middle cross line position
middle_line_position = int(ih*0.325)   
up_line_position = middle_line_position - int(ih*0.09)
down_line_position = middle_line_position + int(ih*0.1)
# Middle vertical line position
hor_middle_line_position = int(iw*0.0)
right_mask_line = int(iw*0.7)
left_mask_line = int(iw*0.12)
hor_up_line_position = hor_middle_line_position - int(iw*0.0)
hor_down_line_position = hor_middle_line_position + int(iw*0.0)

# Store Coco Names in a list
classesFile = "trafficount_vehicle_class.names"
classNames = open(classesFile).read().strip().split('\n')
#print(classNames)
#print(len(classNames))

# class index for our required detection classes
required_class_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

detected_classNames = []

## Model Files

modelConfiguration = 'yolov3_highway_trafficount.cfg'
modelWeigheights = 'yolov3_highway_trafficount.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backend
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy
    
# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
down_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
temp_box_list = []

##############################################
# Function for count vehicle
def count_vehicle(box_id, img):

    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center
    #print('ix : ' + str(ix))
    # Find the current position of the vehicle towards toll
    if ((ix > hor_up_line_position) and (ix < hor_middle_line_position)) or((iy > up_line_position) and (iy < middle_line_position)):
        if id not in temp_up_list:
            temp_up_list.append(id)

    elif ((ix < hor_down_line_position and ix > hor_middle_line_position) or (iy < down_line_position and iy > middle_line_position)):
        if id not in temp_down_list:
            temp_down_list.append(id)
            
    elif ix < hor_up_line_position or iy > up_line_position:
        if id in temp_down_list:
            print ('u==-' + str(id)  + ':' + str(index) + ':' + str(ix) + ',' + str(iy) + ' || ' + str(up_list) + ' || ' + str(down_list))
            temp_down_list.remove(id)
            #if ix > left_mask_line:
            if ix > left_mask_line or ix < right_mask_line:
                up_list[index] = up_list[index]+1

    elif ix > hor_down_line_position or iy < down_line_position:
        if id in temp_up_list:
            print ('d==-' + str(id)  + ':' + str(index) + ':' + str(ix) + ',' + str(iy) + ' || ' + str(up_list) + ' || ' + str(down_list))
            temp_up_list.remove(id)
            
            down_list[index] = down_list[index] + 1
            
    ##### ---------- 
    

    #####  ---------

    # Draw circle in the middle of the rectangle
    #cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    # print(up_list, down_list)

    # Draw ID in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    cv2.putText(img, str(id), center, cv2.FONT_HERSHEY_SIMPLEX, font_size*0.75, (0, 150, 255), font_thickness*1)
#    cv2.putText(img, str(id), center, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness*1)
#    print(str(id), up_list, down_list)

# Function for finding the detected objects from the network output
def postProcess(outputs,img):
    global detected_classNames 
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    #print (boxes)
    #for i in indices.flatten():
    for i in indices:
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        # print(x,y,w,h)

        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        detected_classNames.append(name)
        # Draw classname and confidence score 
        #cv2.putText(img,f'  {classId} {name.upper()} {int(confidence_scores[i]*100)}%', find_center(x, y, w, h), cv2.FONT_HERSHEY_SIMPLEX, font_size*0.75, color, 1)

        # Draw bounding rectangle
        # cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(classIds[i])])
#        print (detection)
#        print (len(detection))

    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    for box_id in boxes_ids:
        count_vehicle(box_id, img)



##############################################

def realTime():
#    total_frame_count
    frame_number = 0
    linenumber = 0

    analysis_start_time = datetime.now()

    if startframetime != '':
        #for frameNum in range(int(round(int(endframetime)-int(startframetime),2)*fps*60)):
        for frameNum in range(int(round(int(startframetime),2)*fps*60)):
                success, img1 = cap.read()
                #        total_frame_count
                frame_number = frame_number + 1

    while True:
        success, img = cap.read()
        img = cv2.resize(img,(0,0),None,frameSizeRatio,frameSizeRatio)
        ih, iw, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)
        
        # Set the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()
        #outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
        outputNames = [(layersNames[i - 1]) for i in net.getUnconnectedOutLayers()]
        # Feed data to the network
        outputs = net.forward(outputNames)
        #print (outputs)
        #print("---")
    
        # Find the objects from the network output
        postProcess(outputs,img)
        
        
        # Draw counting texts in the frame
        cv2.putText(img, "Up", (reportLocation_left+reportLocation_columnTabHeading1, reportLocation_top), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Down", (reportLocation_left+reportLocation_columnTabHeading2, reportLocation_top), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # list Bicycle on screen
        linenumber=linenumber+1
        cv2.putText(img, "Bicycle:", (reportLocation_left, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(up_list[13]), (reportLocation_left+reportLocation_columnTab1, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(down_list[13]), (reportLocation_left+reportLocation_columnTab2, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # list Bullock cart on screen
        linenumber=linenumber+1
        cv2.putText(img, "Bullock cart:", (reportLocation_left, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(up_list[16]), (reportLocation_left+reportLocation_columnTab1, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(down_list[16]), (reportLocation_left+reportLocation_columnTab2, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # list Two wheeler on screen
        linenumber=linenumber+1
        cv2.putText(img, "Two wheeler:", (reportLocation_left, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(up_list[1]+up_list[12]), (reportLocation_left+reportLocation_columnTab1, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(down_list[1]+down_list[12]), (reportLocation_left+reportLocation_columnTab2, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # list cars on screen
        linenumber=linenumber+1
        cv2.putText(img, "Car:", (reportLocation_left, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(up_list[0]), (reportLocation_left+reportLocation_columnTab1, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(down_list[0]), (reportLocation_left+reportLocation_columnTab2, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # list Autorickshaw on screen
        linenumber=linenumber+1
        cv2.putText(img, "Autorickshaw:", (reportLocation_left, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(up_list[4]), (reportLocation_left+reportLocation_columnTab1, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(down_list[4]), (reportLocation_left+reportLocation_columnTab2, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # list Luggage Autorickshaw on screen
        linenumber=linenumber+1
        cv2.putText(img, "Luggage Auto:", (reportLocation_left, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(up_list[14]), (reportLocation_left+reportLocation_columnTab1, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(down_list[14]), (reportLocation_left+reportLocation_columnTab2, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # list Tractor on screen
        linenumber=linenumber+1
        cv2.putText(img, "Tractor:", (reportLocation_left, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(up_list[15]), (reportLocation_left+reportLocation_columnTab1, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(down_list[15]), (reportLocation_left+reportLocation_columnTab2, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # list Tractor on screen
        linenumber=linenumber+1
        cv2.putText(img, "LCV:", (reportLocation_left, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(up_list[11]), (reportLocation_left+reportLocation_columnTab1, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(down_list[11]), (reportLocation_left+reportLocation_columnTab2, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # list MiniBus on screen
        linenumber=linenumber+1
        cv2.putText(img, "Mini Bus:", (reportLocation_left, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(up_list[8]), (reportLocation_left+reportLocation_columnTab1, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(down_list[8]), (reportLocation_left+reportLocation_columnTab2, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # list BUS on screen
        linenumber=linenumber+1
        cv2.putText(img, "Bus:", (reportLocation_left, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(up_list[2]), (reportLocation_left+reportLocation_columnTab1, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(down_list[2]), (reportLocation_left+reportLocation_columnTab2, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # list Truck on screen
        linenumber=linenumber+1
        cv2.putText(img, "Truck (2 Axle):", (reportLocation_left, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(up_list[3]+up_list[6]), (reportLocation_left+reportLocation_columnTab1, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(down_list[3]+down_list[6]), (reportLocation_left+reportLocation_columnTab2, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # list Trailer on screen
        linenumber=linenumber+1
        cv2.putText(img, "Multi-Axle:", (reportLocation_left, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(up_list[5]), (reportLocation_left+reportLocation_columnTab1, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(down_list[5]), (reportLocation_left+reportLocation_columnTab2, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        # list Trailer on screen
        linenumber=linenumber+1
        cv2.putText(img, "EME/JCB:", (reportLocation_left, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(up_list[9]), (reportLocation_left+reportLocation_columnTab1, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, str(down_list[9]), (reportLocation_left+reportLocation_columnTab2, reportLocation_top+reportLineGap*linenumber), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

#standatd text imprint
        cv2.putText(img, 'Prabhanvi Technologies - TraffiCount', (int(iw/2), ih-10), cv2.FONT_HERSHEY_SIMPLEX, font_size*0.75, (200,245,18), font_thickness*1)
        cv2.putText(img, '[9986379673]', (int(iw*0.8), ih-10), cv2.FONT_HERSHEY_SIMPLEX, font_size*0.75, (200,245,18), font_thickness*1)
        
        #cv2.putText(img, 'Video length: ' + str(round(time_length/60,2)) + ' min',(int(iw*0.6), ih-25), cv2.FONT_HERSHEY_SIMPLEX, font_size*0.75, (200,245,18), font_thickness*1)
        
        completed_time = str(math.floor(((time_length*frame_number)/total_frame_count)/60)) + ':' + str(round(((((time_length*frame_number)/total_frame_count)/60) - (math.floor(((time_length*frame_number)/total_frame_count)/60)))*60, 2)) + ' / ' + str(round(time_length/60,2)) + ' min'
        #print(int(round(((time_length*frame_number)/total_frame_count)-(((time_length*frame_number)/total_frame_count)/60),0)))
        #print("Elapsed video duration: " + str(math.floor(((time_length*frame_number)/total_frame_count)/60)) + ":" + str(round(((((time_length*frame_number)/total_frame_count)/60)-(math.floor(((time_length*frame_number)/total_frame_count)/60)))*60, 2)) + ' *')
        td = datetime.now() - analysis_start_time
        tr = 0
        tr2complete = 0
        if frame_number > 25:
            tr = ((datetime.now() - analysis_start_time)*total_frame_count)/frame_number
            tr2complete = (((datetime.now() - analysis_start_time)*total_frame_count)/frame_number) - (datetime.now() - analysis_start_time)
            
            
        # print('Analysis progress: ' + completed_time + '| ' + str(tr2complete))
        linenumber = 0
        
        cv2.putText(img, 'Analysis progress: ' + completed_time, (int(iw*0.6), ih-40), cv2.FONT_HERSHEY_SIMPLEX, font_size*0.75, (200,245,18), font_thickness*1)
        cv2.putText(img, 'Time to Complete: ' + str(tr2complete), (int(iw*0.6), ih-25), cv2.FONT_HERSHEY_SIMPLEX, font_size*0.75, (200,245,118), font_thickness*1)
        
#include logo        
        logo_dim = (iw, ih)
        logo_wm = cv2.resize(logo, logo_dim, interpolation=cv2.INTER_AREA)

        img = cv2.addWeighted(img, 1.0, logo_wm, 0.1, 0.0)

        # write output video file
        outVideo.write(img)
        
        # Draw the crossing lines
        # if horizontal_traffic == 'YES':
            # horizontal for up-down
        cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
        cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 0, 255), 1)
        cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 0, 255), 1)

        # vertical for left-right
        
        cv2.line(img, (left_mask_line, 0), (left_mask_line, ih), (255, 0, 255), 2)
        cv2.line(img, (right_mask_line, 0), (right_mask_line, ih), (255, 0, 255), 2)
#        cv2.line(img, (hor_middle_line_position, 0), (hor_middle_line_position, ih), (255, 0, 255), 2)
#        cv2.line(img, (hor_up_line_position, 0), (hor_up_line_position, ih), (0, 0, 255), 1)
#        cv2.line(img, (hor_down_line_position, 0), (hor_down_line_position, ih), (0, 0, 255), 1)
        # Write status image
        cv2.imwrite(inputfile.split('.')[0] +'_status.jpg',img)
        # Show the video frames output
        if outputfile.upper() == "SHOW":    
            cv2.imshow('Output', img)

### write interim counter csv file

        with open(inputfile.split('.')[0] + "_status.csv", 'w') as f_status:
            cwriter = csv.writer(f_status)
            cwriter.writerow(['Analysis progress: ' + str(round(((frame_number/total_frame_count)*time_length)/fps,2)),'Frames completed: ' + str(frame_number)])
            cwriter.writerow(['Direction', 'Car/Van/Jeep/LCV', 'Mini Bus', 'Bus/Truck (2 Axle)', '3 Axle/EME/JCB', 'Multi Axle'])
            cwriter.writerow(['UP', up_list[0]+up_list[11], up_list[8], up_list[2]+up_list[3]+up_list[6], up_list[9], up_list[5]])
            cwriter.writerow(['DOWN', down_list[0]+down_list[11], down_list[8], down_list[2]+down_list[3]+down_list[6], down_list[9], down_list[5]])
            cwriter.writerow(classNames)
            cwriter.writerow(up_list)
            cwriter.writerow(down_list)
        f_status.close()

        frame_number = frame_number + 1
        
        # write csv file when gracefully completed
        if frame_number > total_frame_count-1:
            td = datetime.now() - analysis_start_time
            print('Analysis completed in ' + str(td))
            #print(f" * * * Counting completed in {td:.03f}")
            print (' *****...***  Analysis complete  ***...*****')
            with open(inputfile.split('.')[0] + "_data.csv", 'w') as f1:
                cwriter = csv.writer(f1)
                cwriter.writerow(['Analysis completed: ' + str(round(((frame_number/total_frame_count)*time_length)/fps,2))])
                cwriter.writerow(['Direction', 'Car/Van/Jeep/LCV', 'Mini Bus', 'Bus/Truck (2 Axle)', '3 Axle/EME/JCB', 'Multi Axle'])
                cwriter.writerow(['UP', up_list[0]+up_list[11], up_list[8], up_list[2]+up_list[3]+up_list[6], up_list[9], up_list[5]])
                cwriter.writerow(['DOWN', down_list[0]+down_list[11], down_list[8], down_list[2]+down_list[3]+down_list[6], down_list[9], down_list[5]])
                cwriter.writerow(classNames)
                cwriter.writerow(up_list)
                cwriter.writerow(down_list)
            f1.close()
            cv2.putText(img, 'Analysis Completed . . .', (int(iw*0.5), int(ih*0.5)), cv2.FONT_HERSHEY_SIMPLEX, font_size*1.25, (200,245,18), font_thickness*1)
            cv2.imwrite(inputfile.split('.')[0] +'_status.jpg',img)        
            return
        else: 
            for frameNum in range(framesGap):
                success, img1 = cap.read()
                #        total_frame_count
                frame_number = frame_number + 1 
                if frame_number == total_frame_count:
                    td = datetime.now() - analysis_start_time
                    print('Analysis completed in ' + str(td))
                    #print(f" * * * Counting completed in {td:.03f}")
                    print (' *****...***  Analysis complete  ***...*****')
                    with open(inputfile.split('.')[0] + "_data.csv", 'w') as f1:
                        cwriter = csv.writer(f1)
                        cwriter.writerow(['Analysis completed: ' + str(round(((frame_number/total_frame_count)*time_length)/fps,2))])
                        cwriter.writerow(['Direction', 'Car/Van/Jeep/LCV', 'Mini Bus', 'Bus/Truck (2 Axle)', '3 Axle/EME/JCB', 'Multi Axle'])
                        cwriter.writerow(['UP', up_list[0]+up_list[11], up_list[8], up_list[2]+up_list[3]+up_list[6], up_list[9], up_list[5]])
                        cwriter.writerow(['DOWN', down_list[0]+down_list[11], down_list[8], down_list[2]+down_list[3]+down_list[6], down_list[9], down_list[5]])
                        cwriter.writerow(classNames)
                        cwriter.writerow(up_list)
                        cwriter.writerow(down_list)
                    f1.close()
                    cv2.putText(img, 'Analysis Completed . . .', (int(iw*0.5), int(ih*0.5)), cv2.FONT_HERSHEY_SIMPLEX, font_size*1.25, (200,245,18), font_thickness*1)
                    cv2.imwrite(inputfile.split('.')[0] +'_status.jpg',img)
                    break #return

        # write csv file when manually stopped
        if cv2.waitKey(1) & 0xFF == ord('q'):
            td = datetime.now() - analysis_start_time
            print('Analysis completed in ' + str(td))
            #print(f" * * * Counting completed in {td:.03f}")
            print (' *****...***  Analysis stopped  ***...*****')
            with open(inputfile.split('.')[0] + "_data.csv", 'w') as f1:
                cwriter = csv.writer(f1)
                cwriter.writerow(['Analysis completed: ' + str(round(((frame_number/total_frame_count)*time_length)/fps,2))])
                cwriter.writerow(['Direction', 'Car/Van/Jeep/LCV', 'Mini Bus', 'Bus/Truck (2 Axle)', '3 Axle/EME/JCB', 'Multi Axle'])
                cwriter.writerow(['UP', up_list[0]+up_list[11], up_list[8], up_list[2]+up_list[3]+up_list[6], up_list[9], up_list[5]])
                cwriter.writerow(['DOWN', down_list[0]+down_list[11], down_list[8], down_list[2]+down_list[3]+down_list[6], down_list[9], down_list[5]])
                cwriter.writerow(classNames)
                cwriter.writerow(up_list)
                cwriter.writerow(down_list)
            f1.close()
            break

        if frame_number == total_frame_count:
            td = datetime.now() - analysis_start_time
            print('Analysis completed in ' + str(td))
            #print(f" * * * Counting completed in {td:.03f}")
            print (' *****...***  Analysis stopped  ***...*****')
            with open(inputfile.split('.')[0] + "_data.csv", 'w') as f1:
                cwriter = csv.writer(f1)
                cwriter.writerow(['Analysis completed: ' + str(round(((frame_number/total_frame_count)*time_length)/fps,2))])
                cwriter.writerow(['Direction', 'Car/Van/Jeep/LCV', 'Mini Bus', 'Bus/Truck (2 Axle)', '3 Axle/EME/JCB', 'Multi Axle'])
                cwriter.writerow(['UP', up_list[0]+up_list[11], up_list[8], up_list[2]+up_list[3]+up_list[6], up_list[9], up_list[5]])
                cwriter.writerow(['DOWN', down_list[0]+down_list[11], down_list[8], down_list[2]+down_list[3]+down_list[6], down_list[9], down_list[5]])
                cwriter.writerow(classNames)
                cwriter.writerow(up_list)
                cwriter.writerow(down_list)
            f1.close()
            cv2.putText(img, 'Analysis Completed . . .', (int(iw*0.5), int(ih*0.5)), cv2.FONT_HERSHEY_SIMPLEX, font_size*1.25, (200,245,18), font_thickness*1)
            cv2.imwrite(inputfile.split('.')[0] +'_status.jpg',img)
            break

    # Write the vehicle counting information in a file and save it
    
    td = datetime.now() - analysis_start_time
    print('Analysis completed in ' + str(td))
    #print(f" * * * Counting completed in {td:.03f}")
    print (' *****...***  Analysis complete  ***...*****')
    with open(inputfile.split('.')[0] + "_data.csv", 'w') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Analysis completed: ' + str(round(((frame_number/total_frame_count)*time_length)/20,2))])
        cwriter.writerow(['Direction', 'Car/Van/Jeep/LCV', 'Mini Bus', 'Bus/Truck (2 Axle)', '3 Axle/EME/JCB', 'Multi Axle'])
        cwriter.writerow(['UP', up_list[0]+up_list[11], up_list[8], up_list[2]+up_list[3]+up_list[6], up_list[9], up_list[5]])
        cwriter.writerow(['DOWN', down_list[0]+down_list[11], down_list[8], down_list[2]+down_list[3]+down_list[6], down_list[9], down_list[5]])
        cwriter.writerow(classNames)
        cwriter.writerow(up_list)
        cwriter.writerow(down_list)
    f1.close()
    # print("Data saved at 'data.csv'")
    # Finally realese the capture object and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    realTime()
    #from_static_image(image_file)
