#Author: Alex McLeod
from pyrobot import Robot
from pyrobot.utils.util import try_cv2_import
import numpy as np
import tensorflow as tf
import math

#purpose: draw the bounding boxes for each image
#IMPORTS:
#	imageOrig: the original image
#	boxes: array containing coordinates of each detected box
#       isObjectClose: array containing boolean values as to whether detected object is close 
def drawBoundingBoxes(imageOrig, boxes, isObjectClose):
    imageHeight = imageOrig.shape[0]
    imageWidth = imageOrig.shape[1]
    count = 0
    #for each detect object draw the bounding box
    for box in boxes:
        xcoord1 = (box[1])
        ycoord1 = (box[2])
        point1 = (int(xcoord1*imageWidth), int(ycoord1*imageHeight))#scale to fit image

        xcoord2 = (box[3])
        ycoord2 = (box[0])
        point2 = (int(xcoord2*imageWidth), int(ycoord2*imageHeight))#scale to fit image

        if(isObjectClose[count] == False):#if object is not close then set rectangle colour to green
            cv2.rectangle(imageOrig, point1, point2, (0, 255, 0), 2)
        else:# if object is close then set rectangle colour to red
            cv2.rectangle(imageOrig, point1, point2, (0, 0, 255), 2)
        count = count + 1
    return imageOrig
        

#purpose: module for rotating the robot 90 degrees
def beginRotate():

    print("Rotating!")
    linear_velocity = 0.0
    rotational_velocity = -2
    execution_time = 3
    bot.base.set_vel(fwd_speed=linear_velocity,
                     turn_speed=rotational_velocity,
                     exe_time=execution_time)  
    bot.base.stop()

#purpose: module for moving the robot forward slightly
def forward():
    print("Moving Forward!")
    linear_velocity = 0.6
    rotational_velocity = 0.0
    execution_time = 1
    bot.base.set_vel(fwd_speed=linear_velocity,
                     turn_speed=rotational_velocity,
                     exe_time=execution_time)
    bot.base.stop()

#purpose: module for checking the distance away that a point is from the robots camera
def calcDistance(x, y, z):
    distance = math.pow((math.pow(x, 2) + math.pow(y, 2) + math.pow(z, 2)), 0.5)
    return distance
    

# purpose: module for check the the distance of detected objects
#IMPORTS:
#	boxes: array containing coordinates of each detected box
#	img: the original image
#	bot 
def checkObjectDistance(boxes, img):
    rotate = False
    isObjectClose = []
    boxIndex = 0

    # for each detected object check the distance from the robot           
    for box in boxes:
        print("Box: ", boxIndex)
        #starting default value
        closestDist = -1
        ycoord = box[0] #top most y coord of box
        xcoord = box[1] #left most x coord of box
        bottom = box[2] #bottom most y coord of box
        right = box[3] #right most x coord of box
        # check if we have reached the bottom of the box
        while ycoord <= bottom:
            xcoord = box[1] # reset to the left most x position
            #check if we have reach the right edge of the box
            while xcoord <= right:
                
                scaledX = int(round(640 * xcoord, 0))
                scaledY = int(round(480 * ycoord, 0))
                cv2.circle(img, (scaledX, scaledY), 1, (0, 255, 0), -1) # draw circle at x, y coordinate
                if((scaledX < 640 and scaledX > 0) and (scaledY < 480 and scaledY > 0)):
                    pt, colour = bot.camera.pix_to_3dpt(scaledY, scaledX, in_cam=True) # get 3 Dimensional point
                    #Check that x, y and z values are not NaN
                    if((math.isnan(pt[0][0]) == False) and (math.isnan(pt[0][1]) == False) and (math.isnan(pt[0][2]) == False)):
                        currentDist = calcDistance(pt[0][0], pt[0][1], pt[0][2]) # calc distance away using 3-D point
                        #Check if any previous point of the object is closer if not set as the closest point
                        if (closestDist == -1) or (currentDist < closestDist): 
                            closestDist = currentDist
                xcoord = xcoord + 0.02 # increase x value to get next point
            ycoord = ycoord + 0.02 # increase y value to get next point
   
        print("Point of Closest Distance: ", closestDist) # print the closest point of the object to the terminal
                
        if (closestDist < 0.0018 and closestDist > 0): # if object is close set rotate to true
            rotate = True 
            isObjectClose.append(True)
        else: # if object is not close then set rotate to false
            isObjectClose.append(False)
        boxIndex = boxIndex + 1
    return rotate, isObjectClose    
                                                               
              
        
# purpose: takes an input model and an image extracting information on objects
#          detected in the image
def run_inference_for_single_image(model, image):
    #convert the image into a numpy array
    image = np.asarray(image)

    # Get input and output tensors.
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']

    # convert to uint8
    input_data = np.array(image, dtype=np.uint8)

    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_data = input_data[tf.newaxis,...]

    # set the input data of the model
    model.set_tensor(input_details[0]['index'], input_data)

    # run model on input
    model.invoke()

    # The function `get_tensor()' is used to get bounding boxes, labels and certainty
    # of objects detected within the image
    output_boxes = model.get_tensor(output_details[0]['index'])
    output_labels = model.get_tensor(output_details[1]['index'])
    output_certainty = model.get_tensor(output_details[2]['index'])

    # return the output from the model
    return output_boxes[0], output_labels[0], output_certainty[0]

#purpose: make a prediction for an image given a specific path
def show_inference(model, imageOrig):
    # set rotate to initially false
    rotate = False

    #load labels into a list
    labelList = [] 
    # open the file containing coco dataset labels 
    f = open("cocoMobilenet/labelMap.txt", "r")
    # create a list containing the labels names
    for labelName in f:
        labelList.append(labelName)
    f.close() 
    
    # switch original image to RGB
    imageOrig = cv2.cvtColor(imageOrig, cv2.COLOR_BGR2RGB)

    # resize the image to fit the model
    imageResized = cv2.resize(imageOrig, (300, 300))
    imageResized = cv2.cvtColor(imageResized, cv2.COLOR_RGB2BGR)
    image_np = np.array(imageResized);
    # input the image into the model to get data
    outputBoxes, outputLabels, outputCertainty = run_inference_for_single_image(model, image_np)

    #count the number of boxes 
    numBoxes = 0
    outputCount = 0
    detections = np.zeros((2, 10))
    boxes = [] 
    predictions = []
    percentageCertaintys = []

    # for each detected object in the image if the predicted accuracy is above %55 the append it to
    # the list of boxes
    for certainty in outputCertainty:
        if certainty > 0.55:
            numBoxes = numBoxes + 1
            percentageCertaintys.append(round(certainty*100, 2))
            boxes.append(outputBoxes[outputCount])
            predictions.append(int(outputLabels[outputCount] + 1))
        outputCount = outputCount + 1

    print("numbox: " + str(numBoxes))

    # output the predictions to the terminal
    for predictionNum in predictions:
        print(predictionNum)
        prediction = labelList[predictionNum]
        print('Prediction: ' + prediction)
        print(boxes[0])
    
    # if an object an object is detected place bounding box on image
    if numBoxes > 0:      
        colours = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

        imageOrig = tf.expand_dims(imageOrig, 0)
   
        imageOrig = tf.image.convert_image_dtype(imageOrig, dtype=tf.float32, saturate=False)
    	boxes = tf.expand_dims(boxes, 0)

    	boxes = np.asarray(boxes)

        # call draw_bounding_boxes to draw bounding boxes on the image
    	#imageOrig = tf.image.draw_bounding_boxes(imageOrig, boxes, colours)

        # reduce the dimensions of the arrays
    	imageOrig = np.squeeze(imageOrig, axis=0)
    	boxes = np.squeeze(boxes, axis=0)
   
        # get the position to place text above bounding boxes
    	#imgHeight = imageOrig.shape[0]
    	#imgWidth = imageOrig.shape[1]
    	#nextTextPos = (int(0.3*imgWidth), int(0.95*imgHeight))

        # get height and width of image to position the text
        imgHeight = imageOrig.shape[0]
        imgWidth = imageOrig.shape[1]

        rotate, isObjectClose = checkObjectDistance(boxes, imageOrig)

        imageOrig = drawBoundingBoxes(imageOrig, boxes, isObjectClose)

    	#add labels with percentages above bounding boxes
    	count = 0
    	for predictionNum in predictions:
            prediction = labelList[predictionNum]
            box = boxes[count]
            xcoord = box[1]
            ycoord = box[0]
            textPosition = (int(xcoord*imgWidth), int(ycoord*imgHeight) - 5)
            stringLen = len(prediction)
            predictionText = prediction[0:stringLen - 1] + ":" + str(percentageCertaintys[count])
            if(isObjectClose[count] == False):
                cv2.putText(imageOrig, predictionText, textPosition, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 300, 0, 255), 1)
            else:
                cv2.putText(imageOrig, predictionText, textPosition, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 300, 255), 1)
            count = count + 1
 
        # position of text that displays whether there has been detections
        textPos1 = (int(0.2*imgWidth), int(0.95*imgHeight))
        textPos2 = (int(0.1*imgWidth), int(0.95*imgHeight))
        if (rotate == False):
            defaultText = "OBJECTS DETECTED"
            cv2.putText(imageOrig, defaultText, textPos1, cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 300, 0, 255), 2)
        else:
            rotationText = "OBJECTS CLOSE, ROTATING!"
            cv2.putText(imageOrig, rotationText, textPos2, cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 300, 255), 2) 

        #check how far away each object is
        #rotate = checkObjectDistance(boxes, imageOrig, bot)

    #display the image to the user
    cv2.imshow('object_detection', imageOrig)
    cv2.waitKey(1000)
    cv2.namedWindow('object_detection')
    return rotate

# purpose: module that gets snapshots and runs tensorflite module on them
def snapshot():
    # Load the cocoMobilenet TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="cocoMobilenet/detect.tflite")
    interpreter.allocate_tensors()

    bot.camera.reset()
    pan = 0.0
    bot.camera.set_pan(pan, wait=True)
    tilt = -0.4
    bot.camera.set_tilt(tilt, wait=True)
    # while the user hasnt pressed q keep getting snapshots from the robots camera
    while True:
        


        rgb = bot.camera.get_rgb() 
        rotate = show_inference(interpreter, rgb)

        # if there are objects in the way then rotate
        if (rotate == True):
            beginRotate()
        # otherwise no objects in the way move forward
        else:
            forward()

        # if user presses q then end simulation
        if cv2.waitKey(1) == ord('q'):
            break


# initialise the robot
bot = Robot('locobot')
# initialise the camera
cv2 = try_cv2_import()
# get snapshots
snapshot()

