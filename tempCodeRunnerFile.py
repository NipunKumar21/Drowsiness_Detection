import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# Initialize the Pygame mixer and load the sound file for the alarm
mixer.init()
sound = mixer.Sound('waku-waku.wav.wav')

# Load Haar cascade classifiers for face, left eye, and right eye detection
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

# Define labels for eye states (close or open)
lbl=['Close','Open']

# Load the pre-trained CNN model
model = load_model('models/cnncat2.h5')

path = os.getcwd()

# Configure the video capture using the default camera/webcam
cap = cv2.VideoCapture(0)

# Set font for displaying text on the frame
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Initializing variables for counting and scoring
count=0
score=0

# Set thickness for drawing rectangles
thicc=2

# Initialize previous eye state predictions
rpred=[99]
lpred=[99]

# Start the main detection loop using a while loop
while(True):
    # Read frames from the video capture
    ret, frame = cap.read()
    
    # Get the height and width of the frame
    height,width = frame.shape[:2] 

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using the Haar cascade for faces
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    
    # Detect left eyes using the Haar cascade for left eyes
    left_eye = leye.detectMultiScale(gray)
    
    # Detect right eyes using the Haar cascade for right eyes
    right_eye = reye.detectMultiScale(gray)

    # Draw a filled rectangle at the bottom of the frame for showing  score txt
    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    # Loop for detecting faces
    for (x,y,w,h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (100,100,100), 1)

    # Loop for detecting  right eyes closed or open
    for (x,y,w,h) in right_eye:

        r_eye=frame[y:y+h,x:x+w]
        
        # Increment the eye counter
        count=count+1
        
        # Convert the right eye to grayscale
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        
        r_eye = cv2.resize(r_eye,(24,24))
        
        # Normalize the right eye image
        r_eye = r_eye/255
        
        r_eye = r_eye.reshape(24,24,-1)
        
        r_eye = np.expand_dims(r_eye, axis=0)
        
        # Make predictions for the right eye using the pre-trained model
        rpred = np.argmax(model.predict(r_eye), axis=-1)
        
        # Determine the label (open or closed) based on the prediction
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        
        # Break the loop after processing the first right eye 
        break

    # Loop for  detection of left eye
    for (x,y,w,h) in left_eye:
        # Extract the region of interest (left eye)
        l_eye=frame[y:y+h,x:x+w]
        
        # Increment the eye counter
        count=count+1
        
        # Convert the left eye to grayscale
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        
        l_eye = cv2.resize(l_eye,(24,24))
        
        # Normalize the left eye image
        l_eye = l_eye/255
        
        l_eye = l_eye.reshape(24,24,-1)
        
        l_eye = np.expand_dims(l_eye, axis=0)
        
        # Make predictions for the left eye using the pre-trained model
        lpred = np.argmax(model.predict(l_eye), axis=-1)
        
        # Determine the label open or closed
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        
        # Break the loop after processing the first left eye (assuming only one face is detected)
        break

    # Update the score based on the eye states
    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame, "Closed", (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
    else:
        score=score-1
        cv2.putText(frame, "Open", (width-100, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    # Make sure the score is not negative KYUKI AGAR NEGATIVE GAYA TOH VO -VE ME INCREAase hota jayega aur jo alarm throsehold hit nahi kargea
    if(score<0):
        score=0
    
    # Display the counter and score on the frame
    cv2.putText(frame,'counter:'+str(score),(25,height-20), cv2.FONT_HERSHEY_TRIPLEX, 1,(0,255,255),1,cv2.LINE_AA)
    
    # Trigger the alarm if the score becomes greater than the value
    if(score>20):
        # Save the frame as an image
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        
        try:
            # Play the alarm sound
            sound.play()
        except:
            pass
        
       
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        
        # Draw a rectangle around the face to highlight drowsiness
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 1 ) 

    # Load a background image
    background = cv2.imread('background.jpeg')
    
    # Resize the background image to match the frame size
    resized = cv2.resize(background, (width, height), interpolation = cv2.INTER_AREA)
    
    # Overlay the frame and the background image
    frame = cv2.addWeighted(frame, 0.5, resized, 0.5, 0)
    
    # Display the frame with overlays
    cv2.imshow('frame',frame)
    
    # Check for the 'q' key press to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
