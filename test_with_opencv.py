import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#load our trained CNN model 

model = load_model('face_detect_model.h5')


#model accept 200x200 height and width and below 
img_width, img_height = 200, 200

#load cascade facew classifier 
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


#start webcam

cap = cv2.VideoCapture(0)  #using webcam


img_count_full = 0

#font 
font  = cv2.FONT_HERSHEY_SIMPLEX

#org
org = (1, 1)
class_lable =' '
color = (0, 255, 0)


while True:
    img_count_full +=1
    
    #capture from webcam reeceiving frames
    ret, frame = cap.read()
    
    
    if ret == False:
        break
    
#resize image with 50 % ratio

    scale = 50
    width = int(frame.shape[1] * scale / 100)
    height = int(frame.shape[0] * scale / 100)
    dims = (width, height)
    
    frame = cv2.resize(frame, dims, interpolation = cv2.INTER_AREA)
    
    
    #convert image to gray_scale 
    
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Detect the faces 
    
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 6)
    
    #take face and predict class and draw rectabgle on face 
    
    img_count = 0
    for (x,y ,w, h) in faces:
        org = (x-10, y-10)
        img_count +=1
        
        color_face = frame[y:y+h, x:x+w]
        cv2.imwrite('faces/input/%d%dface1.png'%(img_count_full, img_count), color_face)
        img = load_img('faces/input/%d%dface1.png'%(img_count_full, img_count), target_size = (img_width, img_height) )
        
        img = img_to_array(img)/255
        img = np.expand_dims(img, axis = 0)
        
        #prediction probability , should be between 0-1  0 for Mask and 1 for not weared
        pred_prob = model.predict(img)
        pred = np.argmax(pred_prob)
        
        
        if pred == 0:
            print("Mask weared", pred_prob[0][0])
            class_lable = "Mask"
            color = (0, 255, 0)
            cv2.imwrite('faces/with_mask/%d%dface1.png'%(img_count_full, img_count), color_face)
            
        else:
            print("Mask not weared", pred_prob[0][1])
            class_lable = "No Mask"
            color = (0, 255 ,0)
            cv2.imwrite('faces/without_mask/%d%dface1.png'%(img_count_full, img_count), color_face)
            
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, class_lable, org, font, 1,(0, 255, 0), 2,  cv2.LINE_AA )
                    
    # display image 
    
        cv2.imshow('Face Mask Detection',frame )
    
      
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
cap.release()
cv2.destroyAllWindows()