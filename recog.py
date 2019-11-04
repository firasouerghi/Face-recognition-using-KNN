
# this one is used to recognize the  
# face after training the model with 
# our data stored using knn 
import cv2 
import numpy as np 
import pandas as pd 
  
from npwriter import f_name 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split

  
  
# reading the data 
data = pd.read_csv(f_name).values 
  
# data partition 
X, Y = data[:, 1:-1], data[:, -1] 

# cross Validation process
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=101)  
#print(X, Y) 
  
# Knn function calling with k = 5 
model = KNeighborsClassifier(n_neighbors =2) 
  
# fdtraining of model 
model.fit(X_train,y_train) 
  
cap = cv2.VideoCapture(0) 
  
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
  
f_list = [] 
  
while True: 
  
    ret, frame = cap.read() 
  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
  
    faces = classifier.detectMultiScale(gray, 1.5, 5) 
  
    x_test = [] 
  
    # Testing data 
    for face in faces: 
        x, y, w, h = face 
        im_face = gray[y:y + h, x:x + w] 
        im_face = cv2.resize(im_face, (100, 100)) 
        x_test.append(im_face.reshape(-1)) 
  
    if len(faces)>0: 
        response = model.predict(np.array(x_test)) 
        # prediction of result using knn 
  
        for i, face in enumerate(faces): 
            x, y, w, h = face 
  
            # drawing a rectangle on the detected face 
            cv2.rectangle(frame, (x, y), (x + w, y + h),(255, 0, 0), 3) 
  
            # adding detected/predicted name for the face 
            cv2.putText(frame, response[i], (x-50, y-50),cv2.FONT_HERSHEY_DUPLEX, 2,(0, 255, 0), 3) 
     
    cv2.imshow("full", frame) 
  
    key = cv2.waitKey(1) 
  
    if key & 0xFF == ord("q") : 
        break
  
cap.release() 
cv2.destroyAllWindows() 
