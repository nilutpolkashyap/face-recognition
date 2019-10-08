# STEP 2 : TRAINING THE CODE AND FORMATTING THE DATASET
# ---The code reads the dataset and the unique ID from the folder where the  dataset is uploaded in STEP 1
# ---The dataset is then converted into a .yml format file which OpenCV reads

import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = r"Data"

def img_ids(path):
	ImgPaths = [os.path.join(path,f) for f in os.listdir(path)]
	faces=[]
	IDs=[]
	for ImgPath in ImgPaths:
		face_Img = Image.open(ImgPath).convert("L")
		face_ar = np.array(face_Img,'uint8')  
		ID = os.path.split(ImgPath)[-1].split(".")[1]
		ID = int(ID)
		faces.append(face_ar)
		IDs.append(ID)
		cv2.waitKey(10)
	return np.array(IDs), faces	
IDs,faces = img_ids(path)
recognizer.train(faces, np.array(IDs))		
recognizer.save(r"DATA.yml")
cv2.destroyAllWindows()