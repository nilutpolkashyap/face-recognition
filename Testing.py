#STEP 3 : IDENTIFYING THE FACE BEFORE THE CAMERA
# ---The codes reads the DATA.yml file in which the ID is also included
# ---If the face matches the uploaded Data, the code shows the Name is referenced through the unique ID of the user
# ---If it doesn't the code shows the message of authentication error

import cv2

fd=cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")
a = cv2.VideoCapture(0)
tre = cv2.face.LBPHFaceRecognizer_create()
tre.read(r"DATA.yml")
ID = 0
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
	z, b = a.read()
	if z == True:
		b = cv2.flip(b,1)
	g=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
	face=fd.detectMultiScale(g,2,5)
	cv2.putText(b,"Press 'Q' to quit",(20,460),font,1,(255,255,0),2)
	for(x,y,w,h) in face:
		cv2.rectangle(b,(x,y),(x+w,y+h),(0,255,0),2)
		ID,conf = tre.predict(g[y:y+h,x:x+w])
		if conf>50:
			ID = "Authentication error"
		else:	
			if ID == 2860:
				ID = "Hey its Nilutpol"
			elif ID == 2:
				ID = "Hey its Dr. Pooja Mam"
			else:
				ID = "Authentication error"	
		cv2.putText(b,str(ID),(x+5,y+h-7),font,1,(0,0,255),3)
	cv2.imshow("Facial Recognition IN ACTION",b)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
cv2.destroyAllWindows()
a.release()
