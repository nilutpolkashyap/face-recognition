# STEP 1 : CREATING THE DATASET
# ---The user is given a unique ID (For eg: System ID, Roll number, etc.)
# ---This ID is uploaded beforehand to reference the name of the user
# ---This part of the code detects the face of the and initiates it to be used later, by creating a dataset
# ---Only the user whose face is to be recognised should be allowed to appear before the camera during this stage

import cv2

fd=cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")
a = cv2.VideoCapture(0)
name=int(input("Enter your Number: "))
r=0
while True:
	z, b = a.read()
	if z == True:
		b = cv2.flip(b,1)
	g=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
	face=fd.detectMultiScale(g,2,5)
	for(x,y,w,h) in face:
		r=r+1
		cv2.imwrite("Data\\Face."+str(name)+"."+str(r)+'.jpg',g[y:y+h,x:x+w])
		cv2.rectangle(b,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.waitKey(100)
	cv2.imshow("LOOK INTO THE CAMERA PLEASE!!",b)
	cv2.waitKey(1)
	if r==25:	
		break
	if cv2.waitKey(1) & 0xFF ==ord("q"):
		break
cv2.destroyAllWindows()
a.release()