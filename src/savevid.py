import numpy as np
import cv2

cap1 = cv2.VideoCapture(1)
cap1.set(3,1280)
cap1.set(4,960)
cap2 = cv2.VideoCapture(2)
cap2.set(3,1280)
cap2.set(4,960)
# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output1.avi',fourcc, 30.0, (1280,960), True)
out2 = cv2.VideoWriter('output2.avi', fourcc, 30.0, (1280,960), True)
while(cap1.isOpened()):
    ret, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if ret==True:
        print(frame1.size)
        # write the flipped frame
        out.write(frame1)
        out2.write(frame2)

        cv2.imshow('frame1',frame1)
        cv2.imshow('frame2', frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap1.release()
#cap2.release()
out.release()
#out2.realease()
cv2.destroyAllWindows()
