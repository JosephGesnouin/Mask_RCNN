import cv2
import numpy as np
import os
from datetime import datetime
from visualize_cv2 import model, display_instances, class_names
import skimage.io
startTime1 = datetime.now()
ROOT_DIR = os.getcwd()
capture = cv2.VideoCapture(os.path.join(ROOT_DIR,'../../../echangeBalle.mov'))
size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('videofile_masked.avi', codec, 60.0, size)

"""while(capture.isOpened()):
    ret, frame = capture.read()
    if ret:
        # add mask to frame
        startTime = datetime.now()
        results = model.detect([frame], verbose=0)
        print(datetime.now() - startTime)
        r = results[0]
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, None)
        output.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
output.release()
cv2.destroyAllWindows()
print(datetime.now() - startTime1)"""
images = []
while(capture.isOpened()):
    ret, frame = capture.read()
    if ret:
        # add mask to frame
        images.append(frame)
        if len(images) == 32:
            startTime = datetime.now()
            results = model.detect(images, verbose=1)
            print("il a fallu ", (datetime.now() - startTime) , "  secondes soit " ,(datetime.now() - startTime)/32 , " par image")
            for i,item in enumerate(zip(images,results)):
                image = item[0] 
                r = item[1]
                frame = display_instances(
                        image, r['rois'], r['masks'], r['class_ids'], class_names, None)
                output.write(frame)
                #cv2.imshow('frame', frame)
            images = []

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
output.release()
cv2.destroyAllWindows()
print(datetime.now() - startTime1)