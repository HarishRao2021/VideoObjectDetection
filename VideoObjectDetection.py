import cv2
import time
## COOOOOOL ... now let's run this on a live video

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'



net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(640, 480)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


    while True:
    success, img = cap.read()
    print("Success is:",success)
    classIds, confs, bbox = net.detect(img,confThreshold=0.5)

    if len(classIds) != 0:
        for classID, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classID-1],(box[0],box[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
            print(classNames[classID-1],"--",classID,"--",box[0],",",box[1])


    cv2.imshow("videooutput",img)
    cv2.waitKey(1)
#    cv2.destroyAllWindows()
#    plt.close()
