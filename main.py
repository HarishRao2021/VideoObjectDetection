import cv2
from matplotlib import pyplot as plt


img = cv2.imread("10.png")
#img = cv2.resize(img,(640,480))
classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(360, 480)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
conf_req=0.60  ## Define required Minimum Confidence Level

# EXECUTE OBJECT DETECTION ALGORITHM
classIds, confs, bbox = net.detect(img,confThreshold=conf_req)

# PRINT RESULTS
try:
    if len(classIds) !=0:
        print(classIds,confs,bbox)
        print("\n")
        print("Classes Detected: I think its a ... ",end="")
        for i in classIds:
            for j in i:
                print(classNames[j-1]+"!!",end=", ")

        print("\n")
        print("Confidence % = ",end="")
        for i in confs:
            for j in i:
                print(j, end=", ")
        print("\n")
        for i in bbox:
            print("BB = ",i)
    else:
        print("I AM UNSURE"+"  -  Confidence is < ",conf_req)
except:
    print("\nERROR")
    print("Len of classIDs was: ", len(classIds))
    print("j is: ", j)
# SHOW IMAGE IN WINDOW WITH PLOTTING GRIDS via MATPLOTLIB
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img1)
plt.show()

#Draw a rectangle around the object detected
if len(classIds) != 0:
    for eachclassID, eachConfidence, eachBbox in zip(classIds.flatten(),confs.flatten(),bbox):
        cv2.rectangle(img,eachBbox,color=(0,255,0),thickness=2)
        cv2.putText(img,classNames[eachclassID-1],(eachBbox[0]+10,eachBbox[1]+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        cv2.imshow("image", img)
else:
    print("I AM UNSURE"+"  -  Confidence is < ",conf_req)

# CLOSE WINDOW ON KEYPRESS "q"


# RELEASE ALL RESOURCES
cv2.waitKey(0)
cv2.destroyAllWindows()
#plt.close()
