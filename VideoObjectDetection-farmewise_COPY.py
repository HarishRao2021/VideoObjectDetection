import time
import cv2
import time
import cv2
import msvcrt
from matplotlib import pyplot as plt


# Function to extract frames
def FrameCapture(path):
    ## COOOOOOL ... now let's run this on a live video

    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    print(classNames)

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(640, 480)
    net.setInputScale(1 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    conf_req = 0.50  ## Define required Minimum Confidence Level

    # Path to video file
    cap = cv2.VideoCapture(path)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

    # Used as counter variable
    count_frame = 0 #Counter to skip frames


    # checks whether frames were extracted
    success = 1

    while success:
        # vidObj object calls read
        # function extract frames
        success, img = cap.read()
        count = 1  # Counter to count length of classids
        # Saves the frames with frame-count
        #cv2.imwrite("output/frame%d.jpg" % count, img)

        print("Executing While Loop... in frame no: ",count_frame+1)
        if (count_frame % 40 == 0) :

            # EXECUTE OBJECT DETECTION ALGORITHM

            classIds, confs, bbox = net.detect(img, confThreshold=conf_req)
            print("length of classIds = ",len(classIds))

            # Draw a rectangle around the object detected
            if len(classIds) != 0:
                for eachclassID, eachConfidence, eachBbox in zip(classIds.flatten(), confs.flatten(), bbox):
                    cv2.rectangle(img, eachBbox, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[eachclassID - 1], (eachBbox[0] - 10, eachBbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 3)
                    #cv2.imshow("image", img)

                    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #plt.imshow(img1)
                    #plt.show()
                    #cv2.imwrite("output/frame%d.jpg" % count, img)
                    print(count, classNames[eachclassID - 1],eachConfidence,"X=",eachBbox[0],"Y=",eachBbox[1])
                    print()
                    count += 1

                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print("I AM UNSURE" + "  -  Confidence is < ", conf_req)

            #time.sleep(5)
            cv2.imshow("image", img)
            cv2.imwrite("output/frame%d.jpg" % (count_frame+1), img)
            print("Parsed Frame no: ", count_frame+1)
            count_frame += 1

        else:
            print("Skipped Frame no: ", count_frame+1)
            count_frame += 1
            cv2.waitKey(0)
            cv2.destroyAllWindows()







print("Start of Execution")
FrameCapture(0)
print("End of Execution")

