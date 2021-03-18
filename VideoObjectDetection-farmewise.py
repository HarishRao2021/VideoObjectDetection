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
    net.setInputSize(320,320)
    net.setInputScale(1/127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    conf_req = 0.60  ## Define required Minimum Confidence Level

    # Path to video file
    cap = cv2.VideoCapture(path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #net.setInputSize(width, height)

    # Used as counter variable
    count_frame = 0 #Counter to skip frames


    # checks whether frames were extracted
    success = 1

    while success:
        # vidObj object calls read
        # function extract frames
        success, img = cap.read()

        # Resizing to 25% of original ,
        # because the input video size was very large in this particular instance (3000 x 2500 pixels)
        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

        count = 1  # Counter to count length of classids
        # Saves the frames with frame-count
        #cv2.imwrite("output/frame%d.jpg" % count, img)

        print("Executing While Loop... in frame no: ",count_frame+1)
        if (count_frame % 40 == 0) :
            cv2.imshow("Pre-image", img)
            print('actual h & w = ',width,height)
            # EXECUTE OBJECT DETECTION ALGORITHM

            classIds, confs, bbox = net.detect(img, confThreshold=conf_req)
            print("length of classIds = ",len(classIds))

            # Draw a rectangle around the object detected
            if len(classIds) != 0:
                for eachclassID, eachConfidence, eachBbox in zip(classIds.flatten(), confs.flatten(), bbox):
                    cv2.rectangle(img, eachBbox, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[eachclassID - 1], (eachBbox[0] - 10, eachBbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    #cv2.imshow("image", img)

                    #plt.imshow(img1)
                    #plt.show()
                    #cv2.imwrite("output/frame%d.jpg" % count, img)
                    print(count, classNames[eachclassID - 1],eachConfidence,"X=",eachBbox[0],"Y=",eachBbox[1])
                    print()
                    count += 1

                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print("No objects found, len of class IDs = ZERO")

            #time.sleep(5)
            cv2.imshow("Post-image", img)
            cv2.imwrite("output/frame%d.jpg" % (count_frame+1), img)
            print("Parsed Frame no: ", count_frame+1)
            count_frame += 1

        else:
            print("Skipped Frame no: ", count_frame+1)
            count_frame += 1
            cv2.waitKey(0)
            cv2.destroyAllWindows()







print("Start of Execution")
FrameCapture('mobile phone.mp4')
print("End of Execution")

