import cv2
import cv2
def crop(video_name):
    vidcap = cv2.VideoCapture(video_name)
    success,image = vidcap.read()
    count = 0
    count2 = 0
    success = True
    while success:
        success,image = vidcap.read()
        if count%10 == 0:
            cv2.imwrite("./frame/" + video_name.split('.')[0] + "_%d.jpg" % count2, image)     # save frame as JPEG file
            # if cv2.waitKey(10) == 27:                     # exit if Escape is hit
            #     break
            count2 += 1
        count += 1

crop("ngatu_1.mp4")