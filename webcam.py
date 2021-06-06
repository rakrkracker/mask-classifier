import cv2
import numpy as np
import tensorflow as tf
import math


def startCameraFeed(model):
    feed = cv2.VideoCapture(0)
    showCameraFeed(feed, model)

    return feed


def releaseCamera(feed):
    # When everything done, release the capture
    feed.release()
    cv2.destroyAllWindows()


def showCameraFeed(feed, model):
    frameRate = feed.get(5)  # frame rate
    frameRate = 30  # temp

    frameNum = 0
    mask_sum = 0
    text = 'Initializing...'
    txtcolor = (0, 255, 0)

    file_index = 0

    while(True):
        # Capture frame-by-frame
        frameId = feed.get(1)  # current frame number
        ret, frame = feed.read()

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Flip image horizontaly (to get mirror image)
        frame = cv2.flip(frame, 1)

        # Resize image to (224, 224, 3)
        img = cv2.resize(frame, (224, 224))
        img = np.expand_dims(img, axis=0)

        # Check if wearing mask
        frameNum += 1

        res = model(img)[0]
        pred = tf.where(res[0] < res[1], 0, 1)
        mask_sum += pred

        if frameNum % math.floor(frameRate / 2) == 0:
            mask_avg = mask_sum / 15
            frameNum = 0
            mask_sum = 0
            text = 'Mask' if mask_avg > 0.5 else 'No mask'
            txtcolor = (0, 0, 255) if mask_avg > 0.5 else (255, 0, 0)

        # Add text
        frame = cv2.putText(
            frame,                      # Image
            text,                       # Text
            (50, 50),                   # Org (origin)
            cv2.FONT_HERSHEY_SIMPLEX,   # Font
            1,                          # Font scale
            txtcolor,                # Color
            2,                          # Thickness
            cv2.LINE_AA                 # Line type
        )

        # Display the resulting frame
        # cv2.imshow('frame', gray)
        # cv2.imwrite('test_image2.png', frame)

        cv2.imshow('frame', frame)
        keypressed = cv2.waitKey(1)
        if keypressed == ord('q'):
            releaseCamera(feed)
            break
        elif keypressed == ord('c'):
            img_name = "images/snapshot_{}.png".format(file_index)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
