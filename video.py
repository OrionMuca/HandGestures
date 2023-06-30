import numpy as np
import cv2

# Open a sample video available in sample-videos
vcap = cv2.VideoCapture('http://192.168.0.227:4747/', cv2.CAP_FFMPEG)

while True:
    # Capture frame-by-frame
    ret, frame = vcap.read()

    if frame is not None:
        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Press 'q' to close the video window
        if cv2.waitKey(22) & 0xFF == ord('q'):
            break
    else:
        print("Frame is None")
        break

# When everything is done, release the capture
vcap.release()
cv2.destroyAllWindows()
print("Video stopped")
