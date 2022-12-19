# import required libraries
import cv2
import numpy as np
import tensorflow as tf

def mask():
    # Download the model created in train.py
    model = tf.keras.models.load_model('mask.h5')
    # using "haar cascade" for face detection
    haar = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')
    # upload the video
    cap = cv2.VideoCapture('clip1.mp4')
    font = cv2.FONT_HERSHEY_COMPLEX
    while cap.isOpened():
        flag, img = cap.read()
        # If the frame is read correctly, the flag is True
        if not flag:
            print("no frame, exit...")
            break
        # make each image gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # face detection
        faces = haar.detectMultiScale(gray, 1.1, 4)

        for x, y, w, h in faces:
            # crop the face in the picture
            img1 = img[y:y + h, x:x + w]
            # resize the image to 100x100
            img1 = cv2.resize(img1, (100, 100))
            # image prediction
            result = model.predict(np.expand_dims(img1, 0))[0]
            # if the result is greater than 0.5, we mark "mask".
            text = "no mask" if result < 0.5 else "mask"
            # if the result is greater than 0.5, we draw a green rectangle, otherwise, red
            color = (0, 0, 255) if result < 0.5 else (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # write with a mask or not
            cv2.putText(img, text, (x, y), font, 1, (255, 0, 255), 2)
        # show the image
        cv2.imshow("frame", img)
        # if the "q" button is pressed, the video stops
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    mask()


