
import tensorflow as tf

# use the model with a USB camera stream
import cv2

# load the saved model
model = tf.keras.models.load_model('my_model.h5')

# set up the camera
cap = cv2.VideoCapture(0)
while True:
    # read frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame from camera")
        break

    # preprocess the frame
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame = tf.expand_dims(frame, 0)

    # use the model to predict the class of the frame
    prediction = model.predict(frame)
    if prediction > 0.5:
        label = "dog"
    else:
        label = "cat"

    # display the label on the frame
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("frame", frame)

    # check for exit key
    if cv2.waitKey(1) == ord('q'):
        break

# release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
