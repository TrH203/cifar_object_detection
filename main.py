import cv2
import numpy as np
import joblib
import time
import csv
# Load your pre-trained SVM model (replace 'your_model.pkl' with your model file)
svm_model = joblib.load('model.pkl')
class_labels = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark",
    "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar",
    "sunflower", "sweet_pepper", "table", "tank", "telephone", "television",
    "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale",
    "willow_tree", "wolf", "woman", "worm"
]
# Create a function to capture video frames and make predictions
def capture_video_and_predict():
    # Open a connection to the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Display the frame in a window
        cv2.imshow('Webcam', frame)

        # Press 'q' to exit the webcam capture loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Resize the frame to match the input size expected by your model (e.g., 32x32)
        resized_frame = cv2.resize(frame, (32, 32))

        # Ensure it has 3 color channels (RGB)
        # if len(resized_frame.shape) == 2:
        # resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)

        # Flatten and reshape the resized frame to match the input format expected by your model
        input_data = resized_frame.reshape(1, -1)

        np.savetxt("visualize.csv", input_data, delimiter=',',fmt="%.0f")
        # Make predictions using the SVM model
        predicted_class = svm_model.predict(input_data)


        # Print or display the predicted class label
        print("Predicted class:", class_labels[predicted_class[0]])
        
        time.sleep(1)
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start capturing video from the webcam and making predictions
capture_video_and_predict()

