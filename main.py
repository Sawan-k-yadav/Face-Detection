import cv2

# Using Harcascade model from offial opencv model zoo for face detect 
harcascade = "model/haarcascade_frontalface_default.xml"

#Drone height and width
drone_width = 1
drone_height = 0.5

cap = cv2.VideoCapture(0) # Here keeping 0 for defaul camera but if we have external camera added 
                          # then we can keep 1

#Setting camera window

cap.set(3, 640)  # Width. Here 3 specifies width property of opencv for captured video
cap.set(4, 480)  # Height. Here 4 specifies width property of opencv for captured video

while True:
    success, img = cap.read() # It will read video frame and its status for success or not

    facecascade = cv2.CascadeClassifier(harcascade) # Loading the model which we have defined

    # As this harcascade model only accepts gray scale image and videos hence need to convert the
    # video frame to gray scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # Now we need to apply this gray scale video to harcascade model
    face  = facecascade.detectMultiScale(img_gray, 1.1, 4) #model will detect all the landmarks of face 
                                                           #and fetch 4 coordinates for bounding boxes

    #After getting those 4 coordinates we can loop through it in video and detect the image
    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) # Taking video, position, color and thickness

        object_height = (y+h) - y
        object_width = (x+w) - x

        # Calculate distance using object size in the frame and actual dimensions
        distance = (object_height * drone_height) / drone_height

        cv2.putText(img, f"Estimated Distance: {int(distance)}m", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Variables for speed estimation
        past_distances = []  # Stores distances from previous frames
        past_frames = 5  # Number of frames to consider for speed estimation
        fps = 30
        # frame_count = 0

        def estimate_speed(current_distance, frame_count, fps):
            if len(past_distances) == frame_count:
                # Calculate speed based on average distance change in past frames
                total_distance_change = current_distance - past_distances[0]
                time_elapsed = frame_count / fps
                speed_estimate = total_distance_change / time_elapsed
                return speed_estimate
            else:
                past_distances.append(current_distance)
                return None  # Not enough data for reliable speed estimation yet

        # Update past_distances list and frame count for next iteration
        frame_count = past_frames + 1
        speed = estimate_speed(distance, frame_count, fps)
        if speed:
            cv2.putText(img, f"Speed: {speed:.2f} m/s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        
        

    # To open camera. It need camera window name and video
    cv2.imshow("Face", img)

    # To close camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



