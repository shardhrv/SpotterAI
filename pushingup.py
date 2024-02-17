import cv2 
import mediapipe as mp
import numpy as np 
import calculations as cl

mp_drawing = mp.solutions.drawing_utils # this will give us all the drawing utililty ie to visualise all the poses
mp_pose = mp.solutions.pose # this will import the pose estimation model

# GET VIDEO FEED
cap = cv2.VideoCapture(1) # stores video capture as a variable

#push up counter
counter = 0
bad_butt_counter = 0
stage = None 
# Breakdown the pushup into two stages; down and up
# Down: Subject is close to the floor. Elbows in line with shoulders. Back straight
# Up: Arms fully extended. Back straight


with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose: # leveraging the tracking and detection confidences as 'pose'
    while cap.isOpened():
        ret, frame = cap.read()

        # Detect things and render
        # Recolour image as by default opencv gets frame colour as BGR and we want it as RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False #setting this to false saves a bunch of memory when we pass it through our pose estimation model

        # Makes detections and store them inside 'results'
        results = pose.process(image)

        # Recolour image again to revert it back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        

        # Extract the landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get Coords
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            # calculate angles
            right_elbow_angle = cl.calculate_angles(shoulder, elbow, wrist)

            butt_angle = cl.calculate_angles(shoulder, hip, knee)
            
            # about 70degrees between wrist, shoulder and knee  

            # Count pushups
            if (right_elbow_angle > 165):
                stage = "up"


            if (right_elbow_angle < 90) and stage == "up":
                stage = "down"
                counter += 1
                # counter for if butt too high or too low. 
                if (butt_angle < 170):
                    bad_butt_counter += 1

                print (counter)

        except:
            pass

        # counter
        # status box
        cv2.rectangle(image, (0,0), (225,133), (245,117,16), -1)
        
        # rep data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(image, 'BAD REPS', (15,72), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(bad_butt_counter), 
                    (10,120), 
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # stage data
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color = (245,117,66), thickness = 2, circle_radius = 2),
                                  mp_drawing.DrawingSpec(color = (245,66,230), thickness = 2, circle_radius = 2)
                                  )
        
        #image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        cv2.imshow('Mediapipe Feed', image) # shows the windows for the video feed

        if cv2.waitKey(10) & 0xFF == ord('q'): # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1
            break

cap.release()
cv2.destroyAllWindows()