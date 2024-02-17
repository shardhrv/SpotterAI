import cv2 
import mediapipe as mp
import numpy as np 
import calculations as cl

mp_drawing = mp.solutions.drawing_utils # this will give us all the drawing utililty ie to visualise all the poses
mp_pose = mp.solutions.pose # this will import the pose estimation model


# GET VIDEO FEED
cap = cv2.VideoCapture(1) # stores video capture as a variable

#bicep curl counter
counter = 0
stage = None

#Different push up types here
#cap = cv2.imread("C:/Users/dhruv/Desktop/Project/pushups/good_up.png" , cv2.IMREAD_COLOR)
#good_down = cv2.imread("C:\Users\dhruv\Desktop\Project\pushups\good_down.png", cv2.IMREAD_COLOR)
#butt_high = cv2.imread("C:\Users\dhruv\Desktop\Project\pushups\butt_low.png", cv2.IMREAD_COLOR)
#butt_low = cv2.imread("C:\Users\dhruv\Desktop\Project\pushups\butt_high.png", cv2.IMREAD_COLOR)



## Setup the mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: # leveraging the tracking and detection confidences as 'pose'
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
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # calculate angles
            angle = cl.calculate_angles(shoulder, elbow, wrist)

            # visualise angle
            cv2.putText(image, str(angle), 
                            tuple(np.multiply(elbow, [640,480]).astype(int)),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )
            
            # Count curls
            
            if angle > 160:
                stage = "down"
            if angle < 30 and stage == "down":
                stage = "up"
                counter += 1
                print (counter)

        except:
            pass

        # counter
        # status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # rep data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # stage data
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)



        # Make the renderings
        '''
        'mp_drawing.draw_landmarks(a,b,c)' uses our drawing utilities and goes ahead to draw to our image 
        a is where it draws to 
        b is showing the coordinates for each landmark on the body (ie the nodes of a graph)
        c is showing which landmarks are connected to where (ie the edges of a graph)
        d changes the color of landmarks (ie nodes)
        e changes the color of connections
        '''
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