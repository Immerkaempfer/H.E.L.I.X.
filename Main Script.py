import cv2
import numpy as np
import mediapipe as mp
import time 
import threading 
from ultralytics import YOLO
from queue import Queue

from Gun_Detecion import GunDetector

gun_detector = GunDetector("Path")
process =YOLO("Path")
mp1_drawing= mp.solutions.drawing_utilits
mp1_pose=mp.solution.pose
cap = cv2.VideoCapture(0)
frame_q = Queue(maxsize=1 )
data_lock =threading.Lock()

landmark_data = []
Gun_data = []

def Main_Script():
    global landmark_data
    global Gun_data

    with mp1_pose.Pose(min_detection_confidence = 0.4, min_tracking_confidince = 0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image.flags.writeable= False         #optimization
            result = pose.process(image)
            gun_centers ,image = gun_detector.detect(image)
            image.flags.writeable = True
            image=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            try:
                landmarks = result.pose_landmakrs.landmark
                #i will do it manually :(
                Left_Shoulder = landmarks[mp1_pose.PoseLandmark.LEFT_SHOULDER.value]
                Right_Shoulder = landmarks[mp1_pose.PoseLandmark.RIGHT_SHOULDER.value]
                Left_hip = landmarks[mp1_pose.PoseLandmark.LEFT_HIP.value]
                Right_Hip = landmarks[mp1_pose.PoseLandmark.RIGHT_HIP.value]
                #wow warum mach ich das manuell 
                Right_Wrist = landmarks[mp1_pose.PoseLandmark.RIGHT_WRIST.value]
                Right_pinky = landmarks[mp1_pose.PoseLandmark.RIGHT_PINKY.value]
                right_index = landmarks[mp1_pose.PoseLandmark.RIGHT_INDEX.value]
                
                left_wrist= landmarks[mp1_pose.PoseLandmark.LEFT_WRIST.value]
                left_pinky = landmarks[mp1_pose.PoseLandmark.LEFT_PINKY.value]
                left_index = landmarks[mp1_pose.PoseLandmark.LEFT_INDEX.value]

                left_elbow = landmarks[mp1_pose.PoseLandmark.LEFT_WRIST.value]
                Right_elbow=landmarks[mp1_pose.PoseLandmark.RIGHT_ELBOW.value]

                left_elbow_x = left_elbow.x
                left_elbow_y = left_elbow.y

                Right_elbow_x = Right_elbow.x
                Right_elbow_y =Right_elbow.y

                x1 = Left_Shoulder.x 
                y1 =Left_Shoulder.y 

                x2 = Right_Shoulder.x
                y2 =Right_Shoulder.y 

                x3 = Left_hip.x
                y3 =Left_hip.y

                x4 =Right_Hip.x
                y4= Right_Hip.y

                x1_Hand1 = Right_Wrist.x
                y1_Hand1 = Right_Wrist.y

                x2_Hand1 = Right_pinky.x
                y2_Hand1 = Right_pinky.y

                x3_Hand1 = right_index.x
                y3_Hand1 =right_index.y

                x1_Hand2 =left_wrist.x
                y1_Hand2= left_wrist.y

                x2_Hand2 = left_pinky.x
                y2_Hand2 =left_pinky.y

                x3_Hand2 =left_index.x
                y3_Hand2 =left_index.y

                Left_Shoulder_XY = x1 , y1
                Left_Elbow_XY = left_elbow_x , left_elbow_y
                Left_Wrist_XY = x1_Hand2 , y1_Hand2

                Right_Shoulder_XY = x2 , y2#
                Right_Elbow_XY = Right_elbow_x , Right_elbow_y
                Right_Wrist_XY = x1_Hand1,y1_Hand1

                x,y = (x1+x2+x3+x4)/4 , (y1 +y2+ y3+y4)/4
                Right_Hand_x,Right_Hand_y = (x1_Hand1 + x2_Hand1+x3_Hand1)/3 , (y1_Hand1+y2_Hand1+y3_Hand1)/3
                Left_Hand_x ,Left_Hand_y =(x1_Hand2+x2_Hand2+x3_Hand2)/3 , (y1_Hand2 + y2_Hand2 +y3_Hand2)/3

                Cen_x= int(x *image.shape[1])
                Cen_y= int(y*image.shape[0])

                Cen_LeftHand_x =int(Left_Hand_x *image.shape[1])
                Cen_LeftHand_y = int(Left_Hand_y *image.shape[0])

                Cen_RightHand_x =int(Right_Hand_x* image.shape[1])
                Cen_RightHand_y= int(Right_Hand_y* image.shape[0])

                Cen_LeftHip_x = int(x3*image.shape[1])
                Cen_LeftHip_y = int(y3*image.shape[0]) 

                Cen_RightHip_x= int(x4 * image.shape[1])
                Cen_RightHip_y = int(y4*image.shape[0])

                Cen_Left_Shoulder_x = int(x1 * image.shape[1])
                Cen_Left_Shoulder_y = int(y1*image.shape[0])

                Cen_Right_Shoulder_x =int(x2 *image.shape[1])
                Cen_Right_Shoulder_y =int(y2*image.shape[0])

                with data_lock:
                    landmark_data.append((Cen_RightHand_x, Cen_RightHand_y , Cen_LeftHip_x,Cen_LeftHip_y,Cen_RightHip_x, Cen_RightHip_y, Cen_LeftHand_x,Cen_Left_Shoulder_x, Cen_Left_Shoulder_y,Cen_Right_Shoulder_x, Cen_Right_Shoulder_y, Cen_x ,Cen_y ))

                cv2.circle(image, (Cen_x, Cen_y), 6 , (0,255,0), -1)
                cv2.circle(image, (Cen_RightHip_x -40, Cen_RightHip_y), 6 ,(0,255,0),-1)              # -40 only temporary
                cv2.circle(image , (Cen_LeftHip_x + 40, Cen_LeftHip_y),6,(0,255,0),-1)                #  +40 only temporary
                cv2.circle(image, (Cen_LeftHand_x, Cen_LeftHand_y),6,(0,255,0), -1)
                cv2.circle(image , (Cen_RightHand_x, Cen_RightHand_y),6,(0,255,0), -1)
                
            except:
                pass

            try:
                for Mitte in gun_centers:
                    cv2.circle(image, Mitte, 6, (0, 255, 0), -1)
                    with data_lock:
                        Gun_data.append(Mitte)
                with data_lock:
                    Gun_data.append(Mitte)
                cv2.circle(image, (Mitte[0], Mitte[1]), 6, (0, 255, 0) ,-1)
            except:
                pass

            mp1_drawing.draw_lamdmarks(image,result.pose_landmarks, mp1_pose.POSE_CONNECTIONS)

            try:
                if not frame_q.full():
                    frame_q.put_nowait(image)
            except:
                pass

def Gun_Draw_detection():
    Zone = 50
    global landmark_data
    global Gun_data

    Right_TIme =0
    Left_TIME=0

    right_HIP  = False
    LEFT_HIP = False
    Gun_seen = False
    Gun_seen_Left = False

    while True:
        Mitte =None
        now = time.time()
        copy_data=[]
        copy_GUn_data=[]

        with data_lock:
            if Gun_data:
                copy_GUn_data=list(Gun_data)
                Gun_data.clear()
            if landmark_data:
                copy_data=list(landmark_data)
                landmark_data.clear()
        if copy_GUn_data:
            Gun_data_2=copy_GUn_data[-1]
            Mitte= Gun_data_2
        if copy_data:
            data=copy_data[-1]
            data_x_Right_Hand, data_y_Right_Hand, data_Left_Hip_x, data_Left_Hip_y,data_Right_Hip_x,data_Right_Hip_y,data_x_Left_Hand, data_y_Left_Hand ,data_x_Left_Shoulder, data_y_Left_Shoulder ,data_x_Right_Shoulder, data_y_Right_Shoulder, Cen_x ,Cen_y = data
            

            if (abs(data_x_Right_Hand-(data_Right_Hip_x)) < Zone and abs(data_y_Right_Hand -(data_Right_Hip_y)) < Zone) and not right_HIP:
                right_HIP = True
                Right_TIme = time.time()
            
            if Mitte is not None and right_HIP:
                Gun_seen = True
                print("GunSeen_Right")
            
            if right_HIP and (now - Right_TIme < 5) and Gun_seen:
                if data_y_Right_Hand < Cen_y:
                    print("Gun_Right_Hand")
                    right_HIP = False
                    Gun_seen =False
            
            else:
                right_HIP =False
                Gun_seen= False
            
            if (abs(data_x_Left_Hand - (data_Left_Hip_x)) < Zone and abs(data_y_Left_Hand- (data_Left_Hip_y)) < Zone) and not LEFT_HIP:
                Left_HIP =True
                Left_TIME = time.time()

            if Mitte is not None and LEFT_HIP:
                Gun_seen_Left=True
                print("GunSeen_Left")
            
            if LEFT_HIP and (now -Left_TIME < 5) and Gun_seen_Left:
                if data_y_Left_Hand < Cen_y:
                    print("Gun_Left_Hand")
                    LEFT_HIP = False
                    Gun_seen_Left = False
            else:
                LEFT_HIP =False
                Gun_seen_Left= False
        time.sleep(0.01)

main_script_thread= threading.Thread(target = Main_Script, daemon = True)
main_script_thread.start()

gun_draw_thread = threading.Thread(target=Gun_Draw_detection, daemon = True)
gun_draw_thread.start()
while True:
    if not frame_q.empty():
        frame =frame_q.get()
        cv2.imshow('Output', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

main_script_thread.join()

cap.release()
cv2.destroyAllWindows()

            
            