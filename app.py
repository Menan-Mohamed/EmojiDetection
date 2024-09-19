import cv2
import mediapipe as mp
from deepface import DeepFace

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
hand_draw = mp.solutions.drawing_utils

while True:

    st, img = cap.read()
    frame = cv2.flip(img, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    coordlist = []

    if result.multi_hand_landmarks :
        for hand in result.multi_hand_landmarks:
            hand_draw.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)

            for id, lm in enumerate(hand.landmark):

                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                coordlist.append([id, cx, cy])

            
            thumb = coordlist[4][1] < coordlist[2][1]
            index = coordlist[8][2] < coordlist[6][2]
            middle = coordlist[12][2] < coordlist[10][2]
            ring = coordlist[16][2] < coordlist[14][2]
            pinky = coordlist[20][2] < coordlist[18][2]

            if thumb and not index and not middle and not ring and not pinky :
                if coordlist[4][2] < coordlist[2][2] :
                    em = cv2.imread('emoji/like.jpg')  
                    em = cv2.resize(em, (100,100))
                    frame[50:50+100,50:50+100] = em
                else :
                    em = cv2.imread('emoji/unlike.png')  
                    em = cv2.resize(em, (100,100))
                    frame[50:50+100,50:50+100] = em
            elif not thumb and index and not middle and not ring and not pinky :
                em = cv2.imread('emoji/index.png')  
                em = cv2.resize(em, (100,100))
                frame[50:50+100,50:50+100] = em
            elif not thumb and index and middle and not ring and not pinky :
                em = cv2.imread('emoji/victor.png')  
                em = cv2.resize(em, (100,100))
                frame[50:50+100,50:50+100] = em
            elif thumb and not index and not middle and not ring and pinky :
                em = cv2.imread('emoji/icon.jpeg')  
                em = cv2.resize(em, (100,100))
                frame[50:50+100,50:50+100] = em
            elif not thumb and not index and not middle and not ring and not pinky :
                em = cv2.imread('emoji/close.png')  
                em = cv2.resize(em, (100,100))
                frame[50:50+100,50:50+100] = em
            elif not thumb and not index and middle and ring and pinky :
                em = cv2.imread('emoji/exact.png')  
                em = cv2.resize(em, (100,100))
                frame[50:50+100,50:50+100] = em
            elif thumb and index and middle and ring and pinky :
                em = cv2.imread('emoji/hand.jpeg')  
                em = cv2.resize(em, (100,100))
                frame[50:50+100,50:50+100] = em
            else :
                print(" ")


    res = DeepFace.analyze(img_path = frame , actions=['emotion'] , enforce_detection=False )
    faces = face.detectMultiScale(rgb_frame , 1.1 , 4)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

        emotion = res[0]["dominant_emotion"]
        txt = str(emotion)
        
        if txt == "happy":
            em = cv2.imread('emoji/happy.png')  
            em = cv2.resize(em, (100,100))
            frame[50:50+100,50:50+100] = em
        elif txt == "sad":
            em = cv2.imread('emoji/sad.jpg')  
            em = cv2.resize(em, (100,100))
            frame[50:50+100,50:50+100] = em
        elif txt == "surprise":
            em = cv2.imread('emoji/wow.jpg')  
            em = cv2.resize(em, (100,100))
            frame[50:50+100,50:50+100] = em
        elif txt == "angry":
            em = cv2.imread('emoji/angry.png')  
            em = cv2.resize(em, (100,100))
            frame[50:50+100,50:50+100] = em
        elif txt == "disgust":
            em = cv2.imread('emoji/dis.jpeg')  
            em = cv2.resize(em, (100,100))
            frame[50:50+100,50:50+100] = em
    


    cv2.imshow('Emoji detection',frame)

    if cv2.waitKey(30) & 0xff == ord('x'):
        break
    
cap.release()
cv2.destroyAllWindows()