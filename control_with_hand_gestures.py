
import cv2
import mediapipe
import pyttsx3

camera = cv2.VideoCapture(0)

mpHands = mediapipe.solutions.hands
hands = mpHands.Hands()
mpDraw = mediapipe.solutions.drawing_utils

checkThumbsUp = False

engine = pyttsx3.init()

if not camera.isOpened():
    print("Error: Camera didn't turn on . . .")
    exit()

while True:
    ret, frame = camera.read()

    if not ret:
        print("Error: Frame unreadable . . .")
        break

    height, width, channel = frame.shape

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hlms = hands.process(frameRGB)

    if hlms.multi_hand_landmarks:

        for handlandmarks in hlms.multi_hand_landmarks:

            for fingerNum, landmark in enumerate(handlandmarks.landmark):
                positionX, positionY = int(landmark.x * width), int(landmark.y * height)

                cv2.putText(frame, str(fingerNum), (positionX, positionY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                if fingerNum > 4 and landmark.y < handlandmarks.landmark[2].y:
                    break

                if fingerNum == 20 and landmark.y > handlandmarks.landmark[2].y:
                    checkThumbsUp = True

                if fingerNum == 4:
                    cv2.circle(frame, (positionX, positionY), 30, (0, 0, 0), cv2.FILLED)

            mpDraw.draw_landmarks(frame, handlandmarks, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Camera", frame)

    if checkThumbsUp:
        engine.say("ThumbsUp")
        engine.runAndWait()
        break


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

print("Finish . . .")
