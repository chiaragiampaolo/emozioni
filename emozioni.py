import cv2
from deepface import DeepFace
from datetime import datetime


cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Errore webcam")
    exit()


while True:
    cattura, frame = cam.read()
    if not cattura:
        break  

    
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='ssd')
    if result:
        emotion = result[0]['dominant_emotion']
    else:
        emotion = None

    
    if emotion:
        print(f"Emozione rilevata: {emotion}")
        with open("emotions.txt", "a") as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"{timestamp} - {emotion}\n")

    
    cv2.imshow("Emotions", frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()
