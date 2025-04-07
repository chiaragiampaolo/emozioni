import cv2
from deepface import DeepFace
from datetime import datetime

percorso_video = input("Inserisci il percorso del video: ").strip()

cam = cv2.VideoCapture(percorso_video)
if not cam.isOpened():
    print("Errore nell'apertura del video.")
    exit()

frame_skip = 5  
frame_count = 0

while True:
    cattura, frame = cam.read()
    if not cattura:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        pass

    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    if result:
        emotion = result[0]['dominant_emotion']
    else:
        emotion = "Errore"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("emotions.txt", "a") as file:
        file.write(f"{timestamp} - {emotion}\n")

    cv2.imshow("Emotions", frame)
    print(f"{timestamp} - {emotion}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
