import cv2
from deepface import DeepFace
from datetime import datetime

# Chiede all'utente il percorso del video
video_path = input("Inserisci il percorso del video: ").strip()

# Prova ad aprire il video
cam = cv2.VideoCapture(video_path)
if not cam.isOpened():
    print("Errore nell'apertura del video. Controlla il percorso.")
    exit()

frame_skip = 5  # Analizza 1 frame ogni 5
frame_count = 0

while True:
    cattura, frame = cam.read()
    if not cattura:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

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
