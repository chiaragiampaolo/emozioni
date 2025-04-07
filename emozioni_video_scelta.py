import cv2
from deepface import DeepFace
from datetime import datetime

print("Inserisci il percorso del file:")
percorso = input()



cam = cv2.VideoCapture(percorso)
if not cam.isOpened():
    print("Errore nell'apertura del video")
    exit()

frame_skip = 5 # Analizza solo 1 frame ogni 10
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

        with open("emotions.txt", "a") as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"{timestamp} - {emotion}\n")

        
        cv2.imshow("Emotions", frame)
        print(f"{timestamp} - {emotion}\n")

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cam.release()
cv2.destroyAllWindows()
