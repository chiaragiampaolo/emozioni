import cv2
from deepface import DeepFace
from datetime import datetime

# Scelta del modello di rilevamento
print("Scegli il modello di rilevamento del volto:")
print("1 - haar cascade")
print("2 - ssd")
print("3 - dlib")
print("4 - mtcnn")
print("5 - retinaface")
scelta = input()

modelli = {
    "1": "haar cascade",
    "2": "ssd",
    "3": "dlib",
    "4": "mtcnn",
    "5": "retinaface"
}

detector_backend = modelli.get(scelta, "ssd")
print(f"Modello selezionato: {detector_backend}")

# Apertura della webcam
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Errore webcam")
    exit()

while True:
    cattura, frame = cam.read()
    if not cattura:
        break

    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend=detector_backend)
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
