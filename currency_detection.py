
import cv2
from ultralytics import YOLO
import numpy as np
from gtts import gTTS
import pyttsx3
import os

def getColours(cls_num):
    base_colors=[(255,0,0),(0,255,0),(0,0,255)]
    color_index=cls_num%len(base_colors)
    increments = [(1,-2,1),(-2,1,-1),(1,-1,2)]
    color=[base_colors[color_index][i]+increments[color_index][i]*(cls_num//len(base_colors))%256 for i in range(3)]
    return tuple(color)

model=YOLO(r'path\to\your\model\runs\classify\train\weights\last.pt')


videocap=cv2.VideoCapture(0)

while True:
    ret,frame=videocap.read()
    if not ret:
        continue
    

    cv2.imshow('Live Feed - Press "q" to Capture',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        captured_image=frame.copy()
        cv2.imwrite("captured_image.jpg", captured_image)

        results= model(captured_image)
        for result in results:
            names_dict = result.names  
            probs = result.probs.data.tolist() 

            detected_class = names_dict[np.argmax(probs)]
            print(f'Detected: {detected_class}')

            cv2.putText(captured_image, f'{detected_class}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Captured Image - Press any key to exit', captured_image)

        text= " ".join(detected_class.split('_'))
        print(text)
        # Convert text to speech
        # alternative 1: using pyttsx3. Comment this part if not using
        engine = pyttsx3.init()

        engine.say(text)
        engine.runAndWait()

        #alternative:2 using gTTS. uncomment this part to use
        ##myobj = gTTS(text=text, lang='en', slow=False)
        ##myobj.save("demo.mp3")
        ##os.system("start demo.mp3")
        cv2.waitKey(0) 
        break


videocap.release()
cv2.destroyAllWindows()
