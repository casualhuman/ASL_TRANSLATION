from fastapi import FastAPI, WebSocket, Depends
import cv2
import numpy as np
from dataset_landmarks_collection import  annotate_input_image, cv2_imshow, extract_keypoints
from aslscripts import class_name
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model 


app = FastAPI()



model = load_model('../asl_model_new1.h5')
class_name = class_name()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


frame_count = 0

@app.websocket("/ws")
async def process_frame(websocket: WebSocket):
    global frame_count

    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        frame_count += 1
        if frame_count % 2 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_result, annotated_image = annotate_input_image(rgb_frame)
            
            try: 
                points = extract_keypoints(detection_result).reshape(1, -1)
            except Exception as e:
                continue

            prediction = model.predict(points).round(2)
            predicted_class = class_name[prediction.argmax()]
            
            await websocket.send_text(predicted_class)

