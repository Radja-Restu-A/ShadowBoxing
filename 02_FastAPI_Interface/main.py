from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import cv2
import mediapipe as mp
import numpy as np
import joblib
import base64
import random
from typing import List, Dict
from fastapi.responses import HTMLResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load('01_Notebook_eksplorasi/boxing_classifier.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    model = None
    print("Model file not found. Please ensure the model is available at the specified path.")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.7)

class GameState:
    def __init__(self):
        self.health = 3
        self.score = 0
        self.current_move = None
        self.moves = ['JAB', 'HOOK', 'UPPERCUT']
        self.generate_new_move()
        
    def generate_new_move(self):
        self.current_move = random.choice(self.moves)
        
    def correct_move(self):
        self.score += 100
        self.generate_new_move()
        
    def wrong_move(self):
        self.health -= 1
        self.generate_new_move()
        
    def timeout(self):
        self.health -= 1
        self.generate_new_move()
        
    def is_game_over(self):
        return self.health <= 0
        
    def reset(self):
        self.health = 3
        self.score = 0
        self.generate_new_move()

game_state = GameState()

class PoseData(BaseModel):
    image: str

class GameStatus(BaseModel):
    health: int
    score: int
    current_move: str
    game_over: bool

def extract_upper_body_landmarks(landmarks):
    upper_body_indices = [11, 12, 13, 14, 15, 16]
    extracted = []
    for idx in upper_body_indices:
        lm = landmarks[idx]
        extracted.extend([lm.x, lm.y, lm.z])
    return extracted

def decode_image(image_data: str):
    try:
        image_bytes = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

@app.post("/predict_pose")
async def predict_pose(pose_data: PoseData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        img = decode_image(pose_data.image)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if not results.pose_landmarks:
            return {"predicted_move": "NO_POSE", "confidence": 0.0}
            
        landmarks = extract_upper_body_landmarks(results.pose_landmarks.landmark)
        prediction = model.predict([landmarks])
        confidence = max(model.predict_proba([landmarks])[0])
        
        predicted_move = prediction[0]
        
        if predicted_move == game_state.current_move and confidence > 0.7:
            game_state.correct_move()
            return {
                "predicted_move": predicted_move,
                "confidence": float(confidence),
                "result": "CORRECT",
                "game_state": {
                    "health": game_state.health,
                    "score": game_state.score,
                    "current_move": game_state.current_move,
                    "game_over": game_state.is_game_over()
                }
            }
        else:
            return {
                "predicted_move": predicted_move,
                "confidence": float(confidence),
                "result": "INCORRECT" if confidence > 0.7 else "LOW_CONFIDENCE"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/timeout")
async def handle_timeout():
    game_state.timeout()
    return {
        "health": game_state.health,
        "score": game_state.score,
        "current_move": game_state.current_move,
        "game_over": game_state.is_game_over()
    }

@app.get("/game_status")
async def get_game_status():
    return {
        "health": game_state.health,
        "score": game_state.score,
        "current_move": game_state.current_move,
        "game_over": game_state.is_game_over()
    }

@app.post("/reset_game")
async def reset_game():
    game_state.reset()
    return {
        "health": game_state.health,
        "score": game_state.score,
        "current_move": game_state.current_move,
        "game_over": game_state.is_game_over()
    }

@app.get("/")
async def get():
    return HTMLResponse(open("02_FastAPI_Interface/boxing_game.html", encoding="utf-8").read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)