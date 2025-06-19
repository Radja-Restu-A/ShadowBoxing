import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import json
import os
import datetime
import numpy as np

GESTURE_DIR = 'boxing_poses'
GESTURE_JSON = os.path.join(GESTURE_DIR, 'boxing_poses.json')

if not os.path.exists(GESTURE_DIR):
    os.makedirs(GESTURE_DIR)

if os.path.exists(GESTURE_JSON):
    with open(GESTURE_JSON, 'r') as f:
        gesture_db = json.load(f)
else:
    gesture_db = []

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

class BoxingPoseCaptureApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Boxing Pose Capture')
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.landmarks = None
        self.imgtk = None
        
        self.panel = tk.Label(master)
        self.panel.pack()
        
        self.boxing_moves = ['JAB', 'HOOK', 'UPPERCUT']
        self.gesture_colors = {
            'JAB': '#43a047',
            'HOOK': '#e53935',
            'UPPERCUT': '#1e88e5'
        }
        
        self.command_var = tk.StringVar(master)
        self.command_var.set(self.boxing_moves[0])
        
        self.command_dropdown = tk.OptionMenu(master, self.command_var, *self.boxing_moves, command=self.update_status_color)
        self.command_dropdown.pack()
        
        self.capture_btn = tk.Button(master, text='Capture Pose', command=self.capture_pose, 
                                   bg='#43a047', fg='white', font=("Arial", 11, "bold"))
        self.capture_btn.pack(pady=3)
        
        self.multi_capture_btn = tk.Button(master, text='Capture 10x', command=self.capture_multiple_poses, 
                                         bg='#1e88e5', fg='white', font=("Arial", 11, "bold"))
        self.multi_capture_btn.pack(pady=3)
        
        self.status = tk.Label(master, text='', font=("Arial", 12, "bold"))
        self.status.pack()
        self.update_status_color(self.command_var.get())
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.status.config(text='Camera error!')
            return
            
        self.frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        self.landmarks = None
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            self.landmarks = self.extract_upper_body_landmarks(results.pose_landmarks.landmark)
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        self.imgtk = ImageTk.PhotoImage(image=img)
        self.panel.imgtk = self.imgtk
        self.panel.config(image=self.imgtk)
        self.master.after(10, self.update_frame)

    def extract_upper_body_landmarks(self, landmarks):
    # Fokus hanya pada bahu hingga pergelangan tangan
        upper_body_indices = [11, 12, 13, 14, 15, 16]
        
        extracted = []
        for idx in upper_body_indices:
            lm = landmarks[idx]
            extracted.append({'x': lm.x, 'y': lm.y, 'z': lm.z})
        
        return extracted

    def capture_pose(self):
        if self.landmarks is None:
            messagebox.showerror('Error', 'No pose detected!')
            return
            
        command = self.command_var.get().strip()
        if not command:
            messagebox.showerror('Error', 'Please select boxing move!')
            return
            
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        img_filename = f'{command}_{timestamp}.png'
        img_path = os.path.join(GESTURE_DIR, img_filename)
        
        cv2.imwrite(img_path, self.frame)
        
        pose_entry = {
            'name': command,
            'landmarks': self.landmarks,
            'image_path': img_path
        }
        
        gesture_db.append(pose_entry)
        
        with open(GESTURE_JSON, 'w') as f:
            json.dump(gesture_db, f, indent=2)
            
        self.update_status_color(command)
        self.status.config(text=f'Pose captured: {command}')

    def capture_multiple_poses(self):
        import time
        count = 0
        target = 10
        max_total_retry = 40 * target
        total_retry = 0
        
        while count < target and total_retry < max_total_retry:
            retry = 0
            found = False
            
            while retry < 30:
                ret, frame = self.cap.read()
                if not ret:
                    self.status.config(text='Camera error!')
                    self.master.update_idletasks()
                    time.sleep(0.2)
                    retry += 1
                    total_retry += 1
                    continue
                    
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                
                if results.pose_landmarks:
                    landmarks = self.extract_upper_body_landmarks(results.pose_landmarks.landmark)
                    
                    command = self.command_var.get().strip()
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    img_filename = f'{command}_{timestamp}_{count+1}.png'
                    img_path = os.path.join(GESTURE_DIR, img_filename)
                    
                    cv2.imwrite(img_path, frame)
                    
                    pose_entry = {
                        'name': command,
                        'landmarks': landmarks,
                        'image_path': img_path
                    }
                    
                    gesture_db.append(pose_entry)
                    
                    with open(GESTURE_JSON, 'w') as f:
                        json.dump(gesture_db, f, indent=2)
                        
                    count += 1
                    self.status.config(text=f'Captured {count}/{target}')
                    self.master.update_idletasks()
                    time.sleep(0.33)
                    found = True
                    break
                    
                if found:
                    break
                else:
                    self.status.config(text=f'Pose not detected! Retry {retry+1}/30 (sample {count+1}/{target})')
                    self.master.update_idletasks()
                    time.sleep(0.13)
                    retry += 1
                    total_retry += 1
                    
            if not found:
                self.status.config(text=f'Skipped (no pose) {count+1}/{target}')
                self.master.update_idletasks()
                time.sleep(0.2)
                
        if count < target:
            self.status.config(text=f'Capture stopped: only {count}/{target} captured')
        else:
            self.status.config(text=f'Completed capture {count}/{target} data!')

    def update_status_color(self, move_name):
        color = self.gesture_colors.get(move_name, '#757575')
        self.status.config(bg=color, fg='white')

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == '__main__':
    root = tk.Tk()
    app = BoxingPoseCaptureApp(root)
    root.mainloop()