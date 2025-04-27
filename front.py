import cv2
import mediapipe as mp
import numpy as np
import math
import time
import sys
import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
from datetime import datetime
import os

class KickDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.kicks = []
        self.last_kick_time = 0
        self.kick_cooldown = 2.0
        
        self.kick_state = "ready"
        self.kick_start_time = 0
        self.lowest_foot_y = float('inf')
        self.current_kicking_leg = None
        self.current_right_angle = 0
        self.current_left_angle = 0
        self.peak_right_angle = 0
        self.peak_left_angle = 0
        
    def angle3(self, p1, p2, p3, width, height):
        """計算三點間的角度"""
        a = math.sqrt(math.pow((p2.x-p3.x)*width, 2) + math.pow((p2.y-p3.y)*height, 2))
        b = math.sqrt(math.pow((p1.x-p3.x)*width, 2) + math.pow((p1.y-p3.y)*height, 2))
        c = math.sqrt(math.pow((p1.x-p2.x)*width, 2) + math.pow((p1.y-p2.y)*height, 2))
        try:
            B = math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
            return B
        except:
            return 0
    
    def detect_kick(self, frame, timestamp):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        h, w = frame.shape[:2]
        
        if not results.pose_landmarks:
            return frame, False, None, None
        
        landmarks = results.pose_landmarks.landmark
        
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        right_heel = landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL]
        right_foot_index = landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        left_heel = landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL]
        left_foot_index = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        
        shoulder_height = (right_shoulder.y + left_shoulder.y) / 2
        
        right_foot_above = right_foot_index.y < shoulder_height
        left_foot_above = left_foot_index.y < shoulder_height
        
        # 計算當前角度
        self.current_right_angle = self.angle3(right_knee, right_heel, right_foot_index, w, h)
        self.current_left_angle = self.angle3(left_knee, left_heel, left_foot_index, w, h)
        
        is_kick = False
        form_correct = None
        kicking_leg = None
        current_time = time.time()
        
        MIN_FOOT_ANGLE = 130
        MAX_FOOT_ANGLE = 210
        
        # 在開始繪製之前先建立影像副本
        display_frame = frame.copy()
        
        if self.kick_state == "ready":
            if (current_time - self.last_kick_time) >= self.kick_cooldown:
                if right_foot_above or left_foot_above:
                    self.kick_state = "kicking"
                    self.kick_start_time = current_time
                    self.current_kicking_leg = "right" if right_foot_above else "left"
                    self.lowest_foot_y = float('inf')
                    kicking_leg = self.current_kicking_leg
        
        elif self.kick_state == "kicking":
            current_foot = right_foot_index if self.current_kicking_leg == "right" else left_foot_index
            foot_above = right_foot_above if self.current_kicking_leg == "right" else left_foot_above
            kicking_leg = self.current_kicking_leg
            
            if current_foot.y < self.lowest_foot_y:
                self.lowest_foot_y = current_foot.y
                self.peak_right_angle = self.current_right_angle
                self.peak_left_angle = self.current_left_angle
            
            if not foot_above:
                form_correct = (MIN_FOOT_ANGLE <= self.peak_right_angle <= MAX_FOOT_ANGLE if self.current_kicking_leg == "right"
                              else MIN_FOOT_ANGLE <= self.peak_left_angle <= MAX_FOOT_ANGLE)
                
                self.kicks.append({
                    'timestamp': timestamp,
                    'correct_form': form_correct,
                    'right_angle': self.peak_right_angle,
                    'left_angle': self.peak_left_angle,
                    'kicking_leg': self.current_kicking_leg,
                    'peak_y': self.lowest_foot_y
                })
                is_kick = True
                self.last_kick_time = current_time
                self.kick_state = "cooldown"
        
        elif self.kick_state == "cooldown":
            if current_time - self.last_kick_time >= self.kick_cooldown:
                self.kick_state = "ready"
                self.current_kicking_leg = None
                kicking_leg = None
                # 繪製資訊和骨架
        self.draw_info(display_frame, form_correct, kicking_leg)
        self.draw_foot_angles(display_frame, right_knee, right_heel, right_foot_index,
                            left_knee, left_heel, left_foot_index,
                            form_correct, kicking_leg)
        
        # 繪製肩膀高度參考線
        shoulder_y = int(shoulder_height * h)
        cv2.line(display_frame, (0, shoulder_y), (w, shoulder_y), (255, 0, 0), 1)
        
        return display_frame, is_kick, form_correct, kicking_leg
    
    def draw_foot_angles(self, frame, right_knee, right_heel, right_foot_index,
                        left_knee, left_heel, left_foot_index,
                        form_correct, kicking_leg):
        h, w = frame.shape[:2]
        
        def to_pixel(landmark):
            return (int(landmark.x * w), int(landmark.y * h))
        
        right_knee_pos = to_pixel(right_knee)
        right_heel_pos = to_pixel(right_heel)
        right_foot_pos = to_pixel(right_foot_index)
        left_knee_pos = to_pixel(left_knee)
        left_heel_pos = to_pixel(left_heel)
        left_foot_pos = to_pixel(left_foot_index)
        
        def get_color(is_kicking_leg):
            if not kicking_leg:
                return (255, 255, 255)  # 白色
            if is_kicking_leg:
                return (0, 255, 0) if form_correct else (0, 0, 255)  # 綠色或紅色
            return (128, 128, 128)  # 灰色
        
        right_color = get_color(kicking_leg == "right")
        left_color = get_color(kicking_leg == "left")
        
        # 繪製右腳
        cv2.line(frame, right_knee_pos, right_heel_pos, right_color, 2)
        cv2.line(frame, right_heel_pos, right_foot_pos, right_color, 2)
        cv2.circle(frame, right_knee_pos, 5, right_color, -1)
        cv2.circle(frame, right_heel_pos, 5, right_color, -1)
        cv2.circle(frame, right_foot_pos, 5, right_color, -1)
        
        # 繪製左腳
        cv2.line(frame, left_knee_pos, left_heel_pos, left_color, 2)
        cv2.line(frame, left_heel_pos, left_foot_pos, left_color, 2)
        cv2.circle(frame, left_knee_pos, 5, left_color, -1)
        cv2.circle(frame, left_heel_pos, 5, left_color, -1)
        cv2.circle(frame, left_foot_pos, 5, left_color, -1)
    
    def draw_info(self, frame, form_correct, kicking_leg):
        # 繪製當前角度
        cv2.putText(frame, f"Right Angle: {self.current_right_angle:.1f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Left Angle: {self.current_left_angle:.1f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.kick_state == "kicking":
            cv2.putText(frame, f"Kicking Leg: {'Right' if kicking_leg == 'right' else 'Left'}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 顯示最高點角度
            if kicking_leg == "right":
                cv2.putText(frame, f"Peak Right Angle: {self.peak_right_angle:.1f}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Peak Left Angle: {self.peak_left_angle:.1f}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if form_correct is not None:
            result_text = "Correct" if form_correct else "Incorrect"
            result_color = (0, 255, 0) if form_correct else (0, 0, 255)
            cv2.putText(frame, f"Form: {result_text}", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)

class KickDetectorUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("跆拳道動作檢測系統")
        self.root.geometry("600x400")
        
        self.setup_ui()
        self.detector = KickDetector()
        self.video_path = None
        
    def setup_ui(self):
        # 建立框架
        input_frame = ttk.LabelFrame(self.root, text="輸入設定", padding="10")
        input_frame.pack(fill="x", padx=10, pady=5)
        
        # 影片選擇
        ttk.Label(input_frame, text="影片來源：").grid(row=0, column=0, sticky="w")
        self.video_path_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.video_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(input_frame, text="瀏覽", command=self.browse_video).grid(row=0, column=2, padx=5)
        
        # 輸出設定
        output_frame = ttk.LabelFrame(self.root, text="輸出設定", padding="10")
        output_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(output_frame, text="輸出資料夾：").grid(row=0, column=0, sticky="w")
        self.output_path_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(output_frame, text="瀏覽", command=self.browse_output).grid(row=0, column=2, padx=5)
        
        # 狀態顯示
        self.status_var = tk.StringVar(value="就緒")
        status_frame = ttk.LabelFrame(self.root, text="狀態", padding="10")
        status_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(status_frame, textvariable=self.status_var).pack(fill="x")
        
        # 進度條
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress.pack(fill="x", padx=10, pady=5)
        
        # 按鈕
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(button_frame, text="開始處理", command=self.start_processing).pack(side="left", padx=5)
        ttk.Button(button_frame, text="退出", command=self.root.quit).pack(side="right", padx=5)

    def browse_video(self):
        filename = filedialog.askopenfilename(
            title="選擇影片檔案",
            filetypes=(("Video files", "*.mp4;*.avi;*.mov"), ("All files", "*.*"))
        )
        if filename:
            self.video_path_var.set(filename)

    def browse_output(self):
        directory = filedialog.askdirectory(title="選擇輸出資料夾")
        if directory:
            self.output_path_var.set(directory)
            
    def start_processing(self):
        if not self.video_path_var.get() or not self.output_path_var.get():
            self.status_var.set("請選擇影片和輸出資料夾")
            return
            
        self.status_var.set("處理中...")
        self.process_video()
        
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path_var.get())
        if not cap.isOpened():
            self.status_var.set("無法開啟影片")
            return
            
        # 獲取原始影片的資訊
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        # 建立輸出資料夾
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_path_var.get(), f"kick_detection_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 建立影片寫入器
        output_video_path = os.path.join(output_dir, 'detected_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = frame_count / fps
            
            # 處理幀並獲取結果
            processed_frame, is_kick, form_correct, kicking_leg = self.detector.detect_kick(frame, timestamp)
            
            # 寫入處理後的幀
            video_writer.write(processed_frame)
            
            # 更新進度
            progress = (frame_count / total_frames) * 100
            self.progress_var.set(progress)
            self.root.update()
            
            frame_count += 1
            
            # 顯示處理中的影片
            cv2.imshow('Processing', processed_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # 釋放資源
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        
        # 保存踢擊資訊到Excel
        self.save_results(output_dir, output_video_path)
        self.status_var.set("處理完成")
        
        # 自動打開輸出資料夾
        os.startfile(output_dir)
    
    def save_results(self, output_dir, video_path):
        # 建立Excel資料
        data = []
        for kick in self.detector.kicks:
            data.append({
                '時間點(秒)': round(kick['timestamp'], 2),
                '踢擊腳': '右腳' if kick['kicking_leg'] == 'right' else '左腳',
                '右腳角度': round(kick['right_angle'], 1),
                '左腳角度': round(kick['left_angle'], 1),
                '動作正確': '正確' if kick['correct_form'] else '不正確'
            })
        
        # 輸出Excel
        df = pd.DataFrame(data)
        excel_path = os.path.join(output_dir, 'kick_detection_results.xlsx')
        df.to_excel(excel_path, index=False)
        
    def run(self):
        self.root.mainloop()

def main():
    # 設定終端機編碼
    sys.stdout.reconfigure(encoding='utf-8')
    
    # 啟動UI
    app = KickDetectorUI()
    app.run()

if __name__ == "__main__":
    main()