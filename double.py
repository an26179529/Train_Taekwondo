import cv2
import mediapipe as mp
import numpy as np
import math
import time
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
from datetime import datetime
import os
from threading import Thread

class DoubleKnifeHandDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.movements = []
        self.last_movement_time = 0
        self.movement_cooldown = 1.5
        
        # 狀態管理
        self.movement_state = "ready"  # ready -> prepared -> executing -> completed
        self.current_front_arm = None
        self.front_arm_angle = 0
        self.back_arm_angle = 0
        
        # 預備姿勢參數
        self.READY_HANDS_MAX_DIST = 0.2  # 預備姿勢時雙手的最大距離（相對於身高）
        self.READY_ELBOW_ANGLE = 90      # 預備姿勢時手肘的大約角度
        self.READY_ANGLE_TOLERANCE = 20   # 角度容許誤差範圍
        
        # 完成動作的角度範圍
        self.FRONT_ARM_MIN_ANGLE = 85
        self.FRONT_ARM_MAX_ANGLE = 125
        self.BACK_ARM_MIN_ANGLE = 85
        self.BACK_ARM_MAX_ANGLE = 125
        
        # 動作追蹤
        self.execution_start_time = 0
        self.MAX_EXECUTION_TIME = 1.0  # 最長執行時間（秒）
            
    def angle3(self, p1, p2, p3, width, height):
        """計算三點間的角度"""
        try:
            # 轉換為實際座標
            p1_coord = np.array([p1.x * width, p1.y * height])
            p2_coord = np.array([p2.x * width, p2.y * height])
            p3_coord = np.array([p3.x * width, p3.y * height])
            
            # 計算向量
            v1 = p1_coord - p2_coord
            v2 = p3_coord - p2_coord
            
            # 計算角度
            angle = math.degrees(
                math.atan2(np.cross(v1, v2), np.dot(v1, v2))
            )
            
            return abs(angle)
        except:
            return 0
            
    def detect_movement(self, frame, timestamp):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        h, w = frame.shape[:2]
        
        if not results.pose_landmarks:
            return frame, False, None, None
            
        landmarks = results.pose_landmarks.landmark
        
        # 取得關鍵點
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # 計算基本測量
        hands_distance = math.sqrt(
            (left_wrist.x - right_wrist.x)**2 + 
            (left_wrist.y - right_wrist.y)**2
        ) * h
        
        body_height = abs(
            (left_shoulder.y + right_shoulder.y)/2 - 
            (left_hip.y + right_hip.y)/2
        ) * h
        
        # 計算角度
        left_angle = self.angle3(left_shoulder, left_elbow, left_wrist, w, h)
        right_angle = self.angle3(right_shoulder, right_elbow, right_wrist, w, h)
        
        # 判斷前手和後手
        if left_shoulder.x > right_shoulder.x:
            self.front_arm_angle = left_angle
            self.back_arm_angle = right_angle
            self.current_front_arm = "left"
        else:
            self.front_arm_angle = right_angle
            self.back_arm_angle = left_angle
            self.current_front_arm = "right"
            
        is_movement = False
        form_correct = None
        current_time = time.time()
        display_frame = frame.copy()
        
        # 判斷預備姿勢
        hands_close = hands_distance < (body_height * self.READY_HANDS_MAX_DIST)
        left_ready = abs(left_angle - self.READY_ELBOW_ANGLE) < self.READY_ANGLE_TOLERANCE
        right_ready = abs(right_angle - self.READY_ELBOW_ANGLE) < self.READY_ANGLE_TOLERANCE
        is_ready_pose = hands_close and left_ready and right_ready

        # 狀態機邏輯
        if self.movement_state == "ready":
            if is_ready_pose and (current_time - self.last_movement_time) >= self.movement_cooldown:
                self.movement_state = "prepared"
                self.execution_start_time = current_time
                
        elif self.movement_state == "prepared":
            if not is_ready_pose:  # 手開始分開
                self.movement_state = "executing"
                
        elif self.movement_state == "executing":
            # 判斷動作是否完成（雙手分開到定點）
            if hands_distance > (body_height * 0.5):  # 雙手分開足夠距離
                if (self.FRONT_ARM_MIN_ANGLE <= self.front_arm_angle <= self.FRONT_ARM_MAX_ANGLE and
                    self.BACK_ARM_MIN_ANGLE <= self.back_arm_angle <= self.BACK_ARM_MAX_ANGLE):
                    self.movement_state = "completed"
                    is_movement = True
                    form_correct = True
                    
                    self.movements.append({
                        'timestamp': timestamp,
                        'front_arm': self.current_front_arm,
                        'front_angle': self.front_arm_angle,
                        'back_angle': self.back_arm_angle,
                        'correct_form': True
                    })
                    
                    self.last_movement_time = current_time
            # 如果動作執行時間過長，重置狀態
            elif current_time - self.execution_start_time > self.MAX_EXECUTION_TIME:
                self.movement_state = "ready"
                    
        elif self.movement_state == "completed":
            if is_ready_pose:  # 回到預備姿勢
                self.movement_state = "ready"
        
        # 繪製資訊和骨架
        self.draw_info(display_frame, form_correct)
        self.draw_skeleton(display_frame, results.pose_landmarks)
        
        return display_frame, is_movement, form_correct, self.current_front_arm
    
    def draw_info(self, frame, form_correct):
        # 顯示角度資訊
        cv2.putText(frame, f"Front Arm Angle: {self.front_arm_angle:.1f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Back Arm Angle: {self.back_arm_angle:.1f}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        # 顯示當前狀態
        state_color = {
            "ready": (255, 255, 255),     # 白色
            "prepared": (255, 255, 0),     # 黃色
            "executing": (0, 255, 255),    # 青色
            "completed": (0, 255, 0)       # 綠色
        }.get(self.movement_state, (255, 255, 255))
        
        cv2.putText(frame, f"State: {self.movement_state}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
                    
        if form_correct is not None:
            result_text = "Correct" if form_correct else "Incorrect"
            result_color = (0, 255, 0) if form_correct else (0, 0, 255)
            cv2.putText(frame, f"Form: {result_text}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
                       
        cv2.putText(frame, f"Front Arm: {self.current_front_arm}", 
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def draw_skeleton(self, frame, pose_landmarks):
        # 繪製骨架連接
        self.mp_drawing.draw_landmarks(
            frame,
            pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

class DoubleKnifeHandUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("跆拳道雙手刀檢測系統")
        self.root.geometry("600x400")
        
        self.setup_ui()
        self.detector = DoubleKnifeHandDetector()
        self.video_path = None
        self.processing = False
        self.should_stop = False
        
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
        self.start_button = ttk.Button(button_frame, text="開始處理", command=self.start_processing)
        self.start_button.pack(side="left", padx=5)
        ttk.Button(button_frame, text="退出", command=self.on_closing).pack(side="right", padx=5)

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
            
    def on_closing(self):
        self.should_stop = True
        self.root.quit()

    def start_processing(self):
        if self.processing:
            return
            
        if not self.video_path_var.get() or not self.output_path_var.get():
            self.status_var.set("請選擇影片和輸出資料夾")
            return
            
        self.processing = True
        self.should_stop = False
        self.status_var.set("處理中...")
        self.start_button.configure(state="disabled")
        
        process_thread = Thread(target=self.process_video)
        process_thread.daemon = True
        process_thread.start()
        
    def process_video(self):
        try:
            cap = cv2.VideoCapture(self.video_path_var.get())
            if not cap.isOpened():
                self.status_var.set("無法開啟影片")
                return
                
            width = int(cap.get(3))  # CAP_PROP_FRAME_WIDTH = 3
            height = int(cap.get(4)) # CAP_PROP_FRAME_HEIGHT = 4
            fps = cap.get(5)         # CAP_PROP_FPS = 5
            total_frames = int(cap.get(7))  # CAP_PROP_FRAME_COUNT = 7
            frame_count = 0
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.output_path_var.get(), f"double_knife_hand_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            output_video_path = os.path.join(output_dir, 'detected_video.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            last_update_time = time.time()
            update_interval = 0.1
            
            while cap.isOpened() and not self.should_stop:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                timestamp = frame_count / fps
                processed_frame, is_movement, form_correct, front_arm = self.detector.detect_movement(frame, timestamp)
                video_writer.write(processed_frame)
                
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    progress = (frame_count / total_frames) * 100
                    self.progress_var.set(progress)
                    self.status_var.set(f"處理中... {progress:.1f}%")
                    self.root.update()
                    last_update_time = current_time
                
                frame_count += 1
                
                cv2.imshow('Processing', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()
            
            if not self.should_stop:
                self.save_results(output_dir, frame_count)
                self.status_var.set("處理完成")
                self.root.after(0, lambda: os.startfile(output_dir))
            
        except Exception as e:
            self.status_var.set(f"錯誤：{str(e)}")
        finally:
            self.processing = False
            self.start_button.configure(state="normal")
        
    def save_results(self, output_dir, frame_count):
        data = []
        for movement in self.detector.movements:
            data.append({
                '時間點(秒)': round(movement['timestamp'], 2),
                '前手': '右手' if movement['front_arm'] == 'right' else '左手',
                '前手角度': round(movement['front_angle'], 1),
                '後手角度': round(movement['back_angle'], 1),
                '動作正確': '正確' if movement['correct_form'] else '不正確'
            })
        
        df = pd.DataFrame(data)
        excel_path = os.path.join(output_dir, 'double_knife_hand_results.xlsx')
        df.to_excel(excel_path, index=False)
        
        # 顯示統計資訊
        correct_movements = sum(1 for m in self.detector.movements if m['correct_form'])
        total_movements = len(self.detector.movements)
        
        if total_movements > 0:
            accuracy = (correct_movements/total_movements*100)
        else:
            accuracy = 0
            
        stats_text = (
            f"分析完成！\n\n"
            f"檢測到的動作數：{total_movements}\n"
            f"結果已儲存至：{output_dir}"
        )
        
        self.root.after(0, lambda: messagebox.showinfo("分析結果", stats_text))
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    app = DoubleKnifeHandUI()
    app.run()

if __name__ == "__main__":
    main()