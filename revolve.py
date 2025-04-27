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

class TaekwondoKickDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 狀態追踪
        self.kick_state = "ready"
        self.support_foot = None
        self.initial_x = None
        self.current_kick_start = 0
        self.last_kick_time = 0
        
        # 踢擊判定參數
        self.kick_cooldown = 1.5
        self.min_knee_diff = 0.15
        self.position_tolerance = 0.2
        self.velocity_threshold = 0.03
        self.min_time_threshold = 0.3
        
        # 踢擊紀錄
        self.kicks = []
    
        
    def reset_state(self):
        """重設偵測狀態"""
        self.kick_state = "ready"
        self.support_foot = None
        self.initial_x = None
    
    def detect_kick(self, frame, current_time):
        """偵測旋踢動作"""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        
        if not results.pose_landmarks:
            return frame, False, None, None
            
        landmarks = results.pose_landmarks.landmark
        height, width = frame.shape[:2]
        display_frame = frame.copy()
        
        # 每一幀都繪製腳部骨架
        self.draw_visualization(display_frame, results.pose_landmarks, 
                            landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL.value],
                            landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value])
        
        if self.kick_state == "ready":
            if current_time - self.last_kick_time < self.kick_cooldown:
                return display_frame, False, None, None
                
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            knee_height_diff = abs(left_knee.y - right_knee.y)
            
            if knee_height_diff > self.min_knee_diff:
                self.kick_state = "kicking"
                self.support_foot = "right" if left_knee.y < right_knee.y else "left"
                x_distance, heel, toe = self.calculate_foot_distance(landmarks, self.support_foot)
                self.initial_x = abs(x_distance)
                self.current_kick_start = current_time
                
                return display_frame, False, None, self.support_foot
        
        elif self.kick_state == "kicking":
            current_x, heel, toe = self.calculate_foot_distance(landmarks, self.support_foot)
            relative_position = current_x / self.initial_x
            
            # 更新視覺化
            self.draw_visualization(display_frame, results.pose_landmarks, heel, toe, relative_position)
            
            # 判定踢擊是否完成
            elapsed_time = current_time - self.current_kick_start
            if elapsed_time > self.min_time_threshold:
                success = abs(relative_position + 1) < self.position_tolerance or \
                        abs(relative_position - 1) < self.position_tolerance
                        
                self.kicks.append({
                    'time': current_time,
                    'success': success,
                    'kicking_leg': 'right' if self.support_foot == 'left' else 'left'
                })
                
                self.last_kick_time = current_time
                self.reset_state()
                return display_frame, True, success, self.support_foot
        
        return display_frame, False, None, self.support_foot
    
    def calculate_foot_distance(self, landmarks, foot_side="left"):
        """計算腳跟相對於腳尖的X軸距離"""
        if foot_side == "left":
            heel = landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL.value]
            toe = landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        else:
            heel = landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL.value]
            toe = landmarks[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]
        
        x_distance = heel.x - toe.x
        return x_distance, heel, toe

    def draw_visualization(self, frame, pose_landmarks, heel, toe, relative_position=None):
        """繪製腳部視覺化效果"""
        height, width = frame.shape[:2]
        
        # 定義顏色和粗細
        connection_color = (255, 255, 255)  # 白色連接線
        joint_color = (0, 255, 0)          # 綠色關節點
        kicking_color = (0, 0, 255)        # 紅色(踢腿標記)
        joint_thickness = 4
        connection_thickness = 2
        
        landmarks = pose_landmarks.landmark
        
        # 定義腳部關節連接
        leg_connections = [
            # 左腿
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            (self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.LEFT_HEEL),
            (self.mp_pose.PoseLandmark.LEFT_HEEL, self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
            # 右腿
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE),
            (self.mp_pose.PoseLandmark.RIGHT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_HEEL),
            (self.mp_pose.PoseLandmark.RIGHT_HEEL, self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
        ]
        
        # 定義腳部關節點
        leg_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.LEFT_HEEL,
            self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_HEEL,
            self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
        ]
        
        # 繪製腳部骨架連接
        for connection in leg_connections:
            start_landmark = landmarks[connection[0].value]
            end_landmark = landmarks[connection[1].value]
            
            start_point = (int(start_landmark.x * width), int(start_landmark.y * height))
            end_point = (int(end_landmark.x * width), int(end_landmark.y * height))
            
            # 判斷是否為踢腿的那條腿，給予不同顏色
            if self.support_foot:
                is_kicking_leg = (
                    (self.support_foot == "right" and connection[0].name.startswith("LEFT")) or
                    (self.support_foot == "left" and connection[0].name.startswith("RIGHT"))
                )
                line_color = kicking_color if is_kicking_leg else connection_color
            else:
                line_color = connection_color
                
            cv2.line(frame, start_point, end_point, line_color, connection_thickness)
        
        # 繪製腳部關節點
        for landmark_type in leg_landmarks:
            landmark = landmarks[landmark_type.value]
            point = (int(landmark.x * width), int(landmark.y * height))
            
            # 判斷是否為踢腿的關節點
            if self.support_foot:
                is_kicking_joint = (
                    (self.support_foot == "right" and landmark_type.name.startswith("LEFT")) or
                    (self.support_foot == "left" and landmark_type.name.startswith("RIGHT"))
                )
                point_color = kicking_color if is_kicking_joint else joint_color
            else:
                point_color = joint_color
                
            cv2.circle(frame, point, joint_thickness, point_color, -1)
        
        # 特別標記當前追踪的腳跟和腳尖
        if self.kick_state == "kicking":
            heel_point = (int(heel.x * width), int(heel.y * height))
            toe_point = (int(toe.x * width), int(toe.y * height))
            
            # 使用較大的圓圈標記
            cv2.circle(frame, heel_point, joint_thickness + 2, kicking_color, -1)
            cv2.circle(frame, toe_point, joint_thickness + 2, kicking_color, -1)
            cv2.line(frame, heel_point, toe_point, (0, 255, 255), 2)  # 黃色連接線
        
        # 顯示狀態資訊
        info_color = (0, 255, 0)  # 綠色文字
        font_scale = 0.7
        font_thickness = 2
        
        # 狀態資訊
        cv2.putText(frame, f"State: {self.kick_state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, info_color, font_thickness)
        
        if relative_position is not None:
            cv2.putText(frame, f"Position: {relative_position:.2f}x", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, info_color, font_thickness)
        
        if self.support_foot:
            kicking_leg = "Left" if self.support_foot == "right" else "Right"
            cv2.putText(frame, f"Kicking Leg: {kicking_leg}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, info_color, font_thickness)

class TaekwondoKickUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("跆拳道旋踢偵測系統")
        self.root.geometry("600x400")
        
        self.setup_ui()
        self.detector = TaekwondoKickDetector()
        
    def setup_ui(self):
        # 輸入設定區域
        input_frame = ttk.LabelFrame(self.root, text="輸入設定", padding="10")
        input_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(input_frame, text="影片來源：").grid(row=0, column=0, sticky="w")
        self.video_path_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.video_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(input_frame, text="瀏覽", command=self.browse_video).grid(row=0, column=2, padx=5)
        
        # 輸出設定區域
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
        
        # 控制按鈕
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
        
        # 獲取影片資訊
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        # 建立輸出目錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_path_var.get(), f"kick_detection_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 設定影片寫入器
        output_video_path = os.path.join(output_dir, 'detected_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            # 進行踢擊偵測
            processed_frame, is_kick, success, kicking_leg = self.detector.detect_kick(frame, current_time)
            
            # 寫入處理後的影片
            video_writer.write(processed_frame)
            
            # 更新進度
            progress = (frame_count / total_frames) * 100
            self.progress_var.set(progress)
            self.root.update()
            
            frame_count += 1
            
            # 顯示即時處理畫面
            cv2.imshow('Processing', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 釋放資源
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()
        
        # 儲存分析結果
        self.save_results(output_dir)
        
        self.status_var.set("處理完成")
        os.startfile(output_dir)  # 自動開啟輸出資料夾
    
    def save_results(self, output_dir):
        # 準備Excel資料
        data = []
        for kick in self.detector.kicks:
            data.append({
                '時間點(秒)': round(kick['time'], 2),
                '踢擊腳': kick['kicking_leg'],
                '是否成功': '成功' if kick['success'] else '失敗'
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
    
    # 啟動應用程式
    app = TaekwondoKickUI()
    app.run()

if __name__ == "__main__":
    main()