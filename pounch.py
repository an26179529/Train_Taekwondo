import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk
import os
from datetime import datetime
import sys

class TaekwondoPunchUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("跆拳道正拳偵測系統")
        self.root.geometry("600x400")
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Drawing connections for arms only
        self.arm_connections = frozenset([
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST),
        ])
        
        # Initialize variables
        self.HISTORY_LENGTH = 5
        self.left_wrist_history = deque(maxlen=self.HISTORY_LENGTH)
        self.right_wrist_history = deque(maxlen=self.HISTORY_LENGTH)
        self.last_punch_time = 0
        self.PUNCH_COOLDOWN = 0.5
        self.TOLERANCE_CM = 5
        self.punch_records = []
        
        self.setup_ui()
    
    def setup_ui(self):
        # Input frame
        input_frame = ttk.LabelFrame(self.root, text="輸入設定", padding="10")
        input_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(input_frame, text="影片來源：").grid(row=0, column=0, sticky="w")
        self.video_path_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.video_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(input_frame, text="瀏覽", command=self.browse_video).grid(row=0, column=2, padx=5)
        
        # Output frame
        output_frame = ttk.LabelFrame(self.root, text="輸出設定", padding="10")
        output_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(output_frame, text="輸出資料夾：").grid(row=0, column=0, sticky="w")
        self.output_path_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(output_frame, text="瀏覽", command=self.browse_output).grid(row=0, column=2, padx=5)
        
        # Status and progress
        self.status_var = tk.StringVar(value="就緒")
        status_frame = ttk.LabelFrame(self.root, text="狀態", padding="10")
        status_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(status_frame, textvariable=self.status_var).pack(fill="x")
        
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress.pack(fill="x", padx=10, pady=5)
        
        # Buttons
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

    def calculate_angle(self, p1, p2, p3):
        vector1 = np.array([p1.x - p2.x, p1.y - p2.y])
        vector2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    def detect_punch_impact(self, landmarks, wrist_history, side='right'):
        if len(wrist_history) < 3:
            return False, 0
        
        if side == 'right':
            shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        else:
            shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]

        velocities = []
        for i in range(1, len(wrist_history)):
            dx = wrist_history[i][0] - wrist_history[i-1][0]
            dt = wrist_history[i][2] - wrist_history[i-1][2]
            if dt > 0:
                velocities.append(dx / dt)
        
        if len(velocities) < 2:
            return False, 0
            
        current_vel = velocities[-1]
        prev_vel = velocities[-2]
        
        elbow_angle = self.calculate_angle(wrist, elbow, shoulder)
        wrist_higher_than_elbow = wrist.y < elbow.y
        
        VELOCITY_THRESHOLD = 40
        MIN_ELBOW_ANGLE = 150
        
        is_punching = (
            abs(current_vel) > VELOCITY_THRESHOLD and
            prev_vel * current_vel < 0 and
            elbow_angle > MIN_ELBOW_ANGLE and
            wrist_higher_than_elbow
        )
        
        return is_punching, abs(current_vel)

    def estimate_solar_plexus(self, landmarks, frame):
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        # 計算肩膀中點
        solar_plexus_x = int((left_shoulder.x + right_shoulder.x) / 2 * frame.shape[1])
        
        # 取得肩膀高度並往下偏移
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        solar_plexus_y = int((shoulder_y + (shoulder_y * 0.1)) * frame.shape[0])  # 肩膀往下10%的位置
        
        return solar_plexus_x, solar_plexus_y

    def draw_arm_landmarks(self, frame, landmarks):
        # Draw only arm landmarks and connections
        for connection in self.arm_connections:
            start_idx = connection[0].value
            end_idx = connection[1].value
            
            start_point = (int(landmarks.landmark[start_idx].x * frame.shape[1]),
                          int(landmarks.landmark[start_idx].y * frame.shape[0]))
            end_point = (int(landmarks.landmark[end_idx].x * frame.shape[1]),
                        int(landmarks.landmark[end_idx].y * frame.shape[0]))
            
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
            cv2.circle(frame, start_point, 5, (0, 0, 255), -1)
            cv2.circle(frame, end_point, 5, (0, 0, 255), -1)

    def process_video(self):
        if not self.video_path_var.get() or not self.output_path_var.get():
            self.status_var.set("請選擇影片和輸出資料夾")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_path_var.get(), f"punch_detection_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path_var.get())
        if not cap.isOpened():
            self.status_var.set("無法開啟影片")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_video_path = os.path.join(output_dir, 'detected_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_count = 0
        current_time = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / fps

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                self.draw_arm_landmarks(frame, results.pose_landmarks)
                landmarks = results.pose_landmarks.landmark
                
                solar_plexus_x, solar_plexus_y = self.estimate_solar_plexus(landmarks, frame)
                cv2.circle(frame, (solar_plexus_x, solar_plexus_y), 5, (0, 0, 255), -1)
                cv2.line(frame, (0, solar_plexus_y), (width, solar_plexus_y), (0, 255, 255), 1)

                left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
                right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
                
                left_wrist_x = int(left_wrist.x * frame.shape[1])
                left_wrist_y = int(left_wrist.y * frame.shape[0])
                right_wrist_x = int(right_wrist.x * frame.shape[1])
                right_wrist_y = int(right_wrist.y * frame.shape[0])

                self.left_wrist_history.append((left_wrist_x, left_wrist_y, current_time))
                self.right_wrist_history.append((right_wrist_x, right_wrist_y, current_time))

                left_impact, left_vel = self.detect_punch_impact(landmarks, self.left_wrist_history, 'left')
                right_impact, right_vel = self.detect_punch_impact(landmarks, self.right_wrist_history, 'right')

                if current_time - self.last_punch_time >= self.PUNCH_COOLDOWN and (left_impact or right_impact):
                    if left_vel > right_vel:
                        fist_y = left_wrist_y
                        current_side = 'left'
                    else:
                        fist_y = right_wrist_y
                        current_side = 'right'
                    
                    height_diff = fist_y - solar_plexus_y
                    is_correct_height = abs(height_diff) <= 20
                    
                    if is_correct_height:
                        result = "成功"
                        suggestion = None
                        cv2.putText(frame, "成功!", (width//2 - 100, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    else:
                        result = "失敗"
                        suggestion = "提高" if height_diff > 0 else "降低"
                        cv2.putText(frame, f"{suggestion}!", (width//2 - 100, 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                    self.punch_records.append({
                        'time': current_time,
                        'side': current_side,
                        'result': result,
                        'suggestion': suggestion
                    })
                    self.last_punch_time = current_time

            out.write(frame)
            
            progress = (frame_count / total_frames) * 100
            self.progress_var.set(progress)
            self.root.update()

            cv2.imshow('Processing', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        self.save_results(output_dir)
        self.status_var.set("處理完成")
        os.startfile(output_dir)

    def save_results(self, output_dir):
        data = []
        for punch in self.punch_records:
            data.append({
                '時間點(秒)': round(punch['time'], 2),
                '出拳手': '右手' if punch['side'] == 'right' else '左手',
                '結果': punch['result'],
                '建議': punch['suggestion'] if punch['suggestion'] else '無'
            })
        
        df = pd.DataFrame(data)
        excel_path = os.path.join(output_dir, 'punch_detection_results.xlsx')
        df.to_excel(excel_path, index=False)

    def start_processing(self):
        if not self.video_path_var.get() or not self.output_path_var.get():
            self.status_var.set("請選擇影片和輸出資料夾")
            return
            
        self.status_var.set("處理中...")
        self.progress_var.set(0)
        self.root.update()
        
        try:
            self.process_video()
        except Exception as e:
            self.status_var.set(f"處理發生錯誤: {str(e)}")
            print(f"Error: {str(e)}")
        
        self.root.update()

    def run(self):
        self.root.mainloop()

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    app = TaekwondoPunchUI()
    app.run()

if __name__ == "__main__":
    main()