import cv2
import mediapipe as mp
import time
import numpy as np
import os
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
import sys

class MountainDefenseUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("山形防禦偵測系統")
        self.root.geometry("400x250")
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.setup_ui()
        
    def setup_ui(self):
        tk.Label(self.root, text="選擇影片檔案:").pack()
        self.video_path = tk.StringVar()
        tk.Entry(self.root, textvariable=self.video_path, width=50).pack()
        tk.Button(self.root, text="瀏覽", command=self.select_video).pack()

        tk.Label(self.root, text="選擇輸出目錄:").pack()
        self.output_dir = tk.StringVar()
        tk.Entry(self.root, textvariable=self.output_dir, width=50).pack()
        tk.Button(self.root, text="瀏覽", command=self.select_output_dir).pack()

        tk.Label(self.root, text="肩寬(cm):").pack()
        self.shoulder_width = tk.StringVar()
        tk.Entry(self.root, textvariable=self.shoulder_width, width=50).pack()

        tk.Button(self.root, text="開始分析", command=self.start_analysis).pack(pady=20)

    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        self.video_path.set(path)

    def select_output_dir(self):
        path = filedialog.askdirectory()
        self.output_dir.set(path)

    def calculate_angle(self, p1, p2, p3, image):
        width = image.shape[1]
        height = image.shape[0]
        a = np.sqrt(((p2.x - p3.x) * width) ** 2 + ((p2.y - p3.y) * height) ** 2)
        b = np.sqrt(((p1.x - p3.x) * width) ** 2 + ((p1.y - p3.y) * height) ** 2)
        c = np.sqrt(((p1.x - p2.x) * width) ** 2 + ((p1.y - p2.y) * height) ** 2)
        angle = np.degrees(np.arccos((a**2 + c**2 - b**2)/(2*a*c)))
        return angle

    def is_pose_for_counting(self, landmarks):
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        return (left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y)

    def is_pose_successful(self, landmarks, img):
        left_angle = self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW],
                                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST], img)
        right_angle = self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                         landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW],
                                         landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST], img)
        
        return 70 <= left_angle <= 110 and 70 <= right_angle <= 110

    def start_analysis(self):
        if not all([self.video_path.get(), self.output_dir.get(), self.shoulder_width.get()]):
            messagebox.showerror("錯誤", "請填寫所有必要資訊")
            return

        try:
            shoulder_width = float(self.shoulder_width.get())
        except ValueError:
            messagebox.showerror("錯誤", "請輸入有效的肩寬數值")
            return

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_folder = os.path.join(self.output_dir.get(), f"mountain_defense_{current_time}")
        os.makedirs(result_folder, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path.get())
        if not cap.isOpened():
            messagebox.showerror("錯誤", "無法開啟影片")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(os.path.join(result_folder, 'detected_video.mp4'),
                            cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        detection_data = []
        hold_frames = 0
        HOLD_THRESHOLD = 3
        MIN_TIME_BETWEEN_POSES = 1
        last_detection_time = 0
        pose_count = 0
        in_pose = False
        current_angles = []
        start_time = 0
        first_pose_recorded = False
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            left_angle = right_angle = 0
            shoulder_height_diff_cm = 0

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                landmarks = results.pose_landmarks.landmark
                
                left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

                left_angle = self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                                               landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW],
                                               landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST], frame)
                right_angle = self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW],
                                                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST], frame)

                shoulder_height_diff_px = abs(left_shoulder.y - right_shoulder.y) * frame_height
                shoulder_width_px = abs(left_shoulder.x - right_shoulder.x) * frame_width
                px_to_cm_ratio = float(self.shoulder_width.get()) / shoulder_width_px
                shoulder_height_diff_cm = shoulder_height_diff_px * px_to_cm_ratio

                is_counting_pose = self.is_pose_for_counting(landmarks)

                if is_counting_pose and not in_pose:
                    hold_frames += 1
                    if hold_frames >= HOLD_THRESHOLD and (current_time - last_detection_time) >= MIN_TIME_BETWEEN_POSES:
                        in_pose = True
                        pose_count += 1
                        last_detection_time = current_time
                        start_time = current_time
                        current_angles = []
                        first_pose_recorded = False

                elif in_pose:
                    current_angles.append({
                        '時間點': current_time,
                        '左手角度': left_angle,
                        '右手角度': right_angle,
                        '肩膀高度差(cm)': shoulder_height_diff_cm,
                    })

                    if left_wrist.y > left_shoulder.y and right_wrist.y > right_shoulder.y:
                        in_pose = False
                        hold_frames = 0
                        
                        # 移除 first_pose_recorded 檢查，直接記錄每次動作
                        if current_angles:
                            mid_idx = len(current_angles) // 2
                            mid_frame = current_angles[mid_idx]
                            detection_data.append({
                                '時間點': mid_frame['時間點'],
                                '左手角度': mid_frame['左手角度'],
                                '右手角度': mid_frame['右手角度'],
                                '肩膀高度差(cm)': mid_frame['肩膀高度差(cm)'],
                                '成功': '是' if 70 <= mid_frame['左手角度'] <= 110 and 70 <= mid_frame['右手角度'] <= 110 else '否'
                            })
                        current_angles = []

                cv2.putText(frame, f"Left: {left_angle:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Right: {right_angle:.1f}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Count: {pose_count}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            out.write(frame)
            cv2.imshow('Mountain Defense Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if detection_data:
            df = pd.DataFrame(detection_data)
            excel_path = os.path.join(result_folder, 'detection_results.xlsx')
            df.to_excel(excel_path, index=False)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        if sys.platform == 'win32':
            os.startfile(result_folder)
        elif sys.platform == 'darwin':
            os.system(f'open "{result_folder}"')
        else:
            os.system(f'xdg-open "{result_folder}"')
            
        messagebox.showinfo("完成", f"分析完成！\n共檢測到 {pose_count} 次山形防禦\n結果已保存到 {result_folder}")

    def run(self):
        self.root.mainloop()

def main():
    # 設定終端機編碼
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    # 啟動UI
    app = MountainDefenseUI()
    app.run()

if __name__ == "__main__":
    main()