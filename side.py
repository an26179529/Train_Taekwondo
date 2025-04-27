import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime, timedelta
import time
import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import os
import sys

class TaekwondoKickUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("跆拳道側踢偵測系統")
        self.root.geometry("600x400")
        
        # 初始化 MediaPipe
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
        
        # 定義關鍵點索引
        self.key_points = {
            'right_heel': 30,
            'right_foot_index': 32,
            'left_heel': 29,
            'left_foot_index': 31,
            'nose': 0
        }
        
        # 初始化追踪變量
        self.previous_heel_position = None
        self.is_side_kicking = False
        self.kicks_record = []
        self.current_kick_info = {
            'start_time': None,
            'end_time': None,
            'heel_moved': False,
            'kicking_leg': None
        }
        
        self.setup_ui()
        
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

    def is_leg_above_head(self, landmarks):
        """檢查是否有腳超過頭部高度"""
        nose_y = landmarks.landmark[self.key_points['nose']].y
        left_foot_y = landmarks.landmark[self.key_points['left_foot_index']].y
        right_foot_y = landmarks.landmark[self.key_points['right_foot_index']].y
        
        return (left_foot_y < nose_y) or (right_foot_y < nose_y)
    
    def check_heel_movement(self, landmarks, support_heel_idx, threshold=0.02):
        """檢查支撐腳腳跟的移動"""
        heel_point = landmarks.landmark[support_heel_idx]
        current_position = np.array([heel_point.x, heel_point.y])
        
        if self.previous_heel_position is None:
            self.previous_heel_position = current_position
            return False, 0
            
        movement = np.linalg.norm(current_position - self.previous_heel_position)
        is_moving = movement > threshold
        
        self.previous_heel_position = current_position
        return is_moving, movement

    def format_time(self, milliseconds):
        """將毫秒轉換為時間格式"""
        return str(timedelta(milliseconds=milliseconds))[:-3]
    
    def start_processing(self):
        """開始處理影片的方法"""
        if not self.video_path_var.get() or not self.output_path_var.get():
            self.status_var.set("請選擇影片和輸出資料夾")
            return
            
        self.status_var.set("處理中...")
        self.progress_var.set(0)  # 重設進度條
        self.root.update()
        
        try:
            self.process_video()
        except Exception as e:
            self.status_var.set(f"處理發生錯誤: {str(e)}")
            print(f"Error: {str(e)}")
        
        self.root.update()

    def process_video(self):
        if not self.video_path_var.get() or not self.output_path_var.get():
            self.status_var.set("請選擇影片和輸出資料夾")
            return

        # 建立輸出資料夾
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.output_path_var.get(), f"side_kick_detection_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path_var.get())
        if not cap.isOpened():
            self.status_var.set("無法開啟影片")
            return

        # 獲取影片資訊
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 設置輸出影片
        output_video_path = os.path.join(output_dir, 'detected_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_count = 0
        self.kicks_record = []  # 重置記錄

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 獲取當前時間戳
            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            
            # 處理影像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # 繪製骨架
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # 檢查是否正在進行側踢
                is_leg_high = self.is_leg_above_head(results.pose_landmarks)
                
                if is_leg_high and not self.is_side_kicking:
                    # 開始新的側踢
                    self.is_side_kicking = True
                    self.current_kick_info = {
                        'start_time': current_time_ms,
                        'end_time': None,
                        'heel_moved': False,
                        'kicking_leg': 'left' if results.pose_landmarks.landmark[self.key_points['left_foot_index']].y < results.pose_landmarks.landmark[self.key_points['right_foot_index']].y else 'right'
                    }
                    
                elif is_leg_high and self.is_side_kicking:
                    # 側踢進行中
                    if not self.current_kick_info['heel_moved']:
                        kicking_leg = self.current_kick_info['kicking_leg']
                        support_heel_idx = self.key_points['left_heel'] if kicking_leg == 'right' else self.key_points['right_heel']
                        is_moving, _ = self.check_heel_movement(results.pose_landmarks, support_heel_idx)
                        if is_moving:
                            self.current_kick_info['heel_moved'] = True
                    
                elif not is_leg_high and self.is_side_kicking:
                    # 結束側踢
                    self.is_side_kicking = False
                    result = "失敗" if self.current_kick_info['heel_moved'] else "成功"
                    
                    kick_record = {
                        'time': self.format_time(self.current_kick_info['start_time']),
                        'timestamp': self.current_kick_info['start_time'] / 1000.0,  # 轉換為秒
                        'result': result,
                        'kicking_leg': self.current_kick_info['kicking_leg']
                    }
                    
                    self.kicks_record.append(kick_record)
            
            # 寫入影片
            out.write(frame)
            
            # 更新進度
            frame_count += 1
            progress = (frame_count / total_frames) * 100
            self.progress_var.set(progress)
            self.root.update()

            # 顯示處理中的影片
            cv2.imshow('Processing', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 釋放資源
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # 保存Excel
        self.save_results(output_dir)
        self.status_var.set("處理完成")
        
        # 自動打開輸出資料夾
        os.startfile(output_dir)

    def save_results(self, output_dir):
        # 建立Excel資料
        data = []
        for kick in self.kicks_record:
            data.append({
                '時間點(秒)': round(kick['timestamp'], 2),
                '踢擊腳': '右腳' if kick['kicking_leg'] == 'right' else '左腳',
                '結果': kick['result']
            })
        
        # 輸出Excel
        df = pd.DataFrame(data)
        excel_path = os.path.join(output_dir, 'side_kick_detection_results.xlsx')
        df.to_excel(excel_path, index=False)

    def run(self):
        self.root.mainloop()

def main():
    # 設定終端機編碼
    sys.stdout.reconfigure(encoding='utf-8')
    
    # 啟動UI
    app = TaekwondoKickUI()
    app.run()

if __name__ == "__main__":
    main()