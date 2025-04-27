import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import subprocess

class TaekwondoAnalysisSystem:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("跆拳道綜合動作偵測系統")
        self.root.geometry("800x600")
        
        # 設定介面樣式
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", font=("Arial", 12), padding=5)
        self.style.configure("TLabel", font=("Arial", 12), background="#f0f0f0")
        self.style.configure("Title.TLabel", font=("Arial", 16, "bold"), background="#f0f0f0")
        
        # 建立主框架
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 標題
        ttk.Label(self.main_frame, text="跆拳道綜合動作偵測系統", style="Title.TLabel").pack(pady=20)
        
        # 創建動作選擇區域
        self.create_action_selection()
        
        # 顯示狀態
        self.status_var = tk.StringVar(value="就緒")
        status_frame = ttk.LabelFrame(self.main_frame, text="系統狀態", padding="10")
        status_frame.pack(fill="x", padx=10, pady=20)
        ttk.Label(status_frame, textvariable=self.status_var).pack(fill="x")
        
        # 底部按鈕
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill="x", padx=10, pady=10)
        ttk.Button(button_frame, text="退出系統", command=self.root.quit).pack(side="right", padx=5)
        
        # 動作模組映射表和啟動類別
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.action_modules = {
            "雙手刀": {
                "path": os.path.join(current_dir, "double.py"),
                "class": "DoubleKnifeHandUI",
                "has_main": True
            },
            "前踢": {
                "path": os.path.join(current_dir, "front.py"),
                "class": "KickDetectorUI",
                "has_main": True
            },
            "山形防禦": {
                "path": os.path.join(current_dir, "mountain.py"),
                "class": "MountainDefenseUI",
                "has_main": False
            },
            "正拳": {
                "path": os.path.join(current_dir, "pounch.py"),
                "class": "TaekwondoPunchUI",
                "has_main": True
            },
            "旋踢": {
                "path": os.path.join(current_dir, "revolve.py"),
                "class": "TaekwondoKickUI",
                "has_main": True
            },
            "側踢": {
                "path": os.path.join(current_dir, "side.py"),
                "class": "TaekwondoKickUI",
                "has_main": True
            }
        }
        
    def create_action_selection(self):
        # 動作選擇框架
        action_frame = ttk.LabelFrame(self.main_frame, text="選擇偵測動作", padding="10")
        action_frame.pack(fill="x", padx=10, pady=10)
        
        # 動作圖示和按鈕
        actions_container = ttk.Frame(action_frame)
        actions_container.pack(fill="x", padx=10, pady=10)
        
        # 建立每個動作的卡片
        actions = [
            {"name": "雙手刀", "desc": "偵測跆拳道雙手刀技術動作"},
            {"name": "前踢", "desc": "偵測跆拳道前踢動作"},
            {"name": "山形防禦", "desc": "偵測山形防禦姿勢"},
            {"name": "正拳", "desc": "偵測正拳出擊動作"},
            {"name": "旋踢", "desc": "偵測旋踢動作"},
            {"name": "側踢", "desc": "偵測側踢動作"}
        ]
        
        # 配置網格以適應多個卡片
        for i in range(3):
            actions_container.columnconfigure(i, weight=1)
        for i in range(2):
            actions_container.rowconfigure(i, weight=1)
        
        row = 0
        col = 0
        for idx, action in enumerate(actions):
            self.create_action_card(actions_container, action, row, col)
            col += 1
            if col > 2:  # 每行顯示3個卡片
                col = 0
                row += 1
    
    def create_action_card(self, parent, action, row, col):
        # 創建卡片框架
        card = ttk.Frame(parent, borderwidth=2, relief="raised")
        card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        
        # 卡片內容
        ttk.Label(card, text=action["name"], font=("Arial", 14, "bold")).pack(pady=(10, 5))
        ttk.Label(card, text=action["desc"], wraplength=200).pack(pady=(0, 10))
        ttk.Button(card, text="選擇此動作", 
                  command=lambda a=action["name"]: self.select_action(a)).pack(pady=(0, 10))
    
    def select_action(self, action_name):
        """選擇動作並啟動相應模組"""
        if action_name in self.action_modules:
            module_info = self.action_modules[action_name]
            module_path = module_info["path"]
            
            self.status_var.set(f"正在啟動 {action_name} 偵測模組...")
            self.root.update()
            
            try:
                # 檢查模組檔案是否存在
                if not os.path.exists(module_path):
                    messagebox.showerror("錯誤", f"找不到模組檔案: {module_path}")
                    self.status_var.set("就緒")
                    return
                
                # 對山形防禦特殊處理
                if action_name == "山形防禦":
                    self.run_mountain_defense_module()
                else:
                    # 使用子進程執行模組
                    self.run_module_as_subprocess(module_path)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                messagebox.showerror("錯誤", f"啟動模組時發生錯誤:\n{str(e)}")
                self.status_var.set("就緒")
        else:
            messagebox.showerror("錯誤", f"未知的動作: {action_name}")
    
    def run_mountain_defense_module(self):
        """特別處理山形防禦模組"""
        try:
            # 隱藏主視窗
            self.root.withdraw()
            
            # 創建臨時啟動腳本
            temp_script = """
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mountain import MountainDefenseUI

if __name__ == "__main__":
    app = MountainDefenseUI()
    app.run()
"""
            temp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_mountain_launcher.py")
            
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(temp_script)
            
            # 運行臨時腳本
            python_executable = sys.executable
            result = subprocess.run([python_executable, temp_file], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    text=True)
            
            # 刪除臨時腳本
            try:
                os.remove(temp_file)
            except:
                pass
                
            if result.returncode != 0:
                print("模組執行失敗:")
                print(f"標準輸出: {result.stdout}")
                print(f"錯誤輸出: {result.stderr}")
                messagebox.showerror("錯誤", f"模組執行失敗:\n{result.stderr}")
            
            # 重新顯示主視窗
            self.root.deiconify()
            self.status_var.set("就緒")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("錯誤", f"執行模組時發生錯誤:\n{str(e)}")
            self.status_var.set("就緒")
            self.root.deiconify()
    
    def run_module_as_subprocess(self, module_path):
        """使用子進程執行指定的模組"""
        try:
            # 隱藏主視窗
            self.root.withdraw()
            
            python_executable = sys.executable
            print(f"使用 Python: {python_executable}")
            print(f"執行模組: {module_path}")
            
            # 創建一個隔離的啟動器腳本，避免模組間互相影響
            # 避免在f-string中使用反斜線
            module_path_normalized = module_path.replace('\\', '/')
            launcher_script = f"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
module_path = "{module_path_normalized}"
module_name = os.path.splitext(os.path.basename(module_path))[0]

# 只導入要使用的模組
spec = __import__(module_name)

# 執行main函數
if hasattr(spec, 'main'):
    spec.main()
else:
    print(f"模組 {{module_name}} 沒有main函數")
"""
            temp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                    f"temp_launcher_{os.path.basename(module_path)}.py")
            
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(launcher_script)
            
            # 執行子進程
            result = subprocess.run([python_executable, temp_file], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    text=True)
            
            # 刪除臨時腳本
            try:
                os.remove(temp_file)
            except:
                pass
                
            if result.returncode != 0:
                print("子進程執行失敗:")
                print(f"標準輸出: {result.stdout}")
                print(f"錯誤輸出: {result.stderr}")
                messagebox.showerror("錯誤", f"模組執行失敗:\n{result.stderr}")
            
            # 重新顯示主視窗
            self.root.deiconify()
            self.status_var.set("就緒")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("錯誤", f"執行子進程時發生錯誤:\n{str(e)}")
            self.status_var.set("就緒")
            # 確保主視窗被重新顯示
            self.root.deiconify()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    # 設定終端機編碼
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    # 啟動應用程式
    app = TaekwondoAnalysisSystem()
    app.run()