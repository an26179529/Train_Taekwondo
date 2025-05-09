# 跆拳道綜合動作偵測系統

## 專案概述

跆拳道綜合動作偵測系統是一套利用電腦視覺技術自動偵測和評估跆拳道基本動作的工具。本系統能夠辨識並評估多種跆拳道動作，包括雙手刀、前踢、山形防禦、正拳、旋踢和側踢等，提供即時的視覺回饋和動作分析報告。

## 系統特色

- **多種動作偵測**：支援六種基本跆拳道動作的偵測與分析
- **即時視覺回饋**：在影片處理過程中顯示骨架和關鍵點，幫助使用者理解動作評估
- **詳細分析報告**：生成含有時間戳記、動作正確性和相關參數的 Excel 報表
- **使用者友善介面**：直覺式的圖形使用者介面，操作簡便
- **高精度偵測**：採用 MediaPipe 姿勢偵測技術，提供準確的人體姿勢追蹤

## 環境需求

- Python 3.7 或更高版本
- Windows、macOS 或 Linux 作業系統
- 鏡頭或預先錄製的影片檔案
- 建議使用 GPU 以獲得更好的處理性能（非必須）

## 安裝指南

1. 確保您的系統已安裝 Python 3.7+
2. 下載或 clone 本專案至本地端
3. 安裝所需的套件：

```bash
pip install -r requirements.txt
```

## 使用方法

### 啟動主程式

執行主程式以開啟跆拳道綜合動作偵測系統：

```bash
python app.py
```

### 選擇偵測動作

1. 在主介面中選擇您想要偵測的動作類型（雙手刀、前踢、山形防禦等）
2. 點選對應動作卡片上的「選擇此動作」按鈕

### 設定輸入與輸出

1. 在彈出的模組視窗中，選擇影片來源（可以是預先錄製的影片或即時鏡頭）
2. 指定輸出資料夾以儲存分析結果
3. 針對某些動作（如山形防禦），可能需要輸入額外參數（例如肩寬）

### 開始分析

1. 點擊「開始處理」按鈕開始進行動作偵測
2. 系統會顯示處理進度和即時視覺化結果
3. 處理完成後，系統會自動開啟結果資料夾

### 查看結果

1. 處理後的影片：包含骨架和評估資訊的視覺化結果
2. Excel 報表：詳細的動作分析資料，包括時間點、角度數據和正確性評估

## 模組說明

本系統包含以下主要模組：

### 1. 雙手刀偵測模組 (double.py)

偵測跆拳道雙手刀技術動作，分析雙臂角度和動作完成度。

主要特點：
- 偵測預備姿勢並追蹤整個動作過程
- 評估前臂和後臂的角度是否在正確範圍內
- 生成包含時間點、前手/後手資訊和正確性評估的報告

### 2. 前踢偵測模組 (front.py)

偵測跆拳道前踢動作，分析腿部高度和腳的角度。

主要特點：
- 追蹤踢腿的軌跡和高度
- 評估腳的角度是否在正確範圍內
- 支援左右腳的前踢偵測

### 3. 山形防禦偵測模組 (mountain.py)

偵測山形防禦姿勢，評估手臂角度和肩膀高度差。

主要特點：
- 分析雙臂角度和肩膀平衡度
- 需要肩寬參數以計算比例
- 評估姿勢的穩定性和正確性

### 4. 正拳偵測模組 (pounch.py)

偵測正拳出擊動作，分析手臂速度和角度。

主要特點：
- 偵測拳擊速度和出拳力道
- 評估拳擊高度是否正確
- 支援左右手的正拳偵測

### 5. 旋踢偵測模組 (revolve.py)

偵測旋踢動作，分析腿部旋轉和支撐腳的穩定性。

主要特點：
- 追蹤踢腿的旋轉軌跡
- 評估支撐腳的穩定性
- 分析旋踢的完成度和正確性

### 6. 側踢偵測模組 (side.py)

偵測側踢動作，分析腿部高度和支撐腳的穩定性。

主要特點：
- 檢測腿部是否達到頭部高度
- 評估支撐腳是否保持穩定
- 分析踢擊完成度

## 設定檔說明

系統設定檔 `config.ini` 包含多項可自訂參數，包括：

- **系統設定**：介面大小、顏色、字體等
- **偵測設定**：MediaPipe 模型參數、輸入輸出設置
- **各動作模組設定**：每種動作的特定參數，如角度閾值、冷卻時間等

您可以根據需要調整這些參數以優化系統性能。

## 疑難排解

- **偵測不準確**：嘗試調整光線條件，確保場景光線充足且均勻
- **程式啟動失敗**：檢查是否安裝了所有必要的套件，特別是 MediaPipe 和 OpenCV
- **影片處理緩慢**：考慮降低影片解析度或在具有 GPU 加速的環境中運行

## 未來發展

- 增加更多跆拳道動作的偵測支援
- 整合深度學習模型以提高偵測精度
- 增加多人同時偵測的功能
- 開發網頁或移動應用程式版本

## 授權聲明

本專案採用 MIT 授權條款。

## 聯絡資訊

如有任何問題或建議，請透過以下方式聯絡：

- 電子郵件：an26179529@gmail.com

## 致謝

- MediaPipe 團隊提供的優秀姿勢偵測框架
- OpenCV 社群的開源貢獻
- 所有參與測試和提供反饋的跆拳道教練和學生