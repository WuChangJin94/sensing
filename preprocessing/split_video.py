import cv2
import os

# 讀取影片
input_video_path = 'labels_output_0522/aerator_flow_img.mp4'  # 替換為你的影片路徑
output_video_path = 'labels_kmeans_0522/aerator_flow_img_kmeans.mp4'  # 指定輸出的影片路徑

if not os.path.exists('sample_files_0522'):
    os.makedirs('sample_files_0522', exist_ok=True)

# 打開影片檔案
cap = cv2.VideoCapture(input_video_path)

# 檢查影片是否成功打開
if not cap.isOpened():
    print("Error opening video file")
    exit()

# 取得影片的基本屬性
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 設定輸出的影片檔案
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width // 2, frame_height // 2))

while True:
    ret, frame = cap.read()
    
    # 如果讀不到畫面，跳出迴圈
    if not ret:
        break
    
    # 分割畫面成上下兩部分
    buttom_half = frame[frame_height // 2:, frame_width // 2:, :]
    
    # 將上半部分寫入輸出影片檔案
    out.write(buttom_half)

    # 顯示處理中的畫面 (可選)
    cv2.imshow('Top Half', buttom_half)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放所有資源
cap.release()
out.release()
cv2.destroyAllWindows()
