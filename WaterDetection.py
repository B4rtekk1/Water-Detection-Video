import cv2
import numpy as np

input_path = 'mono_lake.mp4'
output_path = 'detected_water.mp4'

cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 15
frame_delay = int(600 / fps)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    start_time = cv2.getTickCount()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask = cv2.bitwise_or(mask_green, mask_black)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    out.write(result)

    cv2.imshow('Original', frame)
    cv2.imshow('Detected Water', result)

    end_time = cv2.getTickCount()
    time_elapsed = (end_time - start_time) / cv2.getTickFrequency()
    wait_time = max(1, frame_delay - int(time_elapsed * 1000))
    
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
