import cv2
import mediapipe as mp
import numpy as np

image = cv2.imread("Chords/Chord Am.jpeg")


#создаем детектор
handsDetector = mp.solutions.hands.Hands()
# Отражаем по горизонтали если надо
# (Mediapipе распознает именно зеркальное селфи)
flipped = np.fliplr(image)
# переводим его в формат RGB для распознавания
flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
# Распознаем
results = handsDetector.process(flippedRGB)
# Рисуем распознанное, если распозналось
if results.multi_hand_landmarks is not None:
    mp.solutions.drawing_utils.draw_landmarks(flippedRGB, 
       results.multi_hand_landmarks[0])
# Отражаем обратно, переводим в BGR и показываем результат
res_image = cv2.cvtColor(np.fliplr(flippedRGB), cv2.COLOR_RGB2BGR)
print(results.multi_handedness)
cv2.imshow("Hands", res_image)
cv2.waitKey(0)

# освобождаем ресурсы
handsDetector.close()

# Контуры
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
avg = np.mean(blurred)

thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]
contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)


