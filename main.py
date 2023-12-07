import cv2
import mediapipe as mp
import numpy as np

#создаем детектор
handsDetector = mp.solutions.hands.Hands()
# загружаем изображение
image = cv2.imread("Chord C.jpg")
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
