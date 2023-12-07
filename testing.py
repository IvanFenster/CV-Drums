import cv2
import mediapipe as mp
import numpy as np

#создаем детектор
handsDetector = mp.solutions.hands.Hands(max_num_hands=6)
cap = cv2.VideoCapture("Игра 2.mov")
while(cap.isOpened()):
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)
    # переводим его в формат RGB для распознавания
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    # Распознаем
    results = handsDetector.process(flippedRGB)
    # Рисуем распознанное, если распозналось

    if results.multi_hand_landmarks is not None:
        for i in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(flippedRGB,
                                                  i, mp.solutions.hands.HAND_CONNECTIONS,
                                                  mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                                                  mp.solutions.drawing_styles.get_default_hand_connections_style())

    # переводим в BGR и показываем результат
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)

# освобождаем ресурсы
handsDetector.close()