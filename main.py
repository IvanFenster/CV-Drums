import cv2
import mediapipe as mp
import numpy as np


# Stackoverflow: https://ru.stackoverflow.com/questions/950520/opencv-%D0%BD%D0%B0%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5-%D0%B8%D0%B7%D0%BE%D0%B1%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D0%B9
def compbine_img(img1, img2):
    brows, bcols = img1.shape[:2]
    rows, cols, channels = img2.shape

    # Ниже я изменил roi, чтобы картинка выводилась посередине, а не в левом верхнем углу
    roi = img1[int(brows / 20 * 13) - int(rows / 2):int(brows / 20 * 13) + int(rows / 2), int(bcols / 2) - int(cols / 2):int(bcols / 2) + int(cols / 2)]

    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    dst = cv2.add(img1_bg, img2_fg)
    img1[int(brows / 20 * 13) - int(rows / 2):int(brows / 20 * 13) + int(rows / 2), int(bcols / 2) -
                                                                        int(cols / 2):int(bcols / 2) + int(
        cols / 2)] = dst

    return img1


#создаем детектор
handsDetector = mp.solutions.hands.Hands(max_num_hands=6)
cap = cv2.VideoCapture(0)
drum = cv2.imread("Барабан.png")
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

    cv2.imshow("Hands", compbine_img(res_image, drum))

# освобождаем ресурсы
handsDetector.close()