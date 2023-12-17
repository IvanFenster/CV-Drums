import sys

import cv2
import mediapipe as mp
import numpy as np
import time

import pygame


# Stackoverflow: https://ru.stackoverflow.com/questions/950520/opencv-%D0%BD%D0%B0%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5-%D0%B8%D0%B7%D0%BE%D0%B1%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D0%B9
def compbine_img(img1, img2):
    brows, bcols = img1.shape[:2]
    rows, cols, channels = img2.shape
    # Ниже я изменил roi, чтобы картинка выводилась посередине, а не в левом верхнем углу
    roi = img1[int(brows / 2) - int(rows / 2):int(brows / 2) + int(rows / 2), int(bcols / 2) - int(cols / 2):int(bcols / 2) + int(cols / 2)]

    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    dst = cv2.add(img1_bg, img2_fg)
    img1[int(brows / 2) - int(rows / 2):int(brows / 2) + int(rows / 2), int(bcols / 2) -
                                                                        int(cols / 2):int(bcols / 2) + int(
        cols / 2)] = dst
    cv2.imwrite('res.jpg', img1)

    return img1


def tom_sound():
    tom.play()


pygame.init()
clock = pygame.time.Clock()

tom = pygame.mixer.Sound('audio/tom-37.mp3')

#создаем детектор
handsDetector = mp.solutions.hands.Hands(max_num_hands=6)
cap = cv2.VideoCapture(0)
drum = cv2.imread("image/Background.png")

x_pos = 0
y_pos = 0

tom_flag = False
k = 1
last = time.time()
while(cap.isOpened()):
    clock.tick(60)
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

            prev_x = x_pos
            prev_y = y_pos
            x_pos = i.landmark[8].x
            y_pos = i.landmark[8].y


            if 0.1559 < x_pos < 0.4291 and 0.648 < y_pos < 1:
                now = time.time()

                if tom_flag == False and y_pos > prev_y:
                    if now - last > 0.2:
                        tom_flag = True
                        tom_sound()
                        last = time.time()

            else:
                tom_flag = False

                k += 1



    # переводим в BGR и показываем результат
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)

    cv2.imshow("Drums", compbine_img(res_image, drum))

# освобождаем ресурсы
handsDetector.close()
pygame.quit()