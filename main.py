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


def bass_sound():
    bass.play()


def ride_sound():
    ride.play()

pygame.init()
clock = pygame.time.Clock()

tom = pygame.mixer.Sound('audio/tom_sound.mp3')
bass = pygame.mixer.Sound('audio/bass_sound1.mp3')
ride = pygame.mixer.Sound('audio/ride_sound.mp3')
crash = pygame.mixer.Sound('audio/crash_sound.wav')

#создаем детектор
handsDetector = mp.solutions.hands.Hands(max_num_hands=6)
cap = cv2.VideoCapture(1)
drum = cv2.imread("image/Background2.png")

x_pos_l = 0
y_pos_l = 0
x_pos_r = 0
y_pos_r = 0

tom_flag_r = False
tom_flag_l = False

bass_flag_r = False
bass_flag_l = False

ride_flag_r = False
ride_flag_l = False

k = 1
last = time.time()

TIME_LIMIT = 0.1

while(cap.isOpened()):
    handsType = []
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
        # Определяем правая или левая рука
        for i in results.multi_handedness:
            handType = i.classification[0].label
            handsType.append(handType)
        for i in range(len(results.multi_hand_landmarks)):
            hand = results.multi_hand_landmarks[i]
            now_hand_type = handsType[i]
            mp.solutions.drawing_utils.draw_landmarks(flippedRGB,
                                              hand, mp.solutions.hands.HAND_CONNECTIONS,
                                              mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                                              mp.solutions.drawing_styles.get_default_hand_connections_style())

            # Если правая
            if now_hand_type == "Right":

                prev_x_r = x_pos_r
                prev_y_r = y_pos_r
                x_pos_r = hand.landmark[8].x
                y_pos_r = hand.landmark[8].y

                # TOM
                if prev_y_r < 0.648 < y_pos_r and 0.1559 < x_pos_r < 0.4291:
                    now = time.time()

                    if tom_flag_r == False and y_pos_r > prev_y_r:
                        if now - last > TIME_LIMIT:
                            tom_flag_r = True
                            tom_sound()
                            last = time.time()
                elif 0.1559 < x_pos_r < 0.4291:
                    tom_flag_r = False
                    k += 1

                # BASS
                if prev_y_r < 0.648 < y_pos_r and 0.525 < x_pos_r < 0.7953:
                    now = time.time()

                    if bass_flag_r == False and y_pos_r > prev_y_r:
                        if now - last > TIME_LIMIT:
                            bass_flag_r = True
                            bass_sound()
                            last = time.time()
                elif 0.525 < x_pos_r < 0.7953:
                    bass_flag_r = False
                    k += 1

                # RIDE
                if 0.7322 < x_pos_r < 0.9593 and 0.0611 < y_pos_r < 0.267:
                    now = time.time()

                    if ride_flag_r == False:
                        if now - last > TIME_LIMIT:
                            ride_flag_r = True
                            ride_sound()
                            last = time.time()
                else:
                    ride_flag_r = False
                    k += 1


            # Если левая
            elif now_hand_type == "Left":

                prev_x_l = x_pos_l
                prev_y_l = y_pos_l
                x_pos_l = hand.landmark[8].x
                y_pos_l = hand.landmark[8].y

                # TOM
                if prev_y_l < 0.648 < y_pos_l and 0.1559 < x_pos_l < 0.4291 :
                    now = time.time()

                    if tom_flag_l == False and y_pos_l > prev_y_l:
                        if now - last > TIME_LIMIT:
                            tom_flag_l = True
                            tom_sound()
                            last = time.time()
                elif 0.1559 < x_pos_l < 0.4291:
                    tom_flag_l = False
                    k += 1

                # BASS
                if prev_y_l < 0.648 < y_pos_l and 0.525 < x_pos_l < 0.7953:
                    now = time.time()

                    if bass_flag_l == False and y_pos_l > prev_y_l:
                        if now - last > TIME_LIMIT:
                            bass_flag_l = True
                            bass_sound()
                            last = time.time()
                elif 0.525 < x_pos_l < 0.7953:
                    bass_flag_l = False
                    k += 1

                # RIDE
                if 0.7322 < x_pos_l < 0.9593 and 0.0611 < y_pos_l < 0.267:
                    now = time.time()

                    if ride_flag_l == False:
                        if now - last > TIME_LIMIT:
                            ride_flag_l = True
                            ride_sound()
                            last = time.time()
                else:
                    ride_flag_l = False
                    k += 1

            else:
                tom_flag_r = False
                tom_flag_l = False
                bass_flag_r = False
                bass_flag_l = False


    # переводим в BGR и показываем результат
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)

    cv2.imshow("Drums", compbine_img(res_image, drum))

# освобождаем ресурсы
handsDetector.close()
pygame.quit()