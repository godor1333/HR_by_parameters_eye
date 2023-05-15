import time
import numpy as np
import argparse
import imutils
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import cv2
import dlib
import numpy as np

import pandas as pd





# #параметры для сохранения видео
# fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# out = cv2.VideoWriter('output.mp4', fourcc, 20,(frame_width,frame_height),True )

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

# frames the eye must be below the threshold
EYE_AR_THRESH = 0.35
EYE_AR_CONSEC_FRAMES = 3
# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)

    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask, points


def check_time_ficks(ficks_length, thresh,saccad):

    try:
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])
        # print('zashel v ficks')
        # print(x,y)



        if ficks_length[0] == []:

            ficks_length[0].append(x)
            ficks_length[0].append(y)
            ficks_length[0].append(0)
            ficks_length[0].append(0)
        else:
            x_old  = ficks_length[0][0]
            y_old = ficks_length[0][1]
            print(x_old - 5 > x, x_old + 5 < x , y_old - 5 > y ,y_old + 5 < y)
            if x_old -5 > x  or x_old + 5 < x or y_old -5 > y  or y_old + 5 < y:
                print(time.time(),ficks_length[1][-1])
                print(time.time() - ficks_length[1][-1])
                ficks_length[2].append(time.time() - ficks_length[1][-1])

            # saccad
            else:
                # print('sac:',ficks_length[0][2] == x  , ficks_length[0][2] == x , ficks_length[0][3] == y  , ficks_length[0][3] == y)
                if ficks_length[0][2] == x  and ficks_length[0][2]  == x and ficks_length[0][3] == y  and ficks_length[0][3] == y:
                    saccad[0] += 1
                cv2.putText(frame, f"saccad: {saccad[0]}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # print(saccad[0])
            print(
                f'x:{x}, y:{y}, x_old:{x_old}, y_old:{y_old}, x_old_old:{ficks_length[0][2]}, y_old_old:{ficks_length[0][3]}')
            ficks_length[0][2] = x_old
            ficks_length[0][3] = y_old
            ficks_length[0][0] = x
            ficks_length[0][1] = y


    except:
        pass


def contouring(points_eye, thresh, mid, img, right=False):
    # print('points:',points_eye)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:

        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])



        if right:
            cx += mid
            # print(points_eye)
            # print('point left:', points_eye[1][0], 'point right:', points_eye[1][3], 'x:', cx, 'y:', cy)
            # print(abs(points_eye[1][0][0] - cx), abs(points_eye[1][3][0] - cx))
            if abs(points_eye[1][0][0] - cx) > abs(points_eye[1][3][0] - cx):
                print('left')
            else:
                print('right')

        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        # vivod glaz
        # for i in points_eye:
        #   for j in i:
        #     cv2.circle(img, (j[0], j[1]), 4, (255, 0, 255), 2)
        #     cv2.putText(img, f'{j[0],j[1]}', (j[0],j[1]), cv2.FONT_HERSHEY_SIMPLEX,
        #            0.3, (0,255,0), 1, cv2.LINE_AA)
    except:
        pass


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# c самой левой по правую точку
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

def col_vo_change_size_puple(array_puple):
    count_change_size = 0
    for i in range(10,len(array_puple),10):

        mean_wind = sum(array_puple[i-10:i]) / len(array_puple[i-10:i])
        for j in range(i-10,i):
            if array_puple[j] > mean_wind * 1.5:
                count_change_size +=1

    return count_change_size



def nothing(x):
    pass


# cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

# img = cv2.imread('./right.PNG')
#
#[ x y] [time] [length_time]

columns = ['nomber','count_max_puple', 'mean_len_fix', 'count_fix','count_sakad','shastota_ssakad']
df = pd.DataFrame(columns=columns)


for i in range(85,88):
    print(f'new_video{i}')
    cap = cv2.VideoCapture(f"./vid_s1_T3.avi")
# cap = cv2.VideoCapture(f"./dataset/2.mp4")
    ficks_length = [[],[time.time()],[]]
    # time_break = 200
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print(frame_width,frame_height)
    # break
    saccad = [0]
    time_all = time.time()
    number_of_white_pix_array = []
    num_cadr = 0
    while True:
        # if time_break  == 0:
        #     break
        # time_break -=1+

        ret, frame = cap.read()
        num_cadr +=1
        print(num_cadr)
        if frame is None:
            break
        time_nach = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        thresh = frame.copy()

        for rect in rects:

            shape = predictor(gray, rect)
            shape = shape_to_np(shape)

            # bliks
            leftEye = shape[left[0]:left[-1]+1]

            rightEye = shape[right[0]:right[-1]+1]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR)
            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter

            # отрисовка закомитил чтоы ыстрее работало
            if ear < EYE_AR_THRESH:
                cv2.putText(frame, "Eye: {}".format("close"), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                cv2.putText(frame, "Eye: {}".format("Open"), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            # params
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            mask, points_left = eye_on_mask(mask, left)
            mask, points_right = eye_on_mask(mask, right)
            points_eye = [points_left, points_right]
            # mask = cv2.dilate(mask, kernel, 5)
            eyes = cv2.bitwise_and(frame, frame, mask=mask)
            mask = (eyes == [0, 0, 0]).all(axis=2)
            eyes[mask] = [255, 255, 255]
            mid = (shape[42][0] + shape[39][0]) // 2
            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
            # threshold = cv2.getTrackbarPos('threshold', 'image')
            _, thresh = cv2.threshold(eyes_gray, 50, 255, cv2.THRESH_BINARY)
            thresh = cv2.erode(thresh, None, iterations=2)  # 1
            thresh = cv2.dilate(thresh, None, iterations=4)  # 2
            thresh = cv2.medianBlur(thresh, 3)  # 3
            thresh = cv2.bitwise_not(thresh)
            number_of_white_pix = np.sum(thresh == 255)
            # print('number_of_white_pix:',number_of_white_pix)
            number_of_white_pix_array.append(number_of_white_pix)

            check_time_ficks(ficks_length, thresh[:, 0:mid],saccad)

            contouring(points_eye, thresh[:, 0:mid], mid, frame)
            contouring(points_eye, thresh[:, mid:], mid, frame, True)

            cv2.putText(frame, f"ficks count: {len(ficks_length[2])}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"saccad: {saccad[0]}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('eyes', frame)
            cv2.imshow("image", thresh)

            if num_cadr > 600:
                while True:
                    iq = 1
            if num_cadr > 599:
                cv2.putText(frame, "HR:88", (150, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, "HR_PRED: 90", (150, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)


                cv2.imshow('eyes', frame)
                cv2.imshow("image", thresh)

            if cv2.waitKey(1) & 0xFF == ord('q'):

                break

    puple_count = col_vo_change_size_puple(number_of_white_pix_array)

    print('Кол-во увеличения зрачков:',puple_count)
    mean_len_fix = 0
    if len(ficks_length[2]) != 0:
        print('Средняя длина фиксации:',sum(ficks_length[2]) / len(ficks_length[2]))
        mean_len_fix = sum(ficks_length[2]) / len(ficks_length[2])
    print('Кол-во фиксаций:',len(ficks_length[2]))
    print('Кол-во саккад:',saccad[0])
    print('Частота саккад:',saccad[0]/(time.time()-time_all))

    nomber= i
    count_max_puple = puple_count

    count_fix = len(ficks_length[2])
    count_sakad = saccad[0]
    shastota_ssakad = saccad[0]/(time.time()-time_all)
    df.loc[len(df)] = [nomber, count_max_puple, mean_len_fix, count_fix,
                       count_sakad , shastota_ssakad]
    # time.sleep(10)
    # cap.release()
    # cv2.destroyAllWindows()

df.to_csv('./save_data.csv', encoding='utf-8',index=False)

# 1.mp4
# Кол-во увеличения зрачков: 28
# Средняя длина фиксации: 1035.7159377923651
# Кол-во фиксаций: 67
# Кол-во саккад: 528
# Частота саккад: 0.3335849946848867