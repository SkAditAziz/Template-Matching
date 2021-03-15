import numpy as np
import math
import time
import cv2
import matplotlib.pyplot as plt

drive_path = 'drive/My Drive/Colab Notebooks/Template Matching/'
count = 0
exTotalFrame = []
logTotalFrame = []
exTotalTime = []
logTotalTime = []

iteration = [8,16,32,64]

def getRGBFrame (fi,di,dj,M,N):
        
        redFrame = cv2.cvtColor(fi, cv2.COLOR_GRAY2BGR)

        for i in range(di - 1, di + M + 1):
            redFrame[i][dj - 1][0] = np.int8(0)
            redFrame[i][dj - 1][1] = np.int8(0)
            redFrame[i][dj - 1][2] = np.int8(255)

            redFrame[i][dj + N][0] = np.int8(0)
            redFrame[i][dj + N][1] = np.int8(0)
            redFrame[i][dj + N][2] = np.int8(255)
        for j in range(dj - 1, minimum_j + N + 1):
            redFrame[di - 1][j][0] = np.int8(0)
            redFrame[di - 1][j][1] = np.int8(0)
            redFrame[di - 1][j][2] = np.int8(255)

            redFrame[di + M][j][0] = np.int8(0)
            redFrame[di + M][j][1] = np.int8(0)
            redFrame[di + M][j][2] = np.int8(255)

        return redFrame;

cap = cv2.VideoCapture(drive_path+'input.mov')
video_frames = []
frame_per_sec = cap.get(cv2.CAP_PROP_FPS)

while True:
    success, frame = cap.read()
    if not success:
        break
    graycodedframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    video_frames.append(graycodedframe)
    count += 1

cap.release()

reference_image = cv2.imread(drive_path + 'reference.jpg')
gray_reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

def ex_search(grayRefImg, frames, frame_rate, p):
    cnt = 0
    M = grayRefImg.shape[0]
    N = grayRefImg.shape[1]

    I = frames[0].shape[0]
    J = frames[0].shape[1]

    output_frames = []
    matched_x_coordinate = matched_y_coordinate = 0
    for k in range(len(frames)):
        frame_img = frames[k]
        d_min = float('inf')
        minimum_i = minimum_j = -1
        if k == 0:
            for i in range(I - M):
                for j in range(J - N):
                    ref_image = np.array(grayRefImg)
                    frame_image = np.array(frame_img)
                    d = np.sum(np.square(np.absolute((ref_image.astype(np.int64)) - (frame_image[i: i + M, j: j + N].astype(np.int64)))))
                    cnt += 1
                    if d < d_min:
                        d_min = d
                        minimum_i = i
                        minimum_j = j
            matched_y_coordinate = minimum_i
            matched_x_coordinate = minimum_j
        else:
            for i in range(matched_y_coordinate - p, matched_y_coordinate + p + 1):
                for j in range(matched_x_coordinate - p, matched_x_coordinate + p + 1):
                    if i < 0 or j < 0 or i >= I - M or j >= J - N:
                        continue
                    ref_image = np.array(grayRefImg)
                    frame_image = np.array(frame_img)
                    d = np.sum(np.square(np.absolute((ref_image.astype(np.int64)) - (frame_image[i: i + M, j: j + N].astype(np.int64)))))
                    cnt += 1
                    if d < d_min:
                        d_min = d
                        minimum_i = i
                        minimum_j = j
            matched_y_coordinate = minimum_i
            matched_x_coordinate = minimum_j

        output_frames.append(getRGBFrame(frame_img,minimum_i,minimum_j,M,N))

        #video = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc(*'DIVX'), 60, (width, height))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    outputvideo = cv2.VideoWriter(drive_path + 'ex' + str(p) + '.avi', fourcc, frame_rate, (J, I))

    for output_frame in output_frames:
        outputvideo.write(output_frame)
    outputvideo.release()

    totalframes = (cnt*1.0)/len(frames)

    return totalframes


def log_search(grayRefImg, frames, frame_rate, p):
    cnt = 0
    M = grayRefImg.shape[0]
    N = grayRefImg.shape[1]

    I = frames[0].shape[0]
    J = frames[0].shape[1]

    output_frames = []
    matched_x_coordinate = matched_y_coordinate = 0
    minimum_i = minimum_j = -1
    for k in range(len(frames)):
        frame_img = frames[k]
        if k == 0:
            d_min = float('inf')
            minimum_i = minimum_j = -1
            for i in range(I - M):
                for j in range(J - N):
                    ref_image = np.array(grayRefImg)
                    frame_image = np.array(frame_img)
                    d = np.sum(np.square(np.absolute((ref_image.astype(np.int64)) - (frame_image[i: i + M, j: j + N].astype(np.int64)))))
                    cnt += 1
                    if d < d_min:
                        d_min = d
                        minimum_i = i
                        minimum_j = j
            matched_y_coordinate = minimum_i
            matched_x_coordinate = minimum_j
        else:
            p_tmp = p
            spacing = math.pow(2, math.ceil(math.log(p, 2)) - 1)
            while spacing >= 1:
                d_min = float('inf')
                minimum_i = minimum_j = -1
                for i in [matched_y_coordinate - p_tmp, matched_y_coordinate, matched_y_coordinate + p_tmp]:
                    for j in [matched_x_coordinate - p_tmp, matched_x_coordinate, matched_x_coordinate + p_tmp]:
                        if i < 0 or j < 0 or i >= I - M or j >= J - N:
                            continue
                        ref_image = np.array(grayRefImg)
                        frame_image = np.array(frame_img)
                        d = np.sum(np.square(np.absolute((ref_image.astype(np.int64)) - (frame_image[i: i + M, j: j + N].astype(np.int64)))))
                        cnt += 1
                        if d < d_min:
                            d_min = d
                            minimum_i = i
                            minimum_j = j
                matched_y_coordinate = minimum_i
                matched_x_coordinate = minimum_j
                p_tmp = round(p_tmp / 2.0)
                spacing /= 2

        output_frames.append(getRGBFrame(frame_img,minimum_i,minimum_j,M,N))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    outputvideo = cv2.VideoWriter(drive_path + 'log' + str(p) + '.avi', fourcc, frame_rate, (J, I))

    for output_frame in output_frames:
        outputvideo.write(output_frame)
    outputvideo.release()

    totalframes = (cnt*1.0)/len(frames)

    return totalframes

def hi_search(grayRefImg, frames, frame_rate, p):
    cnt = 0
    M = grayRefImg.shape[0]
    N = grayRefImg.shape[1]

    I = frames[0].shape[0]
    J = frames[0].shape[1]

    output_frames = []
    matched_x_coordinate = matched_y_coordinate = 0
    minimum_i = minimum_j = -1
    d_min = float('inf')
    for i in range(I - M):
                for j in range(J - N):
                    ref_image = np.array(grayRefImg)
                    frame_image = np.array(frame_img)
                    d = np.sum(np.square(np.absolute((ref_image.astype(np.int64)) - (frame_image[i: i + M, j: j + N].astype(np.int64)))))
                    cnt += 1
                    if d < d_min:
                        d_min = d
                        minimum_i = i
                        minimum_j = j
                    matched_y_coordinate = minimum_i
                    matched_x_coordinate = minimum_j
       
    output_frames.append(getRGBFrame(frame_img,minimum_i,minimum_j,M,N))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    outputvideo = cv2.VideoWriter(drive_path + str(p) + '.avi', fourcc, frame_rate, (J, I))

    for output_frame in output_frames:
        outputvideo.write(output_frame)
    outputvideo.release()

    totalframes = (cnt*1.0)/len(frames)

    return totalframes


while True:
    
    if p in iteration :
        
        start_time = time.time()
        print(start_time)
        res = ex_search(gray_reference_image, video_frames, frame_per_sec, p)
        exTotalFrame.append (res)
        end_time = time.time()
        t = round((end_time - start_time) / 60, 3)
        exTotalTime.append(t)

        start_time = time.time()
        print(start_time)
        res = log_search(gray_reference_image, video_frames, frame_per_sec, p)
        exTotalFrame.append (res)
        end_time = time.time()
        t = round((end_time - start_time) / 60, 3)
        exTotalTime.append(t)


print("values of p:")
print(iteration)
print("execution time for exhausted search:")
print(exTotalTime)
print("execution time for logarithm search:")
print(logTotalTime)

plt.plot(iteration, exTotalFrame, label='Exhaustive Search')
plt.title('Exhaustive Search')
plt.xlabel('p')
plt.ylabel('Total frame')
plt.show()

plt.plot(iteration, logTotalFrame, label='2D Logarithmic Search')
plt.title('2D Logarithmic Search')
plt.xlabel('p')
plt.ylabel('Total frame')
plt.show()