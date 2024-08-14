import numpy as np
import cv2 as cv
import time
# 视频文件路径
video_path = 'test.mp4'
cap = cv.VideoCapture(video_path)

# cap = cv.VideoCapture(0)

# ShiTomasi角点检测参数
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
# Lucas-Kanade光流法参数
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


while True:
    # 读取第一帧
    ret, old_frame = cap.read()
    if not ret:
        print("无法读取视频")
        exit()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, **feature_params)
    # 创建一个蒙版用于绘图
    mask = np.zeros_like(old_frame)
    # time.sleep(1)
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 计算光流
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # 选择状态为1的点，即成功跟踪的点
    good_new = p1[st==1]
    good_old = p0[st==1]

    # 计算所有特征点的水平位移
    horizontal_displacements = [new[0] - old[0] for new, old in zip(good_new, good_old)]
    # # 计算所有特征点垂直位移
    # vertical_displacements = [new[1] - old[1] for new, old in zip(good_new, good_old)]
    # # 计算平均垂直位移
    # if len(vertical_displacements) > 0:
    #     average_dy = np.mean(vertical_displacements)
    #     if average_dy > 0:
    #         direction = "向下移动"
    #     elif average_dy < 0:
    #         direction = "向上移动"
    #     else:
    #         direction = "未检测到垂直移动"
    #     print(f"检测到的总体移动方向: {direction}")
    # 计算平均位移
    if len(horizontal_displacements) > 0:
        average_dx = np.mean(horizontal_displacements)
        if average_dx > 0:
            direction = "向右移动"
        elif average_dx < 0:
            direction = "向左移动"
        else:
            direction = "未检测到水平移动"
        print(f"检测到的总体移动方向: {direction}")

    # 绘制跟踪线
    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        # print("c: ", c)
        # print("d: ", d)
        # print("a: ", a)
        # print("b: ", b)
        a, b, c, d = int(a), int(b), int(c), int(d)
        cv.line(mask, (c, d), (a, b), (0, 255, 0), 2)
        frame = cv.circle(frame, (a, b), 5, (0, 255, 0), -1)

    # 显示带有跟踪线的帧
    frame = cv.add(frame, mask)
    cv.imshow('frame', frame)

    # 更新上一帧和特征点
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

cv.destroyAllWindows()
cap.release()