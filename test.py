import cv2

def analyze_video_movement(video_path):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not video.isOpened():
        print("无法打开视频文件")
        return

    # 读取前两帧
    ret, prev_frame = video.read()
    ret, curr_frame = video.read()

    # 创建 SIFT 特征提取器
    sift = cv2.xfeatures2d.SIFT_create()

    while True:
        # 计算特征点和描述符
        prev_keypoints, prev_descriptors = sift.detectAndCompute(prev_frame, None)
        curr_keypoints, curr_descriptors = sift.detectAndCompute(curr_frame, None)

        # 特征匹配
        matcher = cv2.DescriptorMatcher_create(cv2.DIST_L2)
        matches = matcher.match(prev_descriptors, curr_descriptors)

        # 计算位移并确定方向
        displacements = []
        for match in matches:
            prev_pt = prev_keypoints[match.queryIdx].pt
            curr_pt = curr_keypoints[match.trainIdx].pt
            displacement = curr_pt[0] - prev_pt[0]
            displacements.append(displacement)

        average_displacement = sum(displacements) / len(displacements)
        if average_displacement > 0:
            print("物体向右移动")
        elif average_displacement < 0:
            print("物体向左移动")
        else:
            print("物体未移动")

        # 更新帧
        prev_frame = curr_frame
        ret, curr_frame = video.read()

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    video.release()
    cv2.destroyAllWindows()

# 替换为你的视频文件路径
video_path = 'test.mp4'
analyze_video_movement(video_path)