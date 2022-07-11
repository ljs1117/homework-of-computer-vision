"""
1851754 李玖思
自动驾驶机器视觉作业 图像匹配检测作业

Window
VSCode 1.67
Python 3.6.13

OpenCV 3.4.1
Numpy

"""
import cv2
import numpy as np


def stitch(imgA, imgB_warp):
    '''图像裁剪和拼接'''

    # 注意本函数重叠部分取imgA的灰度，相当于线性叠加时imgA的权重为1，imgB_warp的权重为0
    # imgB_warp整体提亮

    # 以imgA的高为目标图像的高
    height = int(imgA.shape[0])
    # 以imgB_warp最右的非零像素点所在的宽为目标图像的宽
    indexNonZero = np.nonzero(imgB_warp)
    width = int(max(indexNonZero[1]))
    dst = np.zeros((height, width), dtype=np.float32)
    # 将imgB_warp最右非零像素点以左的部分复制给目标图像
    dst[:, :] = imgB_warp[0:height, 0:width]
    # 将imgA复制给目标图像
    dst[0:height, 0:int(imgA.shape[1])] = imgA[:, :]
    # 目标图像中属于imgB_warp但不属于重叠部分的部分提亮
    for i in range(height):
        for j in range(imgA.shape[1], width):
            dst[i, j] = 1.3*imgB_warp[i, j]

    return dst


def main():
    '''主函数'''

    # 输出矩阵的有效位数为3
    np.set_printoptions(precision=2)

    # 读入图像
    imgA = cv2.imread('hw4/A.png', cv2.IMREAD_GRAYSCALE)
    imgB = cv2.imread('hw4/B.png', cv2.IMREAD_GRAYSCALE)

    # 用SIFT寻找特征点和描述子
    sift = cv2.xfeatures2d.SIFT_create()
    siftKpA, siftDesA = sift.detectAndCompute(imgA, None)
    siftKpB, siftDesB = sift.detectAndCompute(imgB, None)

    # BruteForce匹配特征点
    siftbf = cv2.BFMatcher()
    siftMatches = siftbf.knnMatch(siftDesA, siftDesB, k=2)  # 返回2个匹配（即最匹配和次匹配）
    good_sift = []

    # 根据Lowe的统计，最匹配和次匹配之间距离差不多时更易出现错误匹配
    # 根据以上规律筛选匹配点
    for m, n in siftMatches:
        if m.distance < 0.7*n.distance:
            good_sift.append(m)

    # 画出匹配图
    sift_match = cv2.drawMatches(
        imgA, siftKpA, imgB, siftKpB, good_sift, None, matchColor=(0, 255, 0))

    # 取出两张图中根据Lowe距离筛选过后的匹配点的坐标
    dst_pts_sift = np.float32(
        [siftKpA[m.queryIdx].pt for m in good_sift]).reshape(-1, 1, 2)
    src_pts_sift = np.float32(
        [siftKpB[m.trainIdx].pt for m in good_sift]).reshape(-1, 1, 2)

    # 用RANSAC滤除离群点并找到B到A的单应变换矩阵
    H_sift, mask_sift = cv2.findHomography(
        src_pts_sift, dst_pts_sift, cv2.RANSAC, 5.0)  # RANSAC最大重投影错误为5，大于该值则为离群点
    matchesMask_sift = mask_sift.ravel().tolist()  # RANSAC过滤后点的掩码，掩码为0不能被画出
    sift_ransac = cv2.drawMatches(imgA, siftKpA, imgB, siftKpB, good_sift, None, matchColor=(
        0, 255, 0), matchesMask=matchesMask_sift)  # RANSAC滤除离群点后画出匹配图

    # 利用单应变换矩阵把第二张图变换到第一张图的坐标系下，长宽分别为A、B图之和
    imgB_warp_sift = cv2.warpPerspective(
        imgB, H_sift, (imgB.shape[1]+imgA.shape[1], imgB.shape[0]+imgA.shape[0]))

    # 图像裁剪、融合
    sift_stitch = stitch(imgA, imgB_warp_sift)

    # 打印单应矩阵保存图像
    print(H_sift)
    cv2.imwrite('SIFT_match.png', sift_match)
    cv2.imwrite('SIFT_ransac.png', sift_ransac)
    cv2.imwrite('SIFT_stitch.png', sift_stitch)

    # 用SURF寻找特征点和描述子
    surf = cv2.xfeatures2d.SURF_create()
    surfKpA, surfDesA = surf.detectAndCompute(imgA, None)
    surfKpB, surfDesB = surf.detectAndCompute(imgB, None)

    # BruteForce匹配特征点
    surfbf = cv2.BFMatcher()
    surfMatches = surfbf.knnMatch(surfDesA, surfDesB, k=2)  # 返回2个匹配（即最匹配和次匹配）

    # 根据Lowe的统计，最匹配和次匹配之间距离差不多时更易出现错误匹配
    # 根据以上规律筛选匹配点
    good_surf = []
    for m, n in surfMatches:
        if m.distance < 0.7*n.distance:
            good_surf.append(m)

    # 画出匹配图
    surf_match = cv2.drawMatches(
        imgA, surfKpA, imgB, surfKpB, good_surf, None, matchColor=(0, 255, 0))

    # 取出两张图中根据Lowe距离筛选过后的匹配点的坐标
    dst_pts_surf = np.float32(
        [surfKpA[m.queryIdx].pt for m in good_surf]).reshape(-1, 1, 2)
    src_pts_surf = np.float32(
        [surfKpB[m.trainIdx].pt for m in good_surf]).reshape(-1, 1, 2)

    # 用RANSAC滤除离群点并找到B到A的单应变换矩阵
    H_surf, mask_surf = cv2.findHomography(
        src_pts_surf, dst_pts_surf, cv2.RANSAC, 5.0)  # RANSAC最大重投影错误为5，大于该值则为离群点
    matchesMask_surf = mask_surf.ravel().tolist()  # RANSAC过滤后点的掩码，掩码为0不能被画出
    surf_ransac = cv2.drawMatches(imgA, surfKpA, imgB, surfKpB, good_surf, None, matchColor=(
        0, 255, 0), matchesMask=matchesMask_surf)  # RANSAC滤除离群点后画出匹配图

    # 利用单应变换矩阵把第二张图变换到第一张图的坐标系下，长宽分别为A、B图之和
    imgB_warp_surf = cv2.warpPerspective(
        imgB, H_surf, (imgB.shape[1]+imgA.shape[1], imgB.shape[0]+imgA.shape[0]))

    # 图像裁剪、融合
    surf_stitch = stitch(imgA, imgB_warp_surf)

    # 打印单应矩阵保存图像
    print(H_surf)
    cv2.imwrite('SURF_match.png', surf_match)
    cv2.imwrite('SURF_ransac.png', surf_ransac)
    cv2.imwrite('SURF_stitch.png', surf_stitch)

    # 用ORB寻找特征点和描述子
    orb = cv2.ORB_create()
    orbKpA, orbDesA = orb.detectAndCompute(imgA, None)
    orbKpB, orbDesB = orb.detectAndCompute(imgB, None)

    # BruteForce匹配特征点
    orbbf = cv2.BFMatcher(cv2.NORM_HAMMING)
    orbMatches = orbbf.knnMatch(orbDesA, orbDesB, k=2)  # 返回2个匹配（即最匹配和次匹配）

    # 根据Lowe的统计，最匹配和次匹配之间距离差不多时更易出现错误匹配
    # 根据以上规律筛选匹配点
    good_orb = []
    for m, n in orbMatches:
        if m.distance < 0.7*n.distance:
            good_orb.append(m)

    # 画出匹配图
    orb_match = cv2.drawMatches(
        imgA, orbKpA, imgB, orbKpB, good_orb, None, matchColor=(0, 255, 0))

    # 取出两张图中根据Lowe距离筛选过后的匹配点的坐标
    dst_pts_orb = np.float32(
        [orbKpA[m.queryIdx].pt for m in good_orb]).reshape(-1, 1, 2)
    src_pts_orb = np.float32(
        [orbKpB[m.trainIdx].pt for m in good_orb]).reshape(-1, 1, 2)

    # 用RANSAC滤除离群点并找到B到A的单应变换矩阵
    H_orb, mask_orb = cv2.findHomography(
        src_pts_orb, dst_pts_orb, cv2.RANSAC, 5.0)  # RANSAC最大重投影错误为5，大于该值则为离群点
    matchesMask_orb = mask_orb.ravel().tolist()  # RANSAC过滤后点的掩码，掩码为0不能被画出
    orb_ransac = cv2.drawMatches(imgA, orbKpA, imgB, orbKpB, good_orb, None, matchColor=(
        0, 255, 0), matchesMask=matchesMask_orb)  # RANSAC滤除离群点后画出匹配图

    # 利用单应变换矩阵把第二张图变换到第一张图的坐标系下，长宽分别为A、B图之和
    imgB_warp_orb = cv2.warpPerspective(
        imgB, H_orb, (imgB.shape[1]+imgA.shape[1], imgB.shape[0]+imgA.shape[0]))

    # 图像裁剪、融合
    orb_stitch = stitch(imgA, imgB_warp_orb)

    # 打印单应矩阵保存图像
    print(H_orb)
    cv2.imwrite('ORB_match.png', orb_match)
    cv2.imwrite('ORB_ransac.png', orb_ransac)
    cv2.imwrite('ORB_stitch.png', orb_stitch)


if __name__ == '__main__':
    main()
