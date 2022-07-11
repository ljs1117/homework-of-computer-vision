"""
1851754 李玖思
自动驾驶机器视觉 形状检测作业

Ubuntu 18.04.4 LTS
x86_64
VSCode 1.54.3
Python 3.6.9

OpenCV 3.2.0
Numpy

"""


import cv2
import numpy as np
import math


def imgCopyBorder(image, border):
    '''图像复制边缘补全像素'''

    imgHeight = int(image.shape[0])  # 图像高
    imgWidth = int(image.shape[1])  # 图像宽

    copyBoderImageHeight = int(imgHeight + 2 * border)  # 复制边缘后图像高
    copyBoderImageWidth = int(imgWidth + 2 * border)  # 复制边缘后图像宽

    imgPadded = np.zeros(
        (copyBoderImageHeight, copyBoderImageWidth), dtype=np.float32)  # 分配空间

    imgPadded[border:border + imgHeight,
              border:border + imgWidth] = image[:, :]  # 中心图像填充

    for i in range(border):  # 复制上下边缘
        imgPadded[i, border:border +
                  imgWidth] = image[0, :]
        imgPadded[i+imgHeight+border, border:border +
                  imgWidth] = image[imgHeight-1, :]
    for j in range(border):  # 复制左右边缘
        imgPadded[:, j] = imgPadded[:, border]
        imgPadded[:, border+imgWidth+j] = imgPadded[:, border+imgWidth-1]

    return imgPadded


def imgConvolution(image, kernel):
    '''卷积'''

    imgHeight = int(image.shape[0])  # 图像高
    imgWidth = int(image.shape[1])  # 图像宽
    border = int(kernel.shape[0]/2)  # 边缘
    imgPadded = imgCopyBorder(image, border)  # 边缘补全像素
    imageConvolved = np.zeros(image.shape, dtype=np.float32)  # 分配空间

    # 卷积
    for i in range(border, border + imgHeight):
        for j in range(border, border + imgWidth):
            imageConvolved[i - border][j - border] = np.sum(
                imgPadded[i - border:i + border + 1, j - border:j + border + 1] * kernel)

    return imageConvolved


def imgGaussianFilter(image, kernelSize=3, sigma=1):
    '''高斯滤波'''

    kernel = np.zeros([kernelSize, kernelSize], dtype=np.float32)  # 高斯核分配空间
    center = kernelSize//2  # 高斯核中心
    squareSigma = 2*math.pow(sigma, 2)  # 标准差的平方
    sum = 0  # 高斯核归一化系数

    # 生成高斯核
    for i in range(kernelSize):
        for j in range(kernelSize):
            kernel[i, j] = math.exp(-(math.pow(i-center, 2) +
                                    math.pow(j-center, 2))/squareSigma)
            sum += kernel[i, j]

    kernel /= sum  # 高斯核归一化
    return imgConvolution(image, kernel)  # 高斯滤波


def gradient(image):
    '''计算梯度幅值和方向'''

    sobelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # x方向sobel算子,正方向向右
    sobelY = np.array([[-1, - 2, -1], [0, 0, 0], [1, 2, 1]]
                      )  # y方向sobel算子，正方向向下
    imgHeight = int(image.shape[0])  # 图像高
    imgWidth = int(image.shape[1])  # 图像宽
    gradients = np.zeros(
        (imgHeight, imgWidth), dtype=np.float32)  # 梯度幅值分配空间
    direction = np.zeros(
        (imgHeight, imgWidth), dtype=np.float32)  # 梯度方向分配空间

    dx = imgConvolution(image, sobelX)  # x向梯度
    dy = imgConvolution(image, sobelY)  # y 向梯度

    for i in range(imgHeight):
        for j in range(imgWidth):
            gradients[i, j] = np.sqrt(dx[i, j]**2+dy[i, j]**2)  # 梯度幅值
            if dx[i, j] == 0:
                direction[i, j] = np.pi/2
            else:
                direction[i, j] = np.arctan(dy[i, j] / dx[i, j])  # 梯度角度

    return gradients, direction, dx, dy


def NMS(gradients, direction):
    '''非极大值抑制 Non-Maximum Suppression'''

    gradientsAfterNMS = np.copy(gradients)  # 分配空间
    Height, Width = gradients.shape

    # 图像边界无法完整进行梯度比较，默认置0，非边缘
    gradientsAfterNMS[0, :] = 0
    gradientsAfterNMS[Height-1, :] = 0
    gradientsAfterNMS[:, 0] = 0
    gradientsAfterNMS[:, Width-1] = 0

    # 梯度角度离散化后,进行梯度比较
    # [0,pi/8]和[pi*7/8,pi]区间认为是水平方向（0°），（pi/8,pi*3/8）认为是正对角线方向（45°）
    # [pi*3/8,pi*5/8]认为是竖直方向（90°）,(pi*5/8,pi*7/8)认为是负对角线方向（135°）
    # 在canny二值化结果差不多的情况下，梯度角度离散化计算效率高于插值
    # 注意本节代码注释部分的同名函数即为插值的NMS

    direction[direction < 0] += np.pi  # 梯度角度映射到[0,pi]
    for i in range(1, Height-1):
        for j in range(1, Width-1):

            if (0 <= direction[i, j] <= np.pi/8) or (np.pi*7/8 <= direction[i, j] <= np.pi):
                # 水平方向比较左右相邻像素
                temp1 = gradients[i, j+1]
                temp2 = gradients[i, j-1]

            elif (np.pi/8 < direction[i, j] < np.pi*3/8):
                # 正对角线方向比较左上和右下相邻像素
                temp1 = gradients[i+1, j+1]
                temp2 = gradients[i-1, j-1]

            elif (np.pi*3/8 <= direction[i, j] <= np.pi*5/8):
                # 竖直方向比较上下相邻像素
                temp1 = gradients[i+1, j]
                temp2 = gradients[i-1, j]

            elif (np.pi*5/8 < direction[i, j] < np.pi*7/8):
                # 负对角线方向比较右上和左下相邻像素
                temp1 = gradients[i-1, j+1]
                temp2 = gradients[i+1, j-1]

            if (gradients[i, j] < temp1) or (gradients[i, j] < temp2):
                # 梯度非极大值则置0
                gradientsAfterNMS[i, j] = 0

    return gradientsAfterNMS


'''
def NMS(gradients, direction):
    # 非极大值抑制，梯度方向通过线性插值得到的版本

    gradientsAfterNMS = np.copy(gradients)  # 分配空间
    Height, Width = gradients.shape

    # 图像边界无法完整进行梯度比较，默认置0，非边缘
    gradientsAfterNMS[0, :] = 0
    gradientsAfterNMS[Height-1, :] = 0
    gradientsAfterNMS[:, 0] = 0
    gradientsAfterNMS[:, Width-1] = 0

    for i in range(1, Height - 1):
        for j in range(1, Width - 1):
            theta = direction[i, j]
            weight = np.tan(theta)
            if gradients[i, j] == 0:
                gradientsAfterNMS[i, j] = 0  # 梯度幅值为0，则一定非极大值
            else:
                if theta > np.pi/4:

                    # g1 g2
                    #      c
                    #       g4 g3

                    g2 = gradients[i-1, j]
                    g4 = gradients[i+1, j]
                    g1 = gradients[i-1, j-1]
                    g3 = gradients[i+1, j+1]
                    weight = 1/weight
                    temp1 = weight*g1+(1-weight)*g2
                    temp2 = weight*g3+(1-weight)*g4

                elif theta >= 0:

                    # g1
                    # g2 c g4
                    #         g3

                    g2 = gradients[i, j-1]
                    g4 = gradients[i, j+1]
                    g1 = gradients[i-1, j-1]
                    g3 = gradients[i+1, j+1]
                    temp1 = weight*g1+(1-weight)*g2
                    temp2 = weight*g3+(1-weight)*g4

                elif theta >= -np.pi/4:

                    #           g3
                    # g2  c  g4
                    # g1

                    g2 = gradients[i, j-1]
                    g4 = gradients[i, j+1]
                    g1 = gradients[i+1, j-1]
                    g3 = gradients[i-1, j+1]
                    weight *= -1
                    temp1 = weight*g1+(1-weight)*g2
                    temp2 = weight*g3+(1-weight)*g4

                else:

                    #      g2  g1
                    #       c
                    # g3  g4

                    g2 = gradients[i-1, j]
                    g4 = gradients[i+1, j]
                    g1 = gradients[i-1, j+1]
                    g3 = gradients[i+1, j-1]
                    weight = -1/weight
                    temp1 = weight*g1+(1-weight)*g2
                    temp2 = weight*g3+(1-weight)*g4

                if temp1 > gradients[i, j] or temp2 > gradients[i, j]:  # 梯度非极大值置0
                    gradientsAfterNMS[i, j] = 0

    return gradientsAfterNMS
'''


def doubleThreshold(gradientsAfterNMS, threshold1, threshold2):
    '''双阈值处理'''

    imgHeight = int(gradientsAfterNMS.shape[0])  # 图像高
    imgWidth = int(gradientsAfterNMS.shape[1])  # 图像宽
    visited = np.zeros((imgHeight, imgWidth))  # 分配空间，该矩阵用于判断该位置是否被搜索过
    result = gradientsAfterNMS.copy()  # 处理结果分配空间

    # 深度优先搜索
    # 以强边缘点开始搜索，搜索出所有与其连通的弱边缘、强边缘点
    def dfs(i, j):
        if i >= imgHeight or i < 0 or j >= imgWidth or j < 0 or visited[i, j] == 1:
            # 索引超出图像范围或者该点已被搜索过则退出搜索
            return
        visited[i, j] = 1  # 标记为已搜索
        if result[i, j] > threshold1:  # 如果梯度值高于低阈值则搜索其8连通像素
            dfs(i-1, j-1)
            dfs(i-1, j)
            dfs(i-1, j+1)
            dfs(i, j-1)
            dfs(i, j+1)
            dfs(i+1, j-1)
            dfs(i+1, j)
            dfs(i+1, j+1)
        else:
            result[i, j] = 0  # 梯度低于低阈值则置0

    for r in range(imgHeight):
        for c in range(imgWidth):
            if visited[r, c] == 1:  # 如该点已搜索过，则继续下一点
                continue
            if result[r, c] >= threshold2:  # 如果该点为强边缘点，则开始深度优先搜索，以连通所有弱边缘、强边缘点
                dfs(r, c)
            elif result[r, c] <= threshold1:  # 如果该点非边缘，则置0抑制，并标记为已搜索
                result[r, c] = 0
                visited[r, c] = 1

    # 所有未搜索到的像素置0
    for r in range(imgHeight):
        for c in range(imgWidth):
            if visited[r, c] == 0:
                result[r, c] = 0
    return result


def toBinary(gradientsAfterDT):
    '''图像二值化'''

    Height, Width = gradientsAfterDT.shape
    binaryImg = np.copy(gradientsAfterDT)
    for i in range(Height):
        for j in range(Width):
            if binaryImg[i, j] > 0:  # 大于0的像素值全部置为255
                binaryImg[i, j] = 255

    return binaryImg


def DDALines(binaryEdge, dx, dy):
    '''DDA算法画线'''

    # 数值微分法画线（Digital Differential Analyzer）
    # 该方法用于在canny检测后的二值化图的非零像素点沿梯度方向画线投票
    # 得到霍夫圆心

    Height, Width = binaryEdge.shape
    lineImage = np.zeros_like(binaryEdge)  # 分配空间

    for i in range(Height):
        for j in range(Width):
            if binaryEdge[i, j] == 255:  # 二值图非零像素点画线投票

                # 算出x，y方向的小增量
                # |k|<1时，deltax=1，deltay<1
                # |k|>1时，deltay=1，deltax<1
                if abs(dx[i, j]) > abs(dy[i, j]):
                    eps = abs(dx[i, j])
                else:
                    eps = abs(dy[i, j])

                deltax = float(dx[i, j]/eps)
                deltay = float(dy[i, j]/eps)

                # 以圆心为起点向dx，dy方向画线
                x = j
                y = i
                while (round(x) < Width) and (round(y) < Height) and (round(x) >= 0) and (round(y) >= 0):
                    # 索引超出图像边缘则停止画线
                    lineImage[round(y), round(x)] += 1
                    x += deltax
                    y += deltay

                # 以圆心为起点向dx，dy反方向画线
                x = j-deltax
                y = i-deltay
                while (round(x) < Width) and (round(y) < Height) and (round(x) >= 0) and (round(y) >= 0):
                    # 索引超出图像边缘则停止画线
                    lineImage[round(y), round(x)] += 1
                    x -= deltax
                    y -= deltay
    return lineImage


def checkDistance(centerList, centerTobechecked, minDist):
    '''距离检查'''

    # 用于检测输入圆心是否与已找到的圆心距离过近
    good = True
    for c in centerList:  # 遍历已找到的所有圆心
        if math.sqrt((c[0]-centerTobechecked[0])**2+(c[1]-centerTobechecked[1])**2) < minDist:
            good = False  # 距离过近则输入圆心不可再被添加
            break
    return good


def selectCircleCenter(lineImage, minDist, centerAccThreshold):
    '''根据投票选择圆心'''

    Height, Width = lineImage.shape

    # 如果投票值小于圆心累加器最低阈值，则累加器置0
    for i in range(Height):
        for j in range(Width):
            if lineImage[i, j] < centerAccThreshold:
                lineImage[i, j] = 0

    # 获取投票局部最大值
    # 为了避免局部区域极大值有多个且相等的取舍问题
    # （既不可以全取，否则无抑制作用，也不可以全舍，否则该区域的圆心被漏检）
    # 故而参考OpenCV算法：将投票值从大到小排序后，依次输入距离检测方法，如果该点与已找到的圆心距离过近则舍去
    # 该方法保证找到局部最大值，且如果局部最大值有多个且相等，排序在前的点被找到后，排序在后的即使投票值相等也被舍去
    # 保证了局部区域内投票数相等的极大值点只取一个

    centers = np.argwhere(lineImage)  # 所有点的索引
    votes = lineImage[(lineImage != 0)]  # 所有点的投票值
    priorities = np.argsort(votes)[::-1]  # 按投票值从大到小将点排序，返回索引
    centerList = []
    for index in priorities:
        centerTobechecked = centers[index]
        good = checkDistance(centerList, centerTobechecked,
                             minDist)  # 检查该点与已找到的点是否距离过近
        if good:
            centerList.append(centerTobechecked)  # 距离不过近，则将该点坐标（索引）添加进已找到圆心的列表

    return centerList


def HoughCircle(binaryEdge, centerList, minR, maxR):
    '''霍夫圆检测'''

    # 用于计算canny二值图非0像素点到圆心的距离，以获得半径

    Height, Width = binaryEdge.shape
    circle = []

    for center in centerList:
        i = center[0]  # 圆心索引
        j = center[1]

        # 初始化半径累加器，半径累加器的索引为半径长度（整数），最大索引为图像对角线长
        radiusAcc = np.zeros(
            math.ceil(math.sqrt(Height ** 2 + Width ** 2)))

        # 遍历canny二值图所有非零点
        for r in range(Height):
            for c in range(Width):

                if binaryEdge[r, c] != 0:
                    tempRadius = round(
                        math.sqrt((r-i)**2+(c-j)**2))  # 二值图的非零点到圆心的距离
                    if (tempRadius >= minR) and (tempRadius <= maxR):  # 介于最大最小半径之间则相应累加器+1
                        radiusAcc[tempRadius] += 1

        radiusList = np.argwhere(radiusAcc == np.max(
            radiusAcc))  # 取该圆心下累加器最大值的索引（对应半径）

        for radius in radiusList:

            circle.append([i, j,  radius[0]])  # 将该圆心和半径添加进圆列表

    return circle


def imgOutput(imgFiltered):
    '''矩阵格式规范后输出'''

    np.abs(imgFiltered, out=imgFiltered)  # 取绝对值
    result = np.clip(imgFiltered, 0, 255)  # [0,255]截断
    imgFiltered.astype(np.uint8)  # 数据格式转换

    return result


def main():
    '''主函数'''

    # 读入图像
    imageLanes = cv2.imread(
        'img/lanes.png', cv2.IMREAD_GRAYSCALE)  # lanes灰度图
    imageWheel = cv2.imread(
        'img/wheel.png', cv2.IMREAD_GRAYSCALE)  # wheel灰度图
    imageWheelRGB = cv2.imread('img/wheel.png', cv2.IMREAD_COLOR)  # wheel彩图

    # 高斯滤波，标准差为1，高斯核3x3
    imageLanesGuassian = imgGaussianFilter(
        imageLanes, 3, 1)

    # 获取lanes的梯度幅值和梯度方向
    gradients, direction, dx, dy = gradient(imageLanesGuassian)
    cv2.imwrite('gradientsOriginLanes.png', np.uint8(imgOutput(gradients)))

    # lanes非极大值抑制后的梯度幅值
    gradientsAfterNMS = NMS(gradients, direction)
    cv2.imwrite('gradientsAfterNMSLanes.png',
                np.uint8(imgOutput(gradientsAfterNMS)))

    # lanes双阈值处理和深度优先搜索后连接的梯度幅值图，低阈值100，高阈值200
    gradientsAfterDT = doubleThreshold(gradientsAfterNMS, 100, 200)
    cv2.imwrite('gradientsAfterDoubleThresholdLanes.png',
                np.uint8(imgOutput(gradientsAfterDT)))

    # lanes梯度二值化图
    binaryImgLanes = toBinary(gradientsAfterDT)
    cv2.imwrite('binaryImageLanes.png',
                np.uint8(imgOutput(binaryImgLanes)))

    # 高斯滤波，标准差为1，高斯核3x3
    imageWheelGaussian = imgGaussianFilter(imageWheel, 3, 1)

    # 获取wheel的梯度幅值和梯度方向
    gradientsW, directionW, dxW, dyW = gradient(imageWheelGaussian)

    # wheel非极大值抑制后的梯度幅值
    gradientsAfterNMSWheel = NMS(gradientsW, directionW)

    # wheel双阈值处理和深度优先搜索后连接的梯度幅值图，低阈值40，高阈值80
    gradientsAfterDTWheel = doubleThreshold(gradientsAfterNMSWheel, 40, 80)

    # wheel梯度二值化图
    binaryImgWheel = toBinary(gradientsAfterDTWheel)
    cv2.imwrite('binaryImageWheel.png',
                np.uint8(imgOutput(binaryImgWheel)))

    # wheel参数空间图（圆心投票）
    parameterSpaceWheel = DDALines(binaryImgWheel, dxW, dyW)
    cv2.imwrite('parameterSpaceWheel.png',
                np.uint8(imgOutput(parameterSpaceWheel)))

    # wheel选择圆心，圆心累加器的最低阈值为60，局部最大值（局部的半径为15）
    centerList = selectCircleCenter(parameterSpaceWheel, 15, 60)

    # 霍夫圆检测，最小半径为0，最大半径为25
    circleList = HoughCircle(binaryImgWheel, centerList, 0, 25)

    # 遍历找到的所有圆和对应半径，并画出
    for circle in circleList:

        imgCircle = cv2.circle(
            imageWheelRGB, (circle[1], circle[0]), circle[2], (0, 0, 255), 2)

    cv2.imwrite('wheelSearched.png', imgCircle)


if __name__ == '__main__':
    main()
