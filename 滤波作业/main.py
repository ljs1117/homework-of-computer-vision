"""
1851754 李玖思
自动驾驶机器视觉 滤波作业

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


def imgAverageFilter(image, kernel):
    '''均值滤波'''

    return imgConvolution(image, kernel) / kernel.size


def imgGaussianFilter(image, kernelSize, sigma):
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


def imgSobelFilter(image, kernel):
    '''Sobel滤波'''

    return imgConvolution(image, kernel)


def imgDerivationFilter(image, kernel):
    '''导数滤波'''

    return imgConvolution(image, kernel)


def imgMedianFilter(image, kernelSize):
    '''中值滤波'''

    imgHeight = int(image.shape[0])  # 图像高
    imgWidth = int(image.shape[1])  # 图像宽
    border = kernelSize//2  # 边缘
    imgPadded = imgCopyBorder(image, border)  # 边缘补全像素
    result = np.zeros(image.shape)  # 结果分配空间

    # 取核的中位数
    for i in range(border, border + imgHeight):
        for j in range(border, border + imgWidth):
            result[i - border][j - border] = np.median(
                imgPadded[i - border:i + border + 1, j - border:j + border + 1])

    return result


def imgBilateralFilter(image, d, sigmaColor, sigmaSpace):
    '''双边滤波'''

    imgHeight = int(image.shape[0])  # 图像高
    imgWidth = int(image.shape[1])  # 图像宽
    result = np.zeros(image.shape, dtype=np.float32)  # 结果分配空间
    r = int(d/2)  # 窗口半径
    copyBorderImage = imgCopyBorder(image, r)  # 边缘补全像素

    for i in range(imgHeight):
        for j in range(imgWidth):
            weightSum = 0
            filterValue = 0

            # 求加权后窗口内每一点的像素
            for row in range(d):
                for col in range(d):
                    distanceSquare = math.pow(
                        row-r, 2)+math.pow(col-r, 2)  # 像素空间距离
                    graySquare = math.pow(
                        int(image[i, j])-int(copyBorderImage[i+row, j+col]), 2)  # 像素差
                    weight = math.exp(-1*(distanceSquare /
                                      (2*math.pow(sigmaSpace, 2))+graySquare/(2*math.pow(sigmaColor, 2))))  # 权重
                    weightSum += weight  # 权重和
                    filterValue += weight*int(copyBorderImage[i+row, j+col])

            result[i, j] = filterValue/weightSum

    return result


def imgGuideFilter(srcImage, guideImage, kernelSize, eps):
    '''导向滤波'''

    srcImage = srcImage.astype(np.float32)  # 输入图像（float32避免溢出）
    guideImage = guideImage.astype(np.float32)  # 引导图像
    averageKernel = np.ones((kernelSize, kernelSize))  # 用于求均值的卷积核

    meanP = imgAverageFilter(srcImage, averageKernel)  # 输入图像窗口内均值
    meanI = imgAverageFilter(guideImage, averageKernel)  # 引导图像窗口内均值
    meanII = imgAverageFilter(np.multiply(
        guideImage, guideImage), averageKernel)  # 引导图像平方的均值
    meanIP = imgAverageFilter(np.multiply(
        srcImage, guideImage), averageKernel)  # 输入图像和引导图像乘积的均值
    varI = meanII-np.multiply(meanI, meanI)  # 引导图像窗口内方差
    covIP = meanIP-np.multiply(meanI, meanP)  # 输入图像和引导图像窗口内协方差

    a = covIP/(varI+eps*math.pow(255, 2))
    # 注意此处正则项*255**2，，否则正则项和方差项不匹配。如果图像作归一化操作则无需
    b = meanP-np.multiply(a, meanI)

    meana = imgAverageFilter(a, averageKernel)  # 每个像素点所在的所有窗口a的均值
    meanb = imgAverageFilter(b, averageKernel)  # 每个像素点所在的所有窗口b的均值
    dstImg = np.multiply(meana, guideImage)+meanb  # 输出图像
    return dstImg


def imgOutput(imgFiltered):
    '''矩阵格式规范后输出'''

    np.abs(imgFiltered, out=imgFiltered)  # 取绝对值
    result = np.clip(imgFiltered, 0, 255)  # [0,255]截断
    imgFiltered.astype(np.uint8)  # 数据格式转换

    return result


def main():
    '''主函数'''

    # 均值滤波卷积核
    kernelAverage3 = np.ones((3, 3))
    kernelAverage7 = np.ones((7, 7))

    # 读入图像
    imageGaussian = cv2.imread(
        'images/gaussian_noise.png', cv2.IMREAD_GRAYSCALE)
    imageOrigin = cv2.imread('images/origin_image.png', cv2.IMREAD_GRAYSCALE)
    imagePepper = cv2.imread(
        'images/pepper_noise.png', cv2.IMREAD_GRAYSCALE)

    # Sobel滤波卷积核
    sobelKernelV3 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobelKernelH3 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # 导数滤波卷积核
    derivationKernelH3 = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
    derivationKernelV3 = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])

    # 3x3均值滤波
    img_gaussian_average3 = imgOutput(
        imgAverageFilter(imageGaussian, kernelAverage3))
    img_origin_average3 = imgOutput(
        imgAverageFilter(imageOrigin, kernelAverage3))
    img_pepper_average3 = imgOutput(
        imgAverageFilter(imagePepper, kernelAverage3))

    # 3x3高斯滤波（标准差取3）
    img_gaussian_gaussian3 = imgOutput(
        imgGaussianFilter(imageGaussian, 3, 3))
    img_origin_gaussian3 = imgOutput(
        imgGaussianFilter(imageOrigin, 3, 3))
    img_pepper_gaissian3 = imgOutput(
        imgGaussianFilter(imagePepper, 3, 3))

    # 3x3 Sobel滤波（y）
    img_gaussian_sobelV3 = imgOutput(
        imgSobelFilter(imageGaussian, sobelKernelV3))
    img_origin_sobelV3 = imgOutput(
        imgSobelFilter(imageOrigin, sobelKernelV3))
    img_pepper_sobelV3 = imgOutput(
        imgSobelFilter(imagePepper, sobelKernelV3))

    # 3x3 Sobel滤波（x）
    img_gaussian_sobelH3 = imgOutput(
        imgSobelFilter(imageGaussian, sobelKernelH3))
    img_origin_sobelH3 = imgOutput(
        imgSobelFilter(imageOrigin, sobelKernelH3))
    img_pepper_sobelH3 = imgOutput(
        imgSobelFilter(imagePepper, sobelKernelH3))

    # 3x3 导数滤波（y）
    img_gaussian_derivationV3 = imgOutput(
        imgDerivationFilter(imageGaussian, derivationKernelV3))
    img_origin_derivationV3 = imgOutput(
        imgDerivationFilter(imageOrigin, derivationKernelV3))
    img_pepper_derivationV3 = imgOutput(
        imgDerivationFilter(imagePepper, derivationKernelV3))

    # 3x3 导数滤波（x）
    img_gaussian_derivationH3 = imgOutput(
        imgDerivationFilter(imageGaussian, derivationKernelH3))
    img_origin_derivationH3 = imgOutput(
        imgDerivationFilter(imageOrigin, derivationKernelH3))
    img_pepper_derivationH3 = imgOutput(
        imgDerivationFilter(imagePepper, derivationKernelH3))

    # 3x3中值滤波
    img_gaussian_median3 = imgOutput(
        imgMedianFilter(imageGaussian, 3))
    img_origin_median3 = imgOutput(
        imgMedianFilter(imageOrigin, 3))
    img_pepper_median3 = imgOutput(
        imgMedianFilter(imagePepper, 3))

    # 7x7均值滤波
    img_gaussian_average7 = imgOutput(
        imgAverageFilter(imageGaussian, kernelAverage7))
    img_origin_average7 = imgOutput(
        imgAverageFilter(imageOrigin, kernelAverage7))
    img_pepper_average7 = imgOutput(
        imgAverageFilter(imagePepper, kernelAverage7))

    # 7x7高斯滤波（标准差取1）
    img_gaussian_gaussian7 = imgOutput(
        imgGaussianFilter(imageGaussian, 7, 1))
    img_origin_gaussian7 = imgOutput(
        imgGaussianFilter(imageOrigin, 7, 1))
    img_pepper_gaissian7 = imgOutput(
        imgGaussianFilter(imagePepper, 7, 1))

    # 7x7双边滤波（像素值域方差30，空间域方差100）
    img_gaussian_bilateral7 = imgOutput(
        imgBilateralFilter(imageGaussian, 7, 30, 100))
    img_origin_bilateral7 = imgOutput(
        imgBilateralFilter(imageOrigin, 7, 30, 100))
    img_pepper_bilateral7 = imgOutput(
        imgBilateralFilter(imagePepper, 7, 30, 100))

    # 7x7导向滤波（正则项0.01,注函数内部由于图像未作归一化故计算时该项*255**2）
    img_gaussian_guided7 = imgOutput(
        imgGuideFilter(imageGaussian, imageGaussian, 7, 0.01))
    img_origin_guided7 = imgOutput(
        imgGuideFilter(imageOrigin, imageOrigin, 7, 0.01))
    img_pepper_guided7 = imgOutput(
        imgGuideFilter(imagePepper, imagePepper, 7, 0.01))

    #输出图像
    cv2.imwrite('result/3x3_Average_gaussian.png', img_gaussian_average3)
    cv2.imwrite('result/3x3_Average_origin.png', img_origin_average3)
    cv2.imwrite('result/3x3_Average_pepper.png', img_pepper_average3)

    cv2.imwrite('result/3x3_Gaussian_gaussian.png', img_gaussian_gaussian3)
    cv2.imwrite('result/3x3_Gaussian_origin.png', img_origin_gaussian3)
    cv2.imwrite('result/3x3_Gaussian_pepper.png', img_pepper_gaissian3)

    cv2.imwrite('result/3x3_Sobel_y_gaussian.png', img_gaussian_sobelV3)
    cv2.imwrite('result/3x3_Sobel_y_origin.png', img_origin_sobelV3)
    cv2.imwrite('result/3x3_Sobel_y_pepper.png', img_pepper_sobelV3)

    cv2.imwrite('result/3x3_Sobel_x_gaussian.png', img_gaussian_sobelH3)
    cv2.imwrite('result/3x3_Sobel_x_origin.png', img_origin_sobelH3)
    cv2.imwrite('result/3x3_Sobel_x_pepper.png', img_pepper_sobelH3)

    cv2.imwrite('result/3x3_Derivation_y_gaussian.png',
                img_gaussian_derivationV3)
    cv2.imwrite('result/3x3_Derivation_y_origin.png', img_origin_derivationV3)
    cv2.imwrite('result/3x3_Derivation_y_pepper.png', img_pepper_derivationV3)

    cv2.imwrite('result/3x3_Derivation_x_gaussian.png',
                img_gaussian_derivationH3)
    cv2.imwrite('result/3x3_Derivation_x_origin.png', img_origin_derivationH3)
    cv2.imwrite('result/3x3_Derivation_x_pepper.png', img_pepper_derivationH3)

    cv2.imwrite('result/3x3_Median_gaussian.png', img_gaussian_median3)
    cv2.imwrite('result/3x3_Median_origin.png', img_origin_median3)
    cv2.imwrite('result/3x3_Median_pepper.png', img_pepper_median3)

    cv2.imwrite('result/7x7_Average_gaussian.png', img_gaussian_average7)
    cv2.imwrite('result/7x7_Average_origin.png', img_origin_average7)
    cv2.imwrite('result/7x7_Average_pepper.png', img_pepper_average7)

    cv2.imwrite('result/7x7_Gaussian_gaussian.png', img_gaussian_gaussian7)
    cv2.imwrite('result/7x7_Gaussian_origin.png', img_origin_gaussian7)
    cv2.imwrite('result/7x7_Gaussian_pepper.png', img_pepper_gaissian7)

    cv2.imwrite('result/7x7_Bilateral_gaussian.png', img_gaussian_bilateral7)
    cv2.imwrite('result/7x7_Bilateral_origin.png', img_origin_bilateral7)
    cv2.imwrite('result/7x7_Bilateral_pepper.png', img_pepper_bilateral7)

    cv2.imwrite('result/7x7_Guided_gaussian.png', img_gaussian_guided7)
    cv2.imwrite('result/7x7_Guided_origin.png', img_origin_guided7)
    cv2.imwrite('result/7x7_Guided_pepper.png', img_pepper_guided7)

if __name__ == '__main__':
    main()
