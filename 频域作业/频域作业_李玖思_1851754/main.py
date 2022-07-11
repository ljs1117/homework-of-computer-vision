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


def GaussianKernel(kernelSize, sigma):
    '''生成高斯核'''

    kernel = np.zeros([kernelSize, kernelSize],
                      dtype=np.float32)  # 高斯核分配空间
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

    return kernel


def imgErosion(image, kernelSize):
    '''腐蚀'''

    imgHeight = int(image.shape[0])  # 图像高
    imgWidth = int(image.shape[1])  # 图像宽
    border = kernelSize//2  # 边缘
    imgPadded = imgCopyBorder(image, border)  # 边缘补全像素
    result = np.zeros(image.shape)  # 结果分配空间

    # 取核内最小值
    for i in range(border, border + imgHeight):
        for j in range(border, border + imgWidth):
            result[i - border][j - border] = np.min(
                imgPadded[i - border:i + border + 1, j - border:j + border + 1])

    return result


def imgDilation(image, kernelSize):
    '''膨胀'''

    imgHeight = int(image.shape[0])  # 图像高
    imgWidth = int(image.shape[1])  # 图像宽
    border = kernelSize//2  # 边缘
    imgPadded = imgCopyBorder(image, border)  # 边缘补全像素
    result = np.zeros(image.shape)  # 结果分配空间

    for i in range(border, border + imgHeight):
        for j in range(border, border + imgWidth):
            result[i - border][j - border] = np.max(
                imgPadded[i - border:i + border + 1, j - border:j + border + 1])

    return result


def imgOpening(image, kernelSize):
    '''开运算'''

    temp = imgErosion(image, kernelSize)  # 腐蚀
    result = imgDilation(temp, kernelSize)  # 膨胀

    return result


def imgClosing(image, kernelSize):
    '''闭运算'''

    temp = imgDilation(image, kernelSize)  # 膨胀
    result = imgErosion(temp, kernelSize)  # 腐蚀

    return result


def imgTopHat(image, kernelSize):
    '''顶帽运算'''

    imgOpened = imgOpening(image, kernelSize)  # 开运算
    result = image-imgOpened

    return result


def imgBlackHat(image, kernelSize):
    '''黑帽运算'''

    imgClosed = imgClosing(image, kernelSize)  # 闭运算
    result = imgClosed-image

    return result


def imgfft2(image, kernelSize=3):
    '''图像的二维傅立叶变换'''

    border = kernelSize//2  # 图像需补全的边缘宽
    imgPadded = imgCopyBorder(image, border)  # 边缘补全像素
    result = np.fft.fft2(imgPadded)  # 傅立叶变换

    return result


def kernelfft2(image, kernel):
    '''核的补零与二维傅立叶变换'''

    imgHeight = int(image.shape[0])  # 图像高
    imgWidth = int(image.shape[1])  # 图像宽
    kernelHeight = int(kernel.shape[0])  # 核高
    kernelWidth = int(kernel.shape[1])  # 核宽
    border = int(kernel.shape[0]/2)  # 图像需补全的边缘宽

    kernelPadded = np.zeros(
        (imgHeight+2*border, imgWidth+2*border), dtype=np.float32)  # 核补零分配空间
    kernelPadded[0:kernelHeight, 0:kernelWidth] = kernel[:, :]  # 核置于左上角

    # 注意：核的位置不影响核的幅值谱，但只有核中心位于补全边缘后图像的左上角时，相乘并进行傅立叶逆变换后得到的图像正好位于中心（逆变换后需去掉补充的边缘），得到的图像与卷积结果一致。故而此处进行循环位移，使位于左上角的核满足上述条件
    kernelPadded = np.roll(kernelPadded, -border, axis=1)  # 核按列循环位移
    kernelPadded = np.roll(kernelPadded, -border, axis=0)  # 核按行循环位移

    # 补零后核的傅立叶变换
    result = np.fft.fft2(kernelPadded)

    return result


def frequencyFilter(imgfft2, kernelfft2):
    '''频域滤波'''

    # 图像与滤波器相乘
    return np.multiply(imgfft2, kernelfft2)


def imgifft2(imgfft2, kernelSize=3):
    '''图像的傅立叶逆变换'''

    height = int(imgfft2.shape[0])  # 图像高
    width = int(imgfft2.shape[1])  # 图像宽

    result = np.fft.ifft2(imgfft2)  # 滤波后图像傅立叶逆变换
    result = np.abs(result)  # 复数取绝对值
    result = np.clip(result, 0, 255)  # 0-255截断

    # 去除补充的边缘
    border = kernelSize//2
    result = result[border:height-border, border:width-border]

    result = result.astype(np.uint8)  # 转化为合理的图片输出格式

    return result


def magnitude(fft_result):
    '''幅值谱输出'''

    result = np.fft.fftshift(fft_result)  # 傅立叶变换结果中心点移至中心位置
    result = np.log(np.abs(result)+1)  # 幅值谱转化为对数形式
    result = result/result.max()*255  # 归一至0～255
    result = result.astype(np.uint8)  # 转化为合理的图片输出格式

    return result


def main():
    '''主函数'''

    # 读入图像
    imageGaussian = cv2.imread(
        'images/gaussian_noise.png', cv2.IMREAD_GRAYSCALE)
    imageOrigin = cv2.imread('images/origin_image.png', cv2.IMREAD_GRAYSCALE)
    imagePepper = cv2.imread(
        'images/pepper_noise.png', cv2.IMREAD_GRAYSCALE)

    # 3x3膨胀
    img_gaussian_dilation = imgDilation(imageGaussian, 3)
    img_origin_dilation = imgDilation(imageOrigin, 3)
    img_pepper_dilation = imgDilation(imagePepper, 3)

    # 3x3腐蚀
    img_gaussian_erosion = imgErosion(imageGaussian, 3)
    img_origin_erosion = imgErosion(imageOrigin, 3)
    img_pepper_erosion = imgErosion(imagePepper, 3)

    # 3x3顶帽运算
    img_gaussian_tophat = imgTopHat(imageGaussian, 3)
    img_origin_tophat = imgTopHat(imageOrigin, 3)
    img_pepper_tophat = imgTopHat(imagePepper, 3)

    # 3x3黑帽运算
    img_gaussian_blackhat = imgBlackHat(imageGaussian, 3)
    img_origin_blackhat = imgBlackHat(imageOrigin, 3)
    img_pepper_blackhat = imgBlackHat(imagePepper, 3)

    # 输出腐蚀、膨胀、顶帽、黑帽运算结果
    cv2.imwrite('3x3_Dilation_gaussian.png', img_gaussian_dilation)
    cv2.imwrite('3x3_Dilation_origin.png', img_origin_dilation)
    cv2.imwrite('3x3_Dilation_pepper.png', img_pepper_dilation)

    cv2.imwrite('3x3_Erosion_gaussian.png', img_gaussian_erosion)
    cv2.imwrite('3x3_Erosion_origin.png', img_origin_erosion)
    cv2.imwrite('3x3_Erosion_pepper.png', img_pepper_erosion)

    cv2.imwrite('3x3_TopHat_gaussian.png', img_gaussian_tophat)
    cv2.imwrite('3x3_TopHat_origin.png', img_origin_tophat)
    cv2.imwrite('3x3_TopHat_pepper.png', img_pepper_tophat)

    cv2.imwrite('3x3_BlackHat_gaussian.png', img_gaussian_blackhat)
    cv2.imwrite('3x3_BlackHat_origin.png', img_origin_blackhat)
    cv2.imwrite('3x3_BlackHat_pepper.png', img_pepper_blackhat)

    # 给定3张图的幅值谱
    magnitude_gaussian = magnitude(imgfft2(imageGaussian))
    magnitude_origin = magnitude(imgfft2(imageOrigin))
    magnitude_pepper = magnitude(imgfft2(imagePepper))

    # 输出幅值谱
    cv2.imwrite('magnitude_gaussian.png', magnitude_gaussian)
    cv2.imwrite('magnitude_origin.png', magnitude_origin)
    cv2.imwrite('magnitude_pepper.png', magnitude_pepper)

    # Sobel滤波卷积核
    sobelKernelV3 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobelKernelH3 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # 高斯滤波核
    gaussianKernel = GaussianKernel(3, 3)

    # 拉普拉斯核
    laplacianKernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # Sobel核（x,y方向）、高斯核（标准差为3）、拉普拉斯核的幅值谱
    magnitude_sobelKernelV3 = magnitude(kernelfft2(imageOrigin, sobelKernelV3))
    magnitude_sobelKernelH3 = magnitude(kernelfft2(imageOrigin, sobelKernelH3))
    magnitude_gaussianKernel = magnitude(
        kernelfft2(imageOrigin, gaussianKernel))
    magnitude_laplacianKernel = magnitude(
        kernelfft2(imageOrigin, laplacianKernel))

    # 输出核的幅值谱
    cv2.imwrite('magnitude_3x3_sobelKernel_y.png', magnitude_sobelKernelV3)
    cv2.imwrite('magnitude_3x3_sobelKernel_x.png', magnitude_sobelKernelH3)
    cv2.imwrite('magnitude_3x3_gaussianKernel.png', magnitude_gaussianKernel)
    cv2.imwrite('magnitude_3x3_laplacianKernel.png', magnitude_laplacianKernel)

    # 原图Sobel(y)频域滤波
    origin_sobelV3 = frequencyFilter(
        imgfft2(imageOrigin), kernelfft2(imageOrigin, sobelKernelV3))  # 频域滤波
    magnitude_origin_sobelV3 = magnitude(origin_sobelV3)  # 幅值谱
    img_origin_sobelV3 = imgifft2(origin_sobelV3)  # 滤波后图像
    cv2.imwrite('magnitude_3x3_sobel_y_origin.png',
                magnitude_origin_sobelV3)  # 输出
    cv2.imwrite('3x3_sobel_y_origin.png', img_origin_sobelV3)

    # 高斯噪声Sobel(y)频域滤波
    gaussian_sobelV3 = frequencyFilter(
        imgfft2(imageGaussian), kernelfft2(imageGaussian, sobelKernelV3))  # 频域滤波
    magnitude_gaussian_sobelV3 = magnitude(gaussian_sobelV3)  # 幅值谱
    img_gaussian_sobelV3 = imgifft2(gaussian_sobelV3)  # 滤波后图像
    cv2.imwrite('magnitude_3x3_sobel_y_gaussian.png',
                magnitude_gaussian_sobelV3)  # 输出
    cv2.imwrite('3x3_sobel_y_gaussian.png', img_gaussian_sobelV3)

    # 原图Sobel(x)频域滤波
    origin_sobelH3 = frequencyFilter(
        imgfft2(imageOrigin), kernelfft2(imageOrigin, sobelKernelH3))  # 频域滤波
    magnitude_origin_sobelH3 = magnitude(origin_sobelH3)  # 幅值谱
    img_origin_sobelH3 = imgifft2(origin_sobelH3)  # 滤波后图像
    cv2.imwrite('magnitude_3x3_sobel_x_origin.png',
                magnitude_origin_sobelH3)  # 输出
    cv2.imwrite('3x3_sobel_x_origin.png', img_origin_sobelH3)

    # 高斯噪声Sobel(x)频域滤波
    gaussian_sobelH3 = frequencyFilter(
        imgfft2(imageGaussian), kernelfft2(imageGaussian, sobelKernelH3))  # 频域滤波
    magnitude_gaussian_sobelH3 = magnitude(gaussian_sobelH3)  # 幅值谱
    img_gaussian_sobelH3 = imgifft2(gaussian_sobelH3)  # 滤波后图像
    cv2.imwrite('magnitude_3x3_sobel_x_gaussian.png',
                magnitude_gaussian_sobelH3)  # 输出
    cv2.imwrite('3x3_sobel_x_gaussian.png', img_gaussian_sobelH3)

    # 原图高斯滤波（标准差为3）频域滤波
    origin_gaussian = frequencyFilter(
        imgfft2(imageOrigin), kernelfft2(imageOrigin, gaussianKernel))  # 频域滤波
    magnitude_origin_gaussian = magnitude(origin_gaussian)  # 幅值谱
    img_origin_gaussian = imgifft2(origin_gaussian)  # 滤波后图像
    cv2.imwrite('magnitude_3x3_gaussianFilter_origin.png',
                magnitude_origin_gaussian)  # 输出
    cv2.imwrite('3x3_gaussianFilter_origin.png', img_origin_gaussian)

    # 高斯噪声高斯滤波（标准差为3）频域滤波
    gaussian_gaussian = frequencyFilter(
        imgfft2(imageGaussian), kernelfft2(imageGaussian, gaussianKernel))  # 频域滤波
    magnitude_gaussian_gaussian = magnitude(gaussian_gaussian)  # 幅值谱
    img_gaussian_gaussian = imgifft2(gaussian_gaussian)  # 滤波后图像
    cv2.imwrite('magnitude_3x3_gaussianFilter_gaussian.png',
                magnitude_gaussian_gaussian)  # 输出
    cv2.imwrite('3x3_gaussianFilter_gaussian.png', img_gaussian_gaussian)

    # 原图拉普拉斯滤波频域滤波
    origin_laplacian = frequencyFilter(
        imgfft2(imageOrigin), kernelfft2(imageOrigin, laplacianKernel))  # 频域滤波
    magnitude_origin_laplacian = magnitude(origin_laplacian)  # 幅值谱
    img_origin_laplacian = imgifft2(origin_laplacian)  # 滤波后图像
    cv2.imwrite('magnitude_3x3_laplacianFilter_origin.png',
                magnitude_origin_laplacian)  # 输出
    cv2.imwrite('3x3_laplacianFilter_origin.png', img_origin_laplacian)

    # 高斯噪声拉普拉斯滤波频域滤波
    gaussian_laplacian = frequencyFilter(
        imgfft2(imageGaussian), kernelfft2(imageGaussian, laplacianKernel))  # 频域滤波
    magnitude_gaussian_laplacian = magnitude(gaussian_laplacian)  # 幅值谱
    img_gaussian_laplacian = imgifft2(gaussian_laplacian)  # 滤波后图像
    cv2.imwrite('magnitude_3x3_laplacianFilter_gaussian.png',
                magnitude_gaussian_laplacian)  # 输出
    cv2.imwrite('3x3_laplacianFilter_gaussian.png', img_gaussian_laplacian)


if __name__ == '__main__':
    main()
