import cv2
import numpy as np

def gaussian_blur_overlay(image_path, blur_kernel_size=(9, 9), blur_sigma_x=1.7):
  """
  对图像应用高斯模糊叠加效果。

  参数：
    image_path (str): 图像文件路径。
    blur_kernel_size (tuple): 高斯模糊内核大小（宽度，高度）。
    blur_sigma_x (int): 高斯模糊在 X 方向上的标准差。
  """
  # 读取图像
  image = cv2.imread(image_path)

  if image is None:
    print(f"错误：无法读取图像 {image_path}")
    return

  # 应用高斯模糊
  blurred_image = cv2.GaussianBlur(image, blur_kernel_size, blur_sigma_x)

  # 叠加原始图像和模糊图像
  alpha = 0.1  # 原始图像的权重
  beta = 1 - alpha  # 模糊图像的权重
  gamma = 0  # 亮度调整
  overlayed_image = cv2.addWeighted(image, alpha, blurred_image, beta, gamma)

  try:
    cv2.imwrite("test.jpg", overlayed_image)
  except Exception as e:
    print(f"保存图像时出错：{e}")

# 示例用法
image_file = "outputs/rebuttal/attack/llava_vlguard/attack/bad_prompt_temp_5000.bmp"  # 替换为你的图像路径
gaussian_blur_overlay(image_file)