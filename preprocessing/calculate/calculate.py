import numpy as np
import math

def calculate_intrinsic_matrix(sensor_width_mm, sensor_height_mm, focal_length_mm, image_resolution):
    """
    센서 크기, 초점 거리, 이미지 해상도를 이용하여 카메라 내부 파라미터 행렬을 계산합니다.
    
    Parameters:
    -----------
    sensor_width_mm : float
        센서의 가로 크기(mm)
    sensor_height_mm : float
        센서의 세로 크기(mm)
    focal_length_mm : float
        렌즈의 초점 거리(mm)
    image_resolution : tuple
        이미지 해상도 (width, height)
        
    Returns:
    --------
    numpy.ndarray
        카메라 내부 파라미터(intrinsic K) 행렬
    """
    
    width, height = image_resolution
    
    # 픽셀 크기 계산
    pixel_size_width = sensor_width_mm / width
    pixel_size_height = sensor_height_mm / height
    
    # 초점 거리(픽셀 단위) 
    fx = focal_length_mm / pixel_size_width
    fy = focal_length_mm / pixel_size_height
    
    # 주점(principal point) 좌표 (일반적으로 이미지 중심)
    cx = width / 2
    cy = height / 2
    
    # 내부 파라미터 행렬 구성
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return K


sensor_width_mm = 8.0  # 실제 센서 가로 크기(mm)
sensor_height_mm = 6.0  # 실제 센서 세로 크기(mm)
focal_length_mm = 5.9  # 초점 거리(mm)
width = 1920  # 이미지 가로 해상도(픽셀)
height = 1080  # 이미지 세로 해상도(픽셀)

print(f"센서 크기: {sensor_width_mm:.2f}mm x {sensor_height_mm:.2f}mm")

K = calculate_intrinsic_matrix(sensor_width_mm, sensor_height_mm, focal_length_mm, (width, height))

print(f"\n계산된 내부 파라미터(Intrinsic K) 행렬:\n{K}")
np.savetxt("intrinsic_K.txt", K, fmt="%.8f")