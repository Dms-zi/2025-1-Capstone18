import cv2
import numpy as np
import glob
import os

'''
No Module named 'numpy' 와 같은 에러는 pip install numpy 혹은
 conda install numpy(conda 환경에서) 로 해결하세요
'''

# 1. ChArUco 보드 설정
squares_x = 9
squares_y = 6
square_length = 0.015
marker_length = 0.011

# OpenCV 4.x 에서 
try:
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, aruco_dict)
except AttributeError:
    # OpenCV 4.7 이상에서만만
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard.create(squares_x, squares_y, square_length, marker_length, aruco_dict)

# 2. 이미지로딩하고 코너찾음음
all_corners = []
all_ids = []
image_size = None
images = glob.glob('calib_images/*.jpg')  # 체커보드 촬영한 사진 저장한 곳

print(f"찾은 이미지 수: {len(images)}")

for fname in images:
    print(f"처리 중: {fname}")
    image = cv2.imread(fname)
    if image is None:
        print(f"이미지를 불러올 수 없음: {fname}")
        continue
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # OpenCV 버전에 따라서 나눴음
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict)
    
    if len(corners) > 0:
        try:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)
        except ValueError:
            # opencv가 4.7이상일때만 
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board, None, None)
        
        if retval is not None and retval > 10:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            if image_size is None:
                image_size = gray.shape[::-1]
    else:
        print(f"마커를 찾을 수 없음: {fname}")

if len(all_corners) == 0:
    print("코너가 없음 사진촬영을 정확하게 다시해보세요.")
    exit()

# 3. 보정 (calibration)
print(f"보정에 사용할 이미지 수: {len(all_corners)}")

try:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, image_size, None, None)
except Exception as e:
    print(f"보정 중 오류 발생: {e}")
    # 대체 방법 시도
    print("대체 방법으로 보정 시도")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners, charucoIds=all_ids, board=board, 
        imageSize=image_size, cameraMatrix=None, distCoeffs=None)

# 4. 정규화 K 계산 (Monodepth2용)
w, h = image_size
fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

K_normalized = np.array([
    [fx / w, 0, cx / w, 0],
    [0, fy / h, cy / h, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.float32)

# 5. txt파일에 저장
output_path = "K_matrix_normalized.txt"
np.savetxt(output_path, K_normalized, fmt="%.8f")

print(f"\n정규화 K 행렬 저장 완료: {output_path}")
print("\nCamera Matrix (K):\n", camera_matrix)
print("\nMonodepth2용 정규화 K 행렬:\n", K_normalized)

# 왜곡계수는 참고용으로만
print("\n왜곡 계수(Distortion Coefficients):\n", dist_coeffs)

'''
출력 예시
정규화 K 행렬 저장 완료: K_matrix_normalized.txt

Camera Matrix (K):
 [[1.42254156e+03 0.00000000e+00 9.58143229e+02]
 [0.00000000e+00 1.41923877e+03 5.42381047e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

Monodepth2용 정규화 K 행렬:
 [[0.74182206 0.00000000 0.49903251 0.00000000]
 [0.00000000 0.74093583 0.50220468 0.00000000]
 [0.00000000 0.00000000 1.00000000 0.00000000]
 [0.00000000 0.00000000 0.00000000 1.00000000]]
'''