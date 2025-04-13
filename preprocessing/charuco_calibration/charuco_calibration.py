import cv2
import numpy as np
import glob
import os
import shutil

'''
No Module named 'numpy' 와 같은 에러는 pip install numpy 또는 conda install numpy (conda 환경에서)
'''

# ChArUco 보드 설정
squares_x = 9
squares_y = 6
square_length = 0.015  # 15mm
marker_length = 0.011  # 11mm

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)

# 이미지 로딩 및 코너 추출
all_corners = []
all_ids = []
image_size = None
used_images = []

images = glob.glob('././calib_images/*.jpg')
print(f"찾은 이미지 수: {len(images)}")

for fname in images:
    print(f"\n처리 중: {fname}")
    image = cv2.imread(fname)
    if image is None:
        print(f"  → 이미지를 불러올 수 없음")
        continue
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
    
    if len(corners) > 0:
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board
        )
        print(f"  → 코너 수: {retval}")
        if retval is not None and retval > 10:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            used_images.append(fname)
            if image_size is None:
                image_size = gray.shape[::-1]
        
        # # 코너가 1개 이상이라면 복사
        # if retval is not None and retval > 0:
        #     shutil.copy(fname, os.path.join(ok_folder, os.path.basename(fname)))
        # else:
        #     print(f"  → 코너 부족 또는 실패 → 제외됨")
        #     os.remove(fname) # 코너가 없는 이미지는 삭제

if len(all_corners) == 0:
    print("\n 유효한 코너가 없습니다. 사진을 다시 촬영해주세요.")
    exit()

# 3. 카메라 보정
print(f"\n 보정에 사용된 이미지 수: {len(used_images)}")
print("사용된 이미지 목록:")
for img_name in used_images:
    print(f"  - {img_name}")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_corners,
    charucoIds=all_ids,
    board=board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None
)
K = camera_matrix

# 5. 결과 출력 및 저장
output_path = "intrinsic_K.txt"
np.savetxt(output_path, K, fmt="%.8f")
print(f"\nintrinsic K 행렬 저장 완료: {output_path}")
print("\n Camera Matrix (K):\n", camera_matrix)
print("\n 왜곡 계수(Distortion Coefficients):\n", dist_coeffs)
print(f"\n Reprojection error: {ret:.4f}")

'''
출력 예시
intrinsic K 행렬 저장 완료: intrinsic_K.txt

Camera Matrix (K):
 [[1.42254156e+03 0.00000000e+00 9.58143229e+02]
 [0.00000000e+00 1.41923877e+03 5.42381047e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
'''