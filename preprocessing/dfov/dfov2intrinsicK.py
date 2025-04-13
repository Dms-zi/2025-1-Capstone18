import numpy as np

def compute_K_from_dfov(dfov_deg, width, height):
    """
    DFOV (대각선 시야각)과 해상도 기준으로 Intrinsic Matrix 계산
    - 출력: 4x4 K 행렬 (픽셀 단위)
    """
    dfov_rad = np.radians(dfov_deg)
    diag_res = np.sqrt(width**2 + height**2)
    f = diag_res / (2 * np.tan(dfov_rad / 2))  # focal length in pixels

    cx = width / 2
    cy = height / 2

    K = np.array([
        [f, 0, cx, 0],
        [0, f, cy, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    return K

def scale_K(K, orig_width, orig_height, new_width, new_height):
    """
    입력 해상도 기준으로 Intrinsic Matrix 리스케일
    """
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height

    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[1, 2] *= scale_y  # cy

    return K_scaled

if __name__ == "__main__":
    dfov = 120                 # 대각선 시야각 -> 해당 폰 스펙 참존존
    orig_width = 1280          # 원본 이미지 가로 해상도 (px)
    orig_height = 720          # 원본 이미지 세로 해상도 (px)
    target_width = 640         # 모델 입력 가로 해상도
    target_height = 192        # 모델 입력 세로 해상도

    # DFOV 기준 Intrinsic Matrix 계산
    K_original = compute_K_from_dfov(dfov, orig_width, orig_height)
    print("원본 해상도 기준 K:\n", K_original)

    # 640x192 해상도 기준으로 리스케일
    K_scaled = scale_K(K_original, orig_width, orig_height, target_width, target_height)
    print("\n리스케일된 K (640x192 기준):\n", K_scaled)

    np.savetxt("origin_intrinsics.txt", K_original, fmt="%.8f")
    np.savetxt("scaled_intrinsics.txt", K_scaled, fmt="%.8f")
