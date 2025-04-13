import numpy as np

def compute_hfov_from_dfov(dfov_deg, width, height):
    """
    DFOV (대각선 시야각)에서 HFOV (수평 시야각) 계산
    
    Args:
        dfov_deg: 대각선 시야각 (도)
        width: 이미지 너비 (픽셀)
        height: 이미지 높이 (픽셀)
        
    Returns:
        hfov_deg: 수평 시야각 (도)
    """
    # 종횡비 계산
    aspect_ratio = width / height
    
    # dfov를 라디안으로 변환
    dfov_rad = np.radians(dfov_deg)
    
    # hfov 계산 공식
    tan_dfov_half = np.tan(dfov_rad / 2)
    factor = np.sqrt(1 + 1 / (aspect_ratio**2))
    tan_hfov_half = tan_dfov_half / factor
    hfov_rad = 2 * np.arctan(tan_hfov_half)
    
    # 도 단위로 변환
    hfov_deg = np.degrees(hfov_rad)
    
    return hfov_deg

def compute_vfov_from_hfov(hfov_deg, width, height):
    """
    HFOV (수평 시야각)에서 VFOV (수직 시야각) 계산
    
    Args:
        hfov_deg: 수평 시야각 (도)
        width: 이미지 너비 (픽셀)
        height: 이미지 높이 (픽셀)
        
    Returns:
        vfov_deg: 수직 시야각 (도)
    """
    # 종횡비 계산
    aspect_ratio = width / height
    
    # hfov를 라디안으로 변환
    hfov_rad = np.radians(hfov_deg)
    
    # vfov 계산 공식
    tan_hfov_half = np.tan(hfov_rad / 2)
    tan_vfov_half = tan_hfov_half / aspect_ratio
    vfov_rad = 2 * np.arctan(tan_vfov_half)
    
    # 도 단위로 변환
    vfov_deg = np.degrees(vfov_rad)
    
    return vfov_deg

def compute_K_from_hfov(hfov_deg, width, height):
    """
    HFOV (수평 시야각)과 해상도 기준으로 Intrinsic Matrix 계산
    
    Args:
        hfov_deg: 수평 시야각 (도)
        width: 이미지 너비 (픽셀)
        height: 이미지 높이 (픽셀)
        
    Returns:
        K: 4x4 내부 파라미터 행렬 (픽셀 단위)
    """
    hfov_rad = np.radians(hfov_deg)
    
    # 초점 거리 계산 (픽셀 단위) - 수평 화각 기준
    f = (width / 2) / np.tan(hfov_rad / 2)
    
    # 주점(principal point) - 일반적으로 이미지 중심
    cx = width / 2
    cy = height / 2
    
    # 4x4 K 행렬 생성
    K = np.array([
        [f, 0, cx, 0],
        [0, f, cy, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    return K

def compute_K_from_dfov(dfov_deg, width, height):
    """
    DFOV (대각선 시야각)과 해상도 기준으로 Intrinsic Matrix 계산
    - 대각선 시야각을 수평 시야각으로 변환 후 계산
    
    Args:
        dfov_deg: 대각선 시야각 (도)
        width: 이미지 너비 (픽셀)
        height: 이미지 높이 (픽셀)
        
    Returns:
        K: 4x4 내부 파라미터 행렬 (픽셀 단위)
    """
    # dfov에서 hfov 계산
    hfov_deg = compute_hfov_from_dfov(dfov_deg, width, height)
    
    # hfov로 K 행렬 계산
    K = compute_K_from_hfov(hfov_deg, width, height)
    
    return K, hfov_deg

def scale_K(K, orig_width, orig_height, new_width, new_height):
    """
    입력 해상도 기준으로 Intrinsic Matrix 리스케일
    
    Args:
        K: 원본 내부 파라미터 행렬
        orig_width: 원본 이미지 너비 (픽셀)
        orig_height: 원본 이미지 높이 (픽셀)
        new_width: 새 이미지 너비 (픽셀)
        new_height: 새 이미지 높이 (픽셀)
        
    Returns:
        K_scaled: 스케일된 내부 파라미터 행렬
    """
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height
    
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[1, 2] *= scale_y  # cy
    
    return K_scaled

def verify_calculations(dfov_deg, width, height):
    """
    계산이 정확한지 검증 (dfov -> hfov -> dfov)
    
    Args:
        dfov_deg: 원본 대각선 시야각 (도)
        width: 이미지 너비 (픽셀)
        height: 이미지 높이 (픽셀)
    """
    # dfov에서 hfov와 vfov 계산
    hfov_deg = compute_hfov_from_dfov(dfov_deg, width, height)
    vfov_deg = compute_vfov_from_hfov(hfov_deg, width, height)
    
    # hfov와 vfov에서 dfov 역산
    hfov_rad = np.radians(hfov_deg)
    vfov_rad = np.radians(vfov_deg)
    
    tan_hfov_half = np.tan(hfov_rad / 2)
    tan_vfov_half = np.tan(vfov_rad / 2)
    tan_dfov_half = np.sqrt(tan_hfov_half**2 + tan_vfov_half**2)
    
    calculated_dfov = 2 * np.degrees(np.arctan(tan_dfov_half))
    
    print("검증 결과:")
    print(f"- 원본 DFOV: {dfov_deg} 도")
    print(f"- 계산된 HFOV: {hfov_deg:.2f} 도")
    print(f"- 계산된 VFOV: {vfov_deg:.2f} 도")
    print(f"- 역산된 DFOV: {calculated_dfov:.2f} 도")
    print(f"- 오차: {abs(dfov_deg - calculated_dfov):.6f} 도")

if __name__ == "__main__":
    # 입력 파라미터
    dfov = 120                 # 대각선 시야각 (도)
    orig_width = 1280          # 원본 이미지 가로 해상도 (px)
    orig_height = 720          # 원본 이미지 세로 해상도 (px)
    target_width = 640         # 모델 입력 가로 해상도
    target_height = 192        # 모델 입력 세로 해상도
    
    # 종횡비 계산 및 출력
    aspect_ratio = orig_width / orig_height
    print(f"이미지 종횡비(가로:세로): {aspect_ratio:.4f}")
    
    # DFOV에서 HFOV 계산
    hfov = compute_hfov_from_dfov(dfov, orig_width, orig_height)
    vfov = compute_vfov_from_hfov(hfov, orig_width, orig_height)
    
    print(f"변환 결과:")
    print(f"- DFOV: {dfov} 도")
    print(f"- HFOV: {hfov:.2f} 도")
    print(f"- VFOV: {vfov:.2f} 도")
    
    # 계산이 정확한지 검증
    verify_calculations(dfov, orig_width, orig_height)
    
    # HFOV 기준 Intrinsic Matrix 계산
    K_hfov = compute_K_from_hfov(hfov, orig_width, orig_height)
    print("\nHFOV 기준 K 행렬:")
    print(K_hfov)
    
    # DFOV 기준 Intrinsic Matrix 계산 (기존 방식)
    K_dfov_original = compute_K_from_dfov(dfov, orig_width, orig_height)
    print("\nDFOV 기준 K 행렬 (구방식):")
    dfov_rad = np.radians(dfov)
    diag_res = np.sqrt(orig_width**2 + orig_height**2)
    f_dfov = diag_res / (2 * np.tan(dfov_rad / 2))
    print(f"- DFOV 기준 초점 거리: {f_dfov:.2f} 픽셀")
    
    # DFOV -> HFOV 변환 후 K 행렬 계산 (새 방식)
    K_from_dfov, calculated_hfov = compute_K_from_dfov(dfov, orig_width, orig_height)
    print("\nDFOV -> HFOV 변환 후 K 행렬 (신방식):")
    print(K_from_dfov)
    print(f"- HFOV 기준 초점 거리: {K_from_dfov[0,0]:.2f} 픽셀")
    
    # 640x192 해상도 기준으로 리스케일
    K_scaled = scale_K(K_from_dfov, orig_width, orig_height, target_width, target_height)
    print("\n리스케일된 K (640x192 기준):")
    print(K_scaled)
    
    # 파일로 저장
    np.savetxt("hfov_K.txt", K_from_dfov, fmt="%.8f")
    np.savetxt("hfov_scaled_K.txt", K_scaled, fmt="%.8f")