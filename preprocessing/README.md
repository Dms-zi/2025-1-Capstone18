
# charuco_calibration
## 출력 예시
python charuco_calibration.py
   
Camera Matrix (K):  
 [[1.42254156e+03 0.00000000e+00 9.58143229e+02]  
 [0.00000000e+00 1.41923877e+03 5.42381047e+02 ]  
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]  
  
  
# hfov로 카메라 내부행렬 구하기
## 출력 예시
python hfov2intrinsicK.py
(DL) python preprocessing/hfov/hfov2intrinsicK.py
이미지 종횡비(가로:세로): 1.7778  
변환 결과:  
- DFOV: 120 도  
- HFOV: 112.96 도  
- VFOV: 80.67 도  
검증 결과:   
- 원본 DFOV: 120 도  
- 계산된 HFOV: 112.96 도  
- 계산된 VFOV: 80.67 도  
- 역산된 DFOV: 120.00 도  
- 오차: 0.000000  도  
   
HFOV 기준 K 행렬:  
[[423.94968   0.      640.        0.     ]  
 [  0.      423.94968 360.        0.     ]  
 [  0.        0.        1.        0.     ]  
 [  0.        0.        0.        1.     ]]  
  
DFOV 기준 K 행렬 (구방식):  
- DFOV 기준 초점 거리: 423.95 픽셀  

DFOV -> HFOV 변환 후 K 행렬 (신방식):  
[[423.94968   0.      640.        0.     ]  
 [  0.      423.94968 360.        0.     ]  
 [  0.        0.        1.        0.     ]  
 [  0.        0.        0.        1.     ]]  
- HFOV 기준 초점 거리: 423.95 픽셀  

리스케일된 K (640x192 기준):  
[[211.97484    0.       320.         0.      ]  
 [  0.       113.053246  96.         0.      ]  
 [  0.         0.         1.         0.      ]  
 [  0.         0.         0.         1.      ]]  