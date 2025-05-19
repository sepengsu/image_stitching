
import cv2
import numpy as np
from itertools import combinations
from typing import List, Optional, Union

class SphericalFocalLengthEstimator:
    """
    SphericalFocalLengthEstimator: 이미지 스티칭에서 구형 투영을 위한 초점 거리 (f) 추정을 위한 클래스.
    """
    
    def __init__(self, method: str = 'fundamental_matrix') -> None:
        self.method = method
    
    def __call__(self, imgs: List[np.ndarray], use_median: bool = True) -> float:
        """
        SphericalFocalLengthEstimator 객체를 호출하여 초점 거리 추정.
        
        Parameters:
            imgs (List[np.ndarray]): 이미지 리스트.
            use_median (bool): 여러 이미지에서 초점 거리 추정 시 중앙값 사용 여부.
        
        Returns:
            float: 추정된 초점 거리.
        """
        if len(imgs) < 2:
            raise ValueError("최소 두 개의 이미지가 필요합니다.")
        
        f_values = []
        
        for img1, img2 in combinations(imgs, 2):
            f = self.estimate_fundamental_matrix(img1, img2)
            f_values.append(f)
        
        if use_median:
            return np.median(f_values)
        else:
            return np.mean(f_values)
    def estimate_fundamental_matrix(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Fundamental Matrix 방법을 사용하여 두 이미지 사이의 초점 거리 추정.
        """
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        
        if F is None or F.shape != (3, 3):
            raise ValueError("Fundamental Matrix 계산에 실패했습니다.")
        
        epipolar_constraints = np.abs(np.sum(pts2 @ F @ pts1.T, axis=2))
        f_estimated = np.sqrt(1 / (2 * np.mean(epipolar_constraints)))
        
        return f_estimated
    
    def spherical_projection(self, img: np.ndarray, f: float) -> np.ndarray:
        """
        구형 투영을 적용하여 이미지 변환.
        """
        h, w = img.shape
        cx, cy = w // 2, h // 2
        output = np.zeros_like(img)
        
        for y in range(h):
            for x in range(w):
                theta = (x - cx) / f
                phi = (y - cy) / f
                X = np.sin(theta) * np.cos(phi)
                Y = np.sin(phi)
                Z = np.cos(theta) * np.cos(phi)
                x_new = int(f * (X / Z) + cx)
                y_new = int(f * (Y / Z) + cy)
                
                if 0 <= x_new < w and 0 <= y_new < h:
                    output[y_new, x_new] = img[y, x]
        
        return output
