import cv2
import numpy as np
from typing import List, Union

class FocalLengthEstimator:
    """
    FocalLengthEstimator: 이미지 스티칭에서 원통형 투영을 위한 초점 거리 (f) 추정을 위한 통합 클래스.
    """
    
    def __init__(self, focal_ratio: float = 1.2, fov_deg: float = 60, max_iterations: int = 10, diff_threshold: float = 0.1, max_threshold: float = 5000, min_threshold: float = 500, use_median: bool = True) -> None:
        """
        FocalLengthEstimator 객체 초기화.
        
        Parameters:
            focal_ratio (float): OpenMVG 방식의 초점 비율 (기본값: 1.2).
            fov_deg (float): 초기 f 추정을 위한 수평 시야각 (기본값: 60도).
            max_iterations (int): f 최적화를 위한 최대 반복 횟수 (기본값: 10).
            diff_threshold (float): 수렴 판단을 위한 f 변화 허용치 (기본값: 0.1).
            max_threshold (float): f의 최대 허용치 (기본값: 5000).
            min_threshold (float): f의 최소 허용치 (기본값: 500).
            use_median (bool): 여러 이미지에서 초점 거리 추정 시 중앙값 사용 여부.
        """
        self.focal_ratio = focal_ratio
        self.fov_deg = fov_deg
        self.max_iterations = max_iterations
        self.diff_threshold = diff_threshold
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.use_median = use_median
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    def _estimate_initial_f_values(self, images: List[np.ndarray]) -> List[float]:
        """
        OpenMVG 방식을 사용하여 초기 f 값을 추정.
        
        Parameters:
            images (List[np.ndarray]): 이미지 리스트.
        
        Returns:
            List[float]: 각 이미지의 초기 f 값 리스트.
        """
        f_values = []
        for img in images:
            height, width = img.shape[:2]
            
            # OpenMVG 방식의 초기 f 추정
            f_estimated = self.focal_ratio * max(width, height)
            f_estimated = max(self.min_threshold, min(f_estimated, self.max_threshold))
            
            print(f"[INFO] Image Size: {width}x{height}, Estimated f: {f_estimated:.2f} pixels")
            f_values.append(f_estimated)
        
        return f_values
    
    def cylindrical_projection(self, img: np.ndarray, f: float) -> np.ndarray:
        """
        원통 좌표계로 이미지 변환.
        """
        H, W = img.shape[:2]
        cx, cy = W // 2, H // 2
        
        # 빈 출력 이미지 생성
        cylindrical_img = np.zeros_like(img)
        
        for y in range(H):
            for x in range(W):
                # 원통 좌표 계산
                theta = np.arctan((x - cx) / f)
                h = (y - cy) / np.sqrt((x - cx) ** 2 + f ** 2)
                
                # 새로운 좌표 계산
                x_new = int(f * theta + cx)
                y_new = int(f * h + cy)
                
                # 범위 내의 좌표만 복사
                if 0 <= x_new < W and 0 <= y_new < H:
                    cylindrical_img[y_new, x_new] = img[y, x]
        
        return cylindrical_img
    
    def optimize_f_values(self, images: List[np.ndarray]) -> List[float]:
        """
        초기 f 값을 최적화 (향후 확장 가능).
        
        Parameters:
            images (List[np.ndarray]): 이미지 리스트.
        
        Returns:
            List[float]: 최적화된 f 값 리스트.
        """
        # 현재는 초기값을 그대로 반환 (추가 최적화 가능)
        return self._estimate_initial_f_values(images)
    
    def __call__(self, images: Union[np.ndarray, List[np.ndarray]]) -> List[float]:
        """
        FocalLengthEstimator 객체를 호출하여 최적화된 f 값 반환.
        
        Parameters:
            images (np.ndarray or List[np.ndarray]): 단일 이미지 또는 이미지 리스트.
        
        Returns:
            List[float]: 최적화된 f 값 리스트.
        """
        # 단일 이미지가 입력된 경우 리스트로 변환
        if isinstance(images, np.ndarray):
            images = [images]
        
        # f 최적화
        optimized_f_values = self.optimize_f_values(images)
        
        return optimized_f_values
