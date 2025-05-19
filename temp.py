class Projection:
    def __init__(self, projection_method="cylindrical", focal_length=500):
        self.projection_method = projection_method.lower()
        self.focal_length = focal_length
        
        # 허용된 프로젝션 방법 체크
        if self.projection_method not in ["cylindrical", "spherical", "none"]:
            raise ValueError("잘못된 프로젝션 방법입니다. 'cylindrical', 'spherical', 'none' 중 하나를 선택하세요.")

    def set_focal_length(self, focal_length):
        if focal_length <= 0:
            raise ValueError("Focal length는 양수여야 합니다.")
        self.focal_length = focal_length

    def project_images(self, images):
        """
        각 이미지에 대해 지정된 투영 방법을 적용
        
        Parameters:
        - images (list): 투영할 이미지 리스트
        
        Returns:
        - list: 투영된 이미지 리스트
        """
        if self.projection_method == "none":
            # 프로젝션 없이 원본 이미지 반환
            return images
        elif self.projection_method == "cylindrical":
            return [self.cylindrical_projection(img) for img in images]
        elif self.projection_method == "spherical":
            return [self.spherical_projection(img) for img in images]
        else:
            raise ValueError("지원되지 않는 프로젝션 방법입니다.")

    def cylindrical_projection(self, img):
        """
        원통형 투영을 적용하여 이미지 변환
        
        Parameters:
        - img (np.ndarray): 원본 이미지
        
        Returns:
        - np.ndarray: 투영된 이미지
        """
        h, w = img.shape[:2]
        f = self.focal_length
        center_x = w // 2
        center_y = h // 2
        
        # 투영할 출력 이미지 초기화
        projected = np.zeros_like(img)

        # 각 픽셀에 대해 투영
        for y in range(h):
            for x in range(w):
                # 원통형 좌표계 변환
                theta = np.arctan((x - center_x) / f)
                h_ = (y - center_y) / np.sqrt((x - center_x) ** 2 + f ** 2)
                
                # 원래 좌표로 변환
                x_new = int(f * theta + center_x)
                y_new = int(f * h_ + center_y)
                
                # 범위 내일 때만 픽셀 복사
                if 0 <= x_new < w and 0 <= y_new < h:
                    projected[y_new, x_new] = img[y, x]
        
        return projected

    def spherical_projection(self, img):
        """
        구형 투영을 적용하여 이미지 변환
        
        Parameters:
        - img (np.ndarray): 원본 이미지
        
        Returns:
        - np.ndarray: 투영된 이미지
        """
        h, w = img.shape[:2]
        f = self.focal_length
        center_x = w // 2
        center_y = h // 2
        
        # 투영할 출력 이미지 초기화
        projected = np.zeros_like(img)

        # 각 픽셀에 대해 투영
        for y in range(h):
            for x in range(w):
                # 구형 좌표계 변환
                theta = np.arctan((x - center_x) / f)
                phi = np.arctan((y - center_y) / f)
                
                x_new = int(f * np.tan(theta) + center_x)
                y_new = int(f * np.tan(phi) + center_y)
                
                # 범위 내일 때만 픽셀 복사
                if 0 <= x_new < w and 0 <= y_new < h:
                    projected[y_new, x_new] = img[y, x]
        
        return projected