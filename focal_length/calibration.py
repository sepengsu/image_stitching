import cv2
import numpy as np
from itertools import combinations

class SfMCalibrator:
    def __init__(self, max_iter=100, reproj_thresh=2.0):
        self.max_iter = max_iter
        self.reproj_thresh = reproj_thresh
        self.images = []
        self.K = None
        self.rotations = []
        self.image_size = None

    def __call__(self, image_list):
        """이미지 객체 리스트를 받아 파라미터 계산"""
        self.set_images(image_list)
        return self.run_pipeline()

    def set_images(self, image_list):
        """이미지 객체 리스트 설정"""
        self.images = [img.copy() for img in image_list if img is not None]
        if not self.images:
            raise ValueError("유효한 이미지가 없습니다.")
        self._init_camera_params()

    def _init_camera_params(self):
        """초기 카메라 파라미터 설정"""
        h, w = self.images[0].shape[:2]
        self.image_size = (w, h)
        self.K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
        self.rotations = [np.eye(3)]

    def _extract_features(self):
        """특징점 추출"""
        sift = cv2.SIFT_create()
        features = []
        for img in self.images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            features.append((kp, des))
        return features

    def _match_features(self, features):
        """모든 가능한 쌍에 대해 특징점 매칭"""
        matcher = cv2.BFMatcher()
        matches = []
        n = len(features)
        
        for (i, j) in combinations(range(n), 2):
            kp1, des1 = features[i]
            kp2, des2 = features[j]
            raw_matches = matcher.knnMatch(des1, des2, k=2)
            good = [m for m, n in raw_matches if m.distance < 0.65 * n.distance]
            matches.append((i, j, kp1, kp2, good))
        
        return matches

    
    def _estimate_pose(self, kp1, kp2, matches):
        """포즈 추정"""
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, cv2.RANSAC, 0.999, self.reproj_thresh)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        return R, mask

    def _triangulate(self, R, kp1, kp2, matches, mask):
        """3D 점 재구성"""
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3,1))])
        P2 = self.K @ np.hstack([R, np.zeros((3,1))])
        
        mask = mask.ravel().astype(bool)
        if np.sum(mask) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])[mask]
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])[mask]
        
        pts1 = pts1.reshape(-1, 2).T.astype(np.float32)
        pts2 = pts2.reshape(-1, 2).T.astype(np.float32)
        
        points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
        return (points_4d[:3] / points_4d[3]).T.astype(np.float32)

    def _bundle_residual(self, params, points_3d, points_2d, camera_indices):
        """번들 조정 잔차 계산"""
        f, cx, cy = params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        residuals = []
        epsilon = 1e-8
        for i, (pt3d, pt2d) in enumerate(zip(points_3d, points_2d)):
            R = self.rotations[camera_indices[i]]
            proj = K @ (R @ pt3d.reshape(3,1))
            
            # Z 값이 너무 작거나 0인 경우 방지
            if np.abs(proj[2]) < epsilon:
                proj[2] = epsilon
            
            proj = proj[:2] / proj[2]
            residuals.append(proj.ravel() - pt2d)
        return np.concatenate(residuals)


    def _numerical_jacobian(self, params, points_3d, points_2d, camera_indices, eps=1e-6):
        """수치적 야코비안 계산"""
        n = len(params)
        J = np.zeros((len(points_2d)*2, n))
        
        for i in range(n):
            params_perturbed = params.copy()
            params_perturbed[i] += eps
            residuals_perturbed = self._bundle_residual(params_perturbed, points_3d, points_2d, camera_indices)
            J[:,i] = (residuals_perturbed - self._bundle_residual(params, points_3d, points_2d, camera_indices)) / eps
        return J

    def _lm_optimize(self, params, points_3d, points_2d, camera_indices):
        """LM 최적화 구현"""
        if isinstance(params, list):
            params = np.array(params, dtype=np.float64)
            
        x = params.astype(np.float64)
        lamb = 1e-2
        v = 2
        I = np.eye(len(x))
        
        try:
            for _ in range(self.max_iter):
                residuals = self._bundle_residual(x, points_3d, points_2d, camera_indices)
                J = self._numerical_jacobian(x, points_3d, points_2d, camera_indices)
                grad = J.T @ residuals
                hess = J.T @ J
                
                if np.linalg.norm(grad, np.inf) < 1e-5:
                    break
                    
                while True:
                    try:
                        delta = np.linalg.lstsq(hess + lamb*I, -grad, rcond=None)[0]
                    except np.linalg.LinAlgError:
                        delta = -grad / (lamb + 1e-6)
                        
                    x_new = x + delta
                    residuals_new = self._bundle_residual(x_new, points_3d, points_2d, camera_indices)
                    
                    actual_reduction = np.sum(residuals**2) - np.sum(residuals_new**2)
                    predicted_reduction = delta.T @ (lamb*delta - grad)
                    rho = actual_reduction / (predicted_reduction + 1e-8)
                    
                    if rho > 0:
                        x = x_new
                        lamb *= max(1/3, 1 - (2*rho - 1)**3)
                        v = 2
                        break
                    else:
                        lamb *= v
                        v *= 2
            return x
        except Exception as e:
            print(f"최적화 실패: {e}")
            return params

    def run_pipeline(self):
        """주 실행 파이프라인"""
        features = self._extract_features()
        matched = self._match_features(features)
        
        all_points_3d = []
        all_points_2d = []
        camera_indices = []
        
        for i, j, kp1, kp2, matches in matched:
            if len(matches) < 4:
                print(f"[경고] 매칭된 포인트가 부족합니다: {len(matches)} (이미지 {i}와 {j})")
                continue
                
            R, mask = self._estimate_pose(kp1, kp2, matches)
            points_3d = self._triangulate(R, kp1, kp2, matches, mask)
            
            if len(points_3d) == 0:
                print(f"[경고] 3D 점 재구성 실패: 이미지 {i}와 {j}")
                continue
                
            all_points_3d.append(points_3d)
            
            # 인라이어 필터링
            inlier_indices = np.where(mask.ravel())[0]
            all_points_2d.extend([kp1[matches[k].queryIdx].pt for k in inlier_indices])
            camera_indices.extend([i] * len(points_3d))
            
            # 올바른 회전 매트릭스 갱신
            if len(self.rotations) <= j:
                self.rotations.append(R @ self.rotations[i])

        if len(all_points_3d) == 0:
            raise RuntimeError("충분한 3D 포인트를 찾을 수 없습니다")
            
        initial_params = np.array([self.K[0,0], self.K[0,2], self.K[1,2]], dtype=np.float64)
        optimized_params = self._lm_optimize(
            initial_params,
            np.vstack(all_points_3d),
            np.array(all_points_2d),
            np.array(camera_indices)
        )
        
        f, cx, cy = optimized_params.astype(float)
        self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        
        return f, self.K, self.rotations


