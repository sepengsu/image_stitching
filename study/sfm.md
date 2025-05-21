# SfMCalibrator 알고리즘 리뷰

## 1. 개요

SfM (Structure from Motion) Calibrator는 다중 시점에서 촬영된 이미지들로부터 카메라의 \*\*내부 파라미터 (K)\*\*와 \*\*외부 파라미터 (R, t)\*\*를 추정하는 파이프라인입니다. 이 알고리즘은 주로 3D 복원, 카메라 추적, 파노라마 스티칭 등 다양한 컴퓨터 비전 응용 분야에서 사용됩니다.

---

## 2. 주요 구성 요소

### 2.1 초기화 (`__init__`)

* **`max_iter`**: 최대 반복 횟수
* **`reproj_thresh`**: RANSAC을 통한 기본 행렬 추정 시의 재투영 오류 허용 범위
* **`images`**: 입력 이미지 목록
* **`K`**: 카메라 내부 파라미터
* **`rotations`**: 각 이미지에 대한 회전 행렬 리스트
* **`image_size`**: 이미지 크기 (너비, 높이)

### 2.2 이미지 설정 (`set_images`)

* 이미지의 복사본을 \*\*`self.images`\*\*에 저장하고 카메라 파라미터를 초기화합니다.

```python
self.images = [img.copy() for img in image_list if img is not None]
if not self.images:
    raise ValueError("유효한 이미지가 없습니다.")
```

### 2.3 카메라 초기 파라미터 설정 (`_init_camera_params`)

* 이미지 크기를 기준으로 기본 카메라 매트릭스를 설정합니다.

```python
h, w = self.images[0].shape[:2]
self.K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
self.rotations = [np.eye(3)]
```

### 2.4 특징점 추출 (`_extract_features`)

* SIFT를 사용하여 각 이미지에서 특징점과 디스크립터를 추출합니다.

```python
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)
```

### 2.5 특징점 매칭 (`_match_features`)

* BFMatcher를 이용하여 각 이미지 쌍 사이의 특징점을 매칭합니다.

```python
matcher = cv2.BFMatcher()
raw_matches = matcher.knnMatch(des1, des2, k=2)
good = [m for m, n in raw_matches if m.distance < 0.7 * n.distance]
```

### 2.6 포즈 추정 (`_estimate_pose`)

* 에센셜 매트릭스를 계산하여 각 이미지 쌍 간의 회전 행렬 R과 변환 벡터 t를 추정합니다.

```python
E, mask = cv2.findEssentialMat(pts1, pts2, self.K, cv2.RANSAC, 0.999, self.reproj_thresh)
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
```

### 2.7 3D 점 재구성 (`_triangulate`)

* 두 카메라 뷰 사이에서 3D 점을 재구성합니다.

```python
points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
return (points_4d[:3] / points_4d[3]).T.astype(np.float32)
```

### 2.8 번들 조정 (`_bundle_residual`)

* 초기 카메라 파라미터를 개선하기 위해 번들 조정을 수행합니다.

```python
proj = K @ (R @ pt3d.reshape(3,1))
proj = proj[:2] / proj[2]
```

### 2.9 수치적 야코비안 계산 (`_numerical_jacobian`)

* 파라미터에 대한 수치적 야코비안을 계산하여 LM 최적화에 사용합니다.

```python
J[:,i] = (residuals_perturbed - self._bundle_residual(params, points_3d, points_2d, camera_indices)) / eps
```

### 2.10 LM 최적화 (`_lm_optimize`)

* Levenberg-Marquardt 알고리즘을 사용하여 카메라 파라미터를 최적화합니다.

```python
x_new = x + delta
```

### 2.11 파이프라인 실행 (`run_pipeline`)

* 전체 워크플로를 실행하여 최적화된 \*\*`f`, `K`, `R`\*\*을 반환합니다.

```python
f, cx, cy = optimized_params.astype(float)
self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
```

---

## 3. 알고리즘의 장점

* **효율성**: SIFT를 이용한 강력한 특징점 검출
* **정확성**: RANSAC 기반의 에센셜 매트릭스 추정
* **확장성**: 번들 조정을 통한 최적화 가능

---

## 4. 개선점

* **정확도 향상**: 더 강력한 outlier 제거 기법 추가 가능
* **속도 개선**: 병렬 처리를 통한 특징점 추출 최적화
* **추가 기능**: 비선형 왜곡 보정 추가

---

## 5. 참고 문헌

* OpenCV: [https://docs.opencv.org](https://docs.opencv.org)
* Szeliski, R. (2010). Computer Vision: Algorithms and Applications. Springer.
* Hartley, R., & Zisserman, A. (2003). Multiple View Geometry in Computer Vision. Cambridge University Press.
