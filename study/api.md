# 고급 이미지 스티칭 알고리즘 가이드

## 개요

이 가이드는 OpenCV의 Stitcher API를 사용하여 Python에서 고급 이미지 스티칭 알고리즘을 구현하는 방법을 설명합니다. 여러 이미지를 하나의 파노라마로 결합하는 주요 단계인 특징점 추출, 매칭, 호모그래피 추정 및 블렌딩 과정을 다룹니다.

---

## 1. 프로그램 개요

* **목적:** 여러 입력 이미지를 정렬하고 결합하여 하나의 파노라마 이미지를 생성합니다.
* **주요 라이브러리:** OpenCV (cv2), NumPy (np), argparse.
* **기능:** 다양한 특징 추출기, 투영 방법 및 노출 보정 기술을 지원합니다.

---

## 2. 주요 처리 단계

### 2.1 인자 파서 설정

* 실행 시 다양한 매개변수를 설정할 수 있도록 구성합니다.
* 주요 설정:

  * **이미지 입력 (`--img_names`)**: 스티칭할 이미지 파일 목록.
  * **특징점 추출기 (`--features`)**: SIFT, ORB, BRISK, AKAZE.
  * **투영 방법 (`--warp`)**: 구면, 평면, 원통형, 어안 렌즈 등.
  * **블렌딩 방법 (`--blend`)**: 멀티밴드, 페더, 기본 블렌딩.

```python
parser = argparse.ArgumentParser(description="고급 이미지 스티칭")
parser.add_argument('img_names', nargs='+', help="스티칭할 파일 목록", type=str)
parser.add_argument('--features', default='orb', choices=['sift', 'orb', 'brisk', 'akaze'], type=str)
parser.add_argument('--warp', default='spherical', choices=['spherical', 'plane', 'affine', 'cylindrical', 'fisheye'])
parser.add_argument('--blend', default='multiband', choices=['multiband', 'feather', 'no'], type=str)
```

---

### 2.2 이미지 로딩 및 스케일링

* 입력 이미지를 읽어와 작업 해상도 (`--work_megapix`)에 따라 스케일링합니다.

```python
full_img = cv.imread(cv.samples.findFile(name))
work_scale = np.sqrt(work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1]))
img = cv.resize(full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
```

---

### 2.3 특징점 추출

* 지정된 특징점 추출 방법을 사용하여 각 이미지의 특징점과 디스크립터를 추출합니다.
* 지원 방법: SIFT, ORB, BRISK, AKAZE.

```python
finder = FEATURES_FIND_CHOICES[args.features]()
img_feat = cv.detail.computeImageFeatures2(finder, img)
```

---

### 2.4 특징점 매칭

* 추출된 특징점을 기반으로 이미지 쌍을 매칭합니다.
* 호모그래피 또는 아핀 변환 매칭 지원.

```python
matcher = get_matcher(args)
p = matcher.apply2(features)
matcher.collectGarbage()
```

---

### 2.5 호모그래피 추정

* 특징점 매칭 결과를 기반으로 카메라 매개변수를 추정합니다.

```python
estimator = ESTIMATOR_CHOICES[args.estimator]()
b, cameras = estimator.apply(features, p, None)
```

---

### 2.6 번들 조정

* 카메라의 초점 거리, 왜곡 등을 최적화하여 왜곡을 줄입니다.

```python
adjuster = BA_COST_CHOICES[args.ba]()
adjuster.setConfThresh(conf_thresh)
adjuster.apply(features, p, cameras)
```

---

### 2.7 워핑 및 정렬

* 계산된 매개변수를 기반으로 이미지를 정렬하고 왜곡을 보정합니다.

```python
warper = cv.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)
```

---

### 2.8 노출 보정

* 이미지의 밝기 차이를 보정하여 매끄러운 연결을 만듭니다.

```python
compensator = get_compensator(args)
compensator.feed(corners, images_warped, masks_warped)
```

---

### 2.9 시접 연결

* 이미지 간의 경계를 자연스럽게 연결합니다.

```python
seam_finder = SEAM_FIND_CHOICES[args.seam]
masks_warped = seam_finder.find(images_warped_f, corners, masks_warped)
```

---

### 2.10 블렌딩 및 최종 합성

* 최종적으로 이미지를 합성하여 하나의 파노라마 이미지를 생성합니다.

```python
blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])
result, result_mask = blender.blend(None, None)
```

---

### 2.11 결과 저장

* 최종 파노라마 이미지를 파일로 저장합니다.

```python
cv.imwrite(result_name, result)
```

---

## 3. 실행 예시

```bash
python stitching_detailed.py img1.jpg img2.jpg img3.jpg --features sift --warp cylindrical --blend multiband
```

---

## 4. 주요 고려 사항

* **메모리 관리**: 큰 이미지 세트를 처리할 때 메모리 사용량이 매우 높아질 수 있습니다.
* **초점 거리 보정**: 카메라의 초점 거리와 왜곡 계수를 정확히 조정해야 왜곡 없는 결과를 얻을 수 있습니다.
* **특징점 매칭 품질**: 이미지의 피처 검출기가 정확하지 않으면 결과가 크게 왜곡될 수 있습니다.

---

## 5. 결론

이 알고리즘은 기본적인 파노라마에서 복잡한 다중 시점 컴포지션까지 다양한 용도에 적합한 유연한 고품질 스티칭 파이프라인을 제공합니다.
