# Python 3.12 가상 환경 (venv) 설정 가이드

이 가이드는 Python 3.12 가상 환경(venv) 생성, 활성화 및 패키지 설치 과정을 포함합니다. Windows와 Unix 계열 시스템 모두에 적용할 수 있습니다.

---

## **1. 사전 준비**

* **Python 3.12**가 시스템에 설치되어 있어야 합니다.
* 설치 여부를 다음 명령어로 확인할 수 있습니다.

```bash
python3.12 --version
```

Python 3.12이 설치되어 있지 않다면 [공식 Python 웹사이트](https://www.python.org/downloads/release/python-3120/)에서 다운로드하세요.

---

## **2. 가상 환경 생성**

* `venv`라는 가상 환경을 생성하려면 다음 명령어를 실행하세요.

### **Windows (CMD/PowerShell)**

```bash
py -3.12 -m venv venv
```

### **Linux/Mac**

```bash
python3.12 -m venv venv
```

이 명령어는 다음과 같은 디렉토리 구조를 생성합니다:

* `bin/` 또는 `Scripts/` - 실행 파일 (Python, pip, activate 스크립트 포함)
* `lib/` - site-packages 디렉토리
* `pyvenv.cfg` - 가상 환경 설정 파일

---

## **3. 가상 환경 활성화**

### **PowerShell에서 실행 정책 설정 (필수)**

PowerShell에서 스크립트를 실행하려면 **실행 정책**을 조정해야 합니다. 다음 명령어로 설정을 변경하세요.

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

* 이 명령어를 실행한 후 **Y**를 입력하여 변경을 승인합니다.
* 이후에도 보안 경고가 표시될 수 있으니 신뢰할 수 있는 스크립트만 실행하세요.

### **Windows (CMD)**

```bash
venv\Scripts\activate.bat
```

### **Windows (PowerShell)**

```powershell
venv\Scripts\Activate.ps1
```

### **Linux/Mac**

```bash
source venv/bin/activate
```

가상 환경이 활성화되면 `(venv)`라는 프롬프트가 표시됩니다.

---

## **4. 가상 환경 확인**

현재 활성화된 Python 버전을 확인하려면 다음 명령어를 사용하세요.

```bash
python --version
```

기대 출력:

```
Python 3.12.x
```

---

## **5. 패키지 설치**

필요한 패키지를 설치하려면 다음 명령어를 사용하세요.

```bash
python -m pip install --upgrade pip setuptools
pip install opencv-python numpy matplotlib ipykernel
```

* **`pip`** - Python 패키지 관리자
* **`setuptools`** - 패키지 빌드 및 배포 도구

또는 `requirements.txt` 파일에서 패키지를 일괄 설치할 수 있습니다.

```bash
pip install -r requirements.txt
```

---

## **6. 가상 환경 비활성화**

가상 환경을 종료하려면 다음 명령어를 입력하세요.

```bash
deactivate
```

이 명령어를 입력하면 기본 Python 환경으로 돌아갑니다.

---

## **7. 추가 팁**

* 설치된 패키지 목록을 확인하려면:

```bash
pip freeze > requirements.txt
```

* 더 이상 필요하지 않은 가상 환경을 삭제하려면:

```bash
rm -rf venv  # Linux/Mac
rmdir /s /q venv  # Windows
```

* PowerShell 실행 정책이 초기화된 경우 다시 설정이 필요합니다.

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

더 복잡한 환경 관리가 필요하면 `virtualenv`나 `pyenv`와 같은 도구도 고려해 보세요.
