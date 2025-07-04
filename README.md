# VitPose Video Analysis 🏃‍♂️

비디오에서 인간의 포즈를 분석하고 키포인트를 추출하는 Gradio 기반 웹 애플리케이션입니다. VitPose 모델을 사용하여 133개 키포인트(Whole Body)를 실시간으로 추출하고 시각화합니다.

## 주요 기능 ✨

- 📹 **비디오 업로드 및 포즈 분석**: MP4 비디오 파일에서 인간의 포즈를 자동으로 검출
- 🎯 **실시간 키포인트 시각화**: 133개 키포인트와 스켈레톤을 실시간으로 표시
- ⏰ **시간 연동 탐색**: 시간 슬라이더와 재생 기능으로 원하는 시점의 키포인트를 실시간으로 확인
- 💾 **데이터 출력**: JSON과 CSV 형식으로 포즈 데이터를 내보내기
- 📊 **실시간 프로그레스**: 비디오 처리 진행상황을 프로그레스 바로 실시간 확인
- 🔍 **JSON 데이터 뷰어**: 포즈 데이터를 실시간으로 탐색하고 확장/축소 가능
- 📋 **로그 시스템**: 처리 과정을 실시간으로 추적하고 관리
- 🚀 **성능 최적화**: 사람 검출 최적화로 처리 속도 향상
- 🤖 **모델 선택**: Huge (정확) / Base (빠름) 모델 선택 가능
- 🔍 **이미지 줌 기능**: 키포인트 이미지 확대/축소/원본 복원
- 📊 **비디오 정보 표시**: 해상도, 길이, FPS, 코덱 등 비디오 정보 자동 표시

## 시스템 요구사항 🔧

- **운영체제**: macOS, Linux, Windows
- **Python**: 3.9 이상
- **GPU**: CUDA 지원 GPU 권장 (CPU도 지원)
- **메모리**: 최소 8GB RAM 권장

## 환경 구성 🚀

### 1. 저장소 클론

```bash
git clone <repository-url>
cd vitpose_web
```

### 2. 가상환경 생성

```bash
# Conda 사용
conda create -n vitpose python=3.9 -y
conda activate vitpose

# 또는 venv 사용
python -m venv vitpose_env
source vitpose_env/bin/activate  # Linux/Mac
# vitpose_env\Scripts\activate  # Windows
```

### 3. 패키지 설치

```bash
# 기본 패키지 업그레이드
pip install --upgrade pip setuptools wheel

# requirements.txt로 일괄 설치
pip install -r requirements.txt
```

## 설치 확인 ✅

VitPose 모델이 정상적으로 import되는지 확인:

```bash
python -c "from transformers import VitPoseForPoseEstimation; print('VitPose 설치 완료!')"
```

## 사용법 📖

### 1. 애플리케이션 실행

```bash
# 가상환경 활성화
conda activate vitpose  # 또는 source vitpose_env/bin/activate

# 애플리케이션 시작
python gradio_app.py
```

### 2. 웹 브라우저에서 접속

```
로컬 URL: http://127.0.0.1:7860
공개 URL: 콘솔에 표시되는 gradio 링크 사용
```

### 3. 비디오 분석 과정

1. **비디오 업로드**: 왼쪽 패널에서 MP4 비디오 파일을 업로드
2. **모델 선택**: Huge (정확) 또는 Base (빠름) 모델 선택
3. **프레임 간격 설정**: 처리할 프레임 간격 선택 (기본: 5프레임마다)
4. **포즈 추출 시작**: "🚀 포즈 추출 시작" 버튼 클릭
5. **진행상황 확인**: 프로그레스 바로 처리 진행률 실시간 확인
6. **키포인트 탐색**: 시간 슬라이더와 재생 버튼으로 원하는 시점의 키포인트 확인
7. **이미지 줌**: 확대/축소/원본 버튼으로 키포인트 상세 확인
8. **데이터 탐색**: JSON 뷰어에서 포즈 데이터 실시간 탐색
9. **데이터 다운로드**: JSON/CSV 파일로 포즈 데이터 다운로드

## 파일 구조 📁

```
vitpose_web/
├── README.md                 # 프로젝트 문서
├── requirements.txt          # 패키지 의존성
├── gradio_app.py            # Gradio 웹 인터페이스 (메인 앱)
├── vitpose_gradio_app.py    # 포즈 분석 핵심 로직
└── output/                  # 생성된 결과 파일들
    ├── *.json              # 포즈 데이터 (JSON)
    └── *.csv               # 포즈 데이터 (CSV)
```

## 주요 기능 상세 🔍

### 키포인트 시각화
- **133개 키포인트**: COCO-WholeBody 표준 기반
  - 🦴 **Body (17개)**: 코, 눈, 귀, 어깨, 팔꿈치, 손목, 엉덩이, 무릎, 발목
  - 👤 **Face (68개)**: 얼굴 윤곽선, 눈썹, 눈, 코, 입
  - 🤚 **Left Hand (21개)**: 왼손 관절들
  - 🤚 **Right Hand (21개)**: 오른손 관절들  
  - 👣 **Left Foot (6개)**: 왼발 관절들 (발끝 포함)
  - 👣 **Right Foot (6개)**: 오른발 관절들 (발끝 포함)
- **색상 구분**: 
  - 🔴 빨간색: 사람 검출 박스
  - 🔵 파란색: 모델 입력 박스
  - 🟢 녹색: 키포인트
  - 🔴 빨간색: 키포인트 연결선
- **신뢰도 기반 필터링**: 0.3 이상의 신뢰도를 가진 키포인트만 표시

### 재생 및 탐색 기능
- **재생/일시정지**: 자동 재생으로 키포인트 변화 확인
- **시간 슬라이더**: 원하는 시점으로 즉시 이동
- **처음으로**: 비디오 시작 지점으로 이동
- **이미지 줌**: 확대/축소/원본 복원으로 상세 확인

### 데이터 출력 형식

#### JSON 형식
```json
{
  "timestamp": 1.234,
  "frame_idx": 74,
  "persons": [
    {
      "person_id": 0,
      "keypoints": [
        {
          "id": 0,
          "name": "nose",
          "x": 320.5,
          "y": 240.3,
          "confidence": 0.95
        }
      ]
    }
  ]
}
```

#### CSV 형식
| timestamp | frame_idx | person_id | keypoint_id | keypoint_name | x | y | confidence |
|-----------|-----------|-----------|-------------|---------------|---|---|------------|
| 1.234 | 74 | 0 | 0 | nose | 320.5 | 240.3 | 0.95 |

### 성능 최적화 기능
- **사람 검출 최적화**: 30프레임마다 사람 검출하여 성능 향상
- **박스 크기 조정**: 이전 프레임 대비 박스 크기 최적화
- **프레임 간격 조정**: 사용자가 선택 가능한 프레임 처리 간격
- **모델 선택**: Huge (정확) / Base (빠름) 모델 선택

## 기술 스택 💻

- **딥러닝**: HuggingFace Transformers, PyTorch
- **포즈 추정 모델**: 
  - Huge: `yonigozlan/synthpose-vitpose-huge-hf`
  - Base: `yonigozlan/synthpose-vitpose-base-hf`
- **컴퓨터 비전**: OpenCV, PIL, NumPy
- **웹 인터페이스**: Gradio 5.x
- **데이터 처리**: Pandas, JSON
- **사람 검출**: RTDetr (PekingU/rtdetr_r50vd_coco_o365)

## 문제 해결 🔧

### 공통 문제

**1. VitPoseForPoseEstimation import 오류**
```bash
# Python 버전 확인 (3.9 이상 필요)
python --version

# transformers 버전 확인
pip show transformers
```

**2. CUDA 메모리 부족**
```bash
# CPU 모드로 실행하려면 vitpose_gradio_app.py에서
device = "cpu"로 설정
```

**3. Gradio 호환성 문제**
```bash
# 최신 버전으로 업그레이드
pip install --upgrade gradio gradio-client
```

### 성능 최적화 팁

- **GPU 사용**: CUDA 지원 GPU 사용 시 처리 속도 대폭 향상
- **프레임 간격 조정**: `frame_interval` 값을 늘려 처리 속도 향상
- **Base 모델 사용**: 빠른 처리가 필요한 경우 Base 모델 선택
- **해상도 조정**: 입력 비디오 해상도를 낮춰 메모리 사용량 감소

## 라이선스 📄

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여 🤝

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다!

## 업데이트 로그 📝

### 최신 업데이트 (2024)
- ✅ Gradio 5.x 업그레이드
- ✅ 실시간 프로그레스 바 추가
- ✅ 이미지 줌 기능 추가
- ✅ 비디오 정보 표시 기능 추가
- ✅ 자동 재생 기능 개선
- ✅ 키포인트 정보 섹션 제거로 UI 간소화
- ✅ 기본 모델을 Base로 변경 (빠른 처리)
- ✅ 사람 검출 최적화로 성능 향상
- ✅ 발끝 키포인트 지원 추가 