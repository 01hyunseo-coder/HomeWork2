# 1. Base Image - 모델 구동 가볍고 안정적인 파이썬 3.10 slim 버전 사용
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 환경 변수 설정 (파이썬 바이트코드 생성 방지, 로그 버퍼링 비활성화)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/cache

# 4. 필수 시스템 패키지 설치 및 캐시 정리 (최적화)
# OpenCV DNN 구동을 위한 필수 라이브러리와 모델 다운로드를 위한 curl을 설치합니다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5. OpenCV DNN 얼굴 인식 모델 다운로드
# 가볍고 정확한 얼굴 인식을 위해 OpenCV의 DNN 모델 파일을 미리 다운로드합니다.
RUN mkdir -p models && \
    curl -L https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt -o models/deploy.prototxt && \
    curl -L https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel -o models/res10_300x300_ssd_iter_140000.caffemodel

# 6. 의존성 패키지 캐싱 레이어 활용을 위해 requirements.txt 먼저 복사
COPY requirements.txt .

# 7. 패키지 설치 (CPU-only PyTorch 및 최적화)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# 8. 모델 사전 다운로드 (빌드 속도 및 런타임 성능 최적화)
RUN python -c "from transformers import pipeline; pipeline('image-classification', model='dima806/facial_emotions_image_detection')"

# 9. 앱 소스 코드 복사
COPY . .

# 10. 컨테이너 개방 포트 알림
EXPOSE 8000

# 11. 서버 실행 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
