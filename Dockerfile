# 1. Base Image - 모델 구동 가볍고 안정적인 파이썬 3.10 slim 버전 사용
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 환경 변수 설정 (파이썬 바이트코드 생성 방지, 로그 버퍼링 비활성화)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/cache

# 4. 필수 시스템 패키지 설치 및 캐시 정리 (최적화)
# MediaPipe는 이미지 처리를 위해 다수의 공유 라이브러리를 필요로 합니다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5. 의존성 패키지 캐싱 레이어 활용을 위해 requirements.txt 먼저 복사
COPY requirements.txt .

# 6. 패키지 설치 (CPU-only PyTorch 및 최적화)
# --index-url을 사용하여 CPU 전용 버전을 설치하여 이미지 용량과 빌드 시간을 획기적으로 줄입니다.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# 7. 모델 사전 다운로드 (빌드 속도 및 런타임 성능 최적화)
# 이미지 빌드 시 모델을 미리 캐싱하여 서비스 시작 속도를 높입니다.
# 소스 코드 변경 시에도 이 레이어는 캐시되어 유지됩니다.
RUN python -c "from transformers import pipeline; pipeline('image-classification', model='dima806/facial_emotions_image_detection')"

# 8. 앱 소스 코드 복사
COPY . .

# 9. 컨테이너 개방 포트 알림
EXPOSE 8000

# 10. 서버 실행 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
