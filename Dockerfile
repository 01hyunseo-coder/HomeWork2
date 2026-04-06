# 1. Base Image - 모델 구동 가볍고 안정적인 파이썬 3.10 slim 버전 사용
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 환경 변수 설정 (파이썬 바이트코드 생성 방지, 로그 버퍼링 비활성화)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 4. 필수 시스템 패키지 설치 및 캐시 정리 (최적화)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5. 의존성 패키지 캐싱 레이어 활용을 위해 requirements.txt 먼저 복사
COPY requirements.txt .

# 6. 패키지 설치 (설치 후 pip 캐시를 정리하여 용량 최적화)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# (선택 사항) 로컬 모델 다운로드를 사용할 수 있지만 CI 환경에서 불안정할 수 있으므로 런타임에 다운로드합니다.

# 7. 앱 소스 코드 복사
COPY . .

# 8. 컨테이너 개방 포트 알림
EXPOSE 8000

# 9. 서버 실행 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
