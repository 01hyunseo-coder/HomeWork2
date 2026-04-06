# MLOps Emotion Classification API

FastAPI와 DeepFace를 기반으로 한 얼굴 감정 분류(Facial Emotion Classification) 모델의 추론 API 입니다.

## 🚀 실행 가이드 (How to run)

1. **가상환경 생성 및 활성화 (선택 사항)**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **패키지 설치**
   ```bash
   pip install -r requirements.txt
   ```

3. **서버 실행**
   ```bash
   uvicorn main:app --reload --host 0.0.0.1 --port 8000
   ```

## 🧪 테스트 방법
서버가 실행된 후 웹 브라우저나 Postman 등을 사용하여 다음 URL로 접속할 수 있습니다:
- **Swagger UI (테스트 화면)**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/`
