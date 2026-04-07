import io
import os
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline

# [INIT] FastAPI 앱 초기화
app = FastAPI(
    title="Emotion Classification API",
    description="Transformers(ViT) 기반의 얼굴 감정 분류 API 서버 및 프리미엄 웹 UI입니다.",
    version="2.0.0"
)

# [MODELS] 로딩 및 캐싱
# CPU 전용 PyTorch 환경에 최적화하여 로드합니다.
print("Loading Transformers model 'dima806/facial_emotions_image_detection'...")
classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

# [FACE DETECTION] OpenCV Haar Cascade 로드
# 가장 가볍고 CPU에서 빠른 얼굴 인식 방법입니다.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# [STATIC FILES MOUNT]
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    """메인 웹 UI 페이지를 반환합니다."""
    return FileResponse("static/index.html")

@app.get("/api/health")
def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "ok", "message": "Emotion Classification API (Transformers) is up and running!"}

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다. 이미지 파일을 제공해주세요.")
    
    try:
        # 1. 이미지 읽기 및 OpenCV 포맷 변환
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("이미지를 디코딩할 수 없습니다.")
        
        # 2. 얼굴 인식 (Face Detection) - 더 높은 정확도를 위해 얼굴 영역만 크롭
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # 이미지 전체를 기본으로 사용하되, 얼굴이 감지되면 가장 큰 얼굴 영역을 사용합니다.
        processed_img = img
        if len(faces) > 0:
            # 가장 영역이 넓은 얼굴 선택
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            processed_img = img[y:y+h, x:x+w]
        
        # 3. OpenCV(BGR) -> PIL(RGB) 변환 (Transformers 모델 필요 형식)
        rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        # 4. Transformers 모델 추론
        # 결과값: [{'label': 'happy', 'score': 0.9}, ...]
        results = classifier(pil_img)
        
        # 5. 기존 DeepFace 출력 포맷에 맞춰 데이터 가공
        emotion_probabilities = {res['label']: float(res['score'] * 100) for res in results}
        dominant_emotion = results[0]['label'] if results else "unknown"
        
        # 라벨 소문자 정규화 (UI 매칭용)
        dominant_emotion = dominant_emotion.lower()
        
        return JSONResponse(content={
            "dominant_emotion": dominant_emotion,
            "emotion_probabilities": emotion_probabilities
        })
            
    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류 발생: {str(e)}")
