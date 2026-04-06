import io
import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from deepface import DeepFace

app = FastAPI(
    title="Emotion Classification API",
    description="MLOps 기반의 얼굴 감정 분류 API 서버 및 프리미엄 웹 UI입니다.",
    version="1.1.0"
)

# [STATIC FILES MOUNT] - CSS, JS, Images 등의 정적 파일을 서비스합니다.
# static 폴더가 없으면 에러가 날 수 있으므로 체크 후 마운트합니다.
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
    return {"status": "ok", "message": "Emotion Classification API is up and running!"}

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    # ... (기존 로직 유지)
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다. 이미지 파일을 제공해주세요.")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("이미지를 디코딩할 수 없습니다.")
        
        result = DeepFace.analyze(
            img_path=img, 
            actions=['emotion'], 
            enforce_detection=False
        )
        
        if isinstance(result, list):
            result = result[0]
            
        emotion_probs = result.get('emotion', {})
        # JSON 직렬화를 위해 float 변환
        emotion_probs = {k: float(v) for k, v in emotion_probs.items()}
            
        return JSONResponse(content={
            "dominant_emotion": result.get('dominant_emotion'),
            "emotion_probabilities": emotion_probs
        })
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류 발생: {str(e)}")

