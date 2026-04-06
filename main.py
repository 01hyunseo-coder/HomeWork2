import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from deepface import DeepFace

app = FastAPI(
    title="Emotion Classification API",
    description="MLOps 기반의 얼굴 감정 분류 API 서버입니다.",
    version="1.0.0"
)

@app.get("/")
def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "ok", "message": "Emotion Classification API is up and running!"}

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """
    이미지 파일을 업로드 받아 얼굴 감정을 분류합니다.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다. 이미지 파일을 제공해주세요.")
    
    try:
        # 이미지를 비동기적으로 읽어오기
        contents = await file.read()
        
        # Bytes를 NumPy 데이터로 치환 후 OpenCV 이미지로 디코드
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("이미지를 디코딩할 수 없습니다.")
        
        # DeepFace를 사용한 모델 추론 (얼굴 감지 실패 시 에러 우회를 위해 enforce_detection=False 설정)
        result = DeepFace.analyze(
            img_path=img, 
            actions=['emotion'], 
            enforce_detection=False
        )
        
        # DeepFace 버전/복수 얼굴 감지에 따른 반환 형식 통일
        if isinstance(result, list):
            result = result[0]
            
        emotion_probs = result.get('emotion', {})
        # numpy float32 타입을 JSON 직렬화가 가능한 파이썬 기본 float 타입으로 변환합니다.
        emotion_probs = {k: float(v) for k, v in emotion_probs.items()}
            
        return JSONResponse(content={
            "dominant_emotion": result.get('dominant_emotion'),
            "emotion_probabilities": emotion_probs
        })
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류 발생: {str(e)}")
