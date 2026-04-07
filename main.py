import io
import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline

# [INIT] FastAPI 앱 초기화
app = FastAPI(
    title="Emotion Classification API",
    description="Transformers(ViT) + MediaPipe 기반의 정밀 얼굴 감정 분류 API입니다.",
    version="3.0.0"
)

# [MODELS] 로딩 및 캐싱
print("Loading Transformers model 'dima806/facial_emotions_image_detection'...")
classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

# [FACE DETECTION] MediaPipe Face Detection 초기화
# 고글, 모자, 마스크 등에도 강인한 최신 딥러닝 기반 얼굴 인식기입니다.
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

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
    return {"status": "ok", "message": "Emotion Classification API (MediaPipe) is up and running!"}

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
        
        # 2. MediaPipe를 통한 얼굴 인식
        # 이미지를 RGB로 변환하여 MediaPipe에 전달합니다.
        results_mp = face_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        processed_img = img
        if results_mp.detections:
            # 가장 확신도가 높은 첫 번째 얼굴 선택
            detection = results_mp.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # 정규화된 좌표를 픽셀 좌표로 변환
            ih, iw, _ = img.shape
            x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
            
            # [PADDING] 얼굴 주변에 25% 여유를 주어 모델의 정확도를 높입니다.
            padding_w = int(w * 0.25)
            padding_h = int(h * 0.25)
            
            x1 = max(0, x - padding_w)
            y1 = max(0, y - padding_h)
            x2 = min(iw, x + w + padding_w)
            y2 = min(ih, y + h + padding_h)
            
            processed_img = img[y1:y2, x1:x2]
        
        # 3. OpenCV(BGR) -> PIL(RGB) 변환
        rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        # 4. Transformers 모델 추론
        results = classifier(pil_img)
        
        # 5. 기존 출력 포맷 매칭
        emotion_probabilities = {res['label']: float(res['score'] * 100) for res in results}
        dominant_emotion = results[0]['label'] if results else "unknown"
        dominant_emotion = dominant_emotion.lower()
        
        return JSONResponse(content={
            "dominant_emotion": dominant_emotion,
            "emotion_probabilities": emotion_probabilities
        })
            
    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류 발생: {str(e)}")
