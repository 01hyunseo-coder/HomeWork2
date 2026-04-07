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
    description="Transformers(ViT) + OpenCV DNN 기반의 안정적인 얼굴 감정 분류 API입니다.",
    version="4.0.0"
)

# [MODELS] Transformers 모델 로딩
print("Loading Transformers model 'dima806/facial_emotions_image_detection'...")
classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

# [FACE DETECTION] OpenCV DNN Face Detector 초기화
# 하르 캐스케이드보다 정확하고, 미디이파이보다 환경 의존성이 낮은 매우 안정적인 모델입니다.
PROTOTXT_PATH = "models/deploy.prototxt"
MODEL_PATH = "models/res10_300x300_ssd_iter_140000.caffemodel"

# 로컬 테스트 시 models 폴더가 없을 경우를 대비한 경로 체크
if os.path.exists(PROTOTXT_PATH) and os.path.exists(MODEL_PATH):
    face_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    print("OpenCV DNN Face Detector loaded successfully.")
else:
    face_net = None
    print("WARNING: Face detection model files not found. Inference might be less accurate.")

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
    return {"status": "ok", "message": "Emotion Classification API (OpenCV DNN) is up and running!"}

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
        
        h, w = img.shape[:2]
        processed_img = img

        # 2. OpenCV DNN을 통한 얼굴 인식
        if face_net is not None:
            # DNN 입력을 위해 300x300으로 리사이즈 및 Mean Subtraction 수행
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_net.setInput(blob)
            detections = face_net.forward()

            # 가장 확신도가 높은 얼굴 하나를 찾습니다.
            best_face = None
            max_confidence = 0.5 # 최소 확신도 임계값

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > max_confidence:
                    max_confidence = confidence
                    # 상대 좌표를 절대 픽셀 좌표로 변환
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    best_face = box.astype("int")

            if best_face is not None:
                (x1, y1, x2, y2) = best_face
                fw, fh = x2 - x1, y2 - y1

                # [PADDING] 얼굴 주변에 25% 여유를 주어 모델의 정확도를 높입니다.
                padding_w = int(fw * 0.25)
                padding_h = int(fh * 0.25)
                
                px1 = max(0, x1 - padding_w)
                py1 = max(0, y1 - padding_h)
                px2 = min(w, x2 + padding_w)
                py2 = min(h, y2 + padding_h)
                
                processed_img = img[py1:py2, px1:px2]
        
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
