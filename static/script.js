document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const cancelBtn = document.getElementById('cancel-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultSection = document.getElementById('result-section');
    const dominantBadge = document.getElementById('dominant-badge');
    const emotionBars = document.getElementById('emotion-bars');
    const loader = document.getElementById('loader');

    let selectedFile = null;

    // --- Drag & Drop ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault(); e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('active'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('active'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        handleFileSelect(file);
    });

    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        handleFileSelect(e.target.files[0]);
    });

    function handleFileSelect(file) {
        if (!file || !file.type.startsWith('image/')) {
            alert('이미지 파일(JPG, PNG 등)만 업로드 가능합니다.');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();

        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            analyzeBtn.classList.remove('disabled');
            analyzeBtn.disabled = false;
        };

        reader.readAsDataURL(file);
    }

    cancelBtn.addEventListener('click', () => {
        selectedFile = null;
        imagePreview.src = '';
        dropZone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        analyzeBtn.classList.add('disabled');
        analyzeBtn.disabled = true;
        resultSection.classList.add('hidden');
    });

    // --- API Call ---
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        loader.classList.remove('hidden');
        resultSection.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('분석 중 오류 발생');

            const data = await response.json();
            renderResults(data);
        } catch (error) {
            console.error(error);
            alert('서버 응답 오류가 발생했습니다. 다시 시도해 주세요.');
        } finally {
            loader.classList.add('hidden');
        }
    });

    function renderResults(data) {
        const dominant = data.dominant_emotion;
        const probs = data.emotion_probabilities;

        dominantBadge.textContent = dominant;
        dominantBadge.style.background = getEmotionColor(dominant);

        emotionBars.innerHTML = '';

        // Sort by probability for better display
        const sortedEmotions = Object.entries(probs).sort((a,b) => b[1] - a[1]);

        sortedEmotions.forEach(([emotion, value]) => {
            const barItem = document.createElement('div');
            barItem.className = 'emotion-bar-item';
            
            const roundedVal = (value).toFixed(1);
            
            barItem.innerHTML = `
                <div class="bar-label">
                    <span>${emotion.charAt(0).toUpperCase() + emotion.slice(1)}</span>
                    <span>${roundedVal}%</span>
                </div>
                <div class="bar-bg">
                    <div class="bar-fill" style="width: 0%; background: ${getEmotionColor(emotion)}"></div>
                </div>
            `;
            
            emotionBars.appendChild(barItem);

            // Animate bar filling
            setTimeout(() => {
                barItem.querySelector('.bar-fill').style.width = `${roundedVal}%`;
            }, 100);
        });

        resultSection.classList.remove('hidden');
        
        // Scroll to results
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    function getEmotionColor(emotion) {
        const colors = {
            'happy': '#ffdf33',
            'sad': '#4dabff',
            'angry': '#ff4d4d',
            'neutral': '#a5a5a5',
            'fear': '#7b68ee',
            'surprise': '#ff8c00',
            'disgust': '#32cd32'
        };
        return colors[emotion.toLowerCase()] || '#8a7cfd';
    }
});
