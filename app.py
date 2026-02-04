from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import cv2
from deepface import DeepFace
import io
import os

app = FastAPI(title="Face Search API")

def get_db_connection():
    db_host = os.getenv("DB_HOST", "localhost")
    db_name = os.getenv("DB_NAME", "face_db")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "password")
    
    return psycopg2.connect(
        f"dbname={db_name} user={db_user} password={db_password} host={db_host}",
        cursor_factory=RealDictCursor
    )

@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница с формой загрузки изображения"""
    html_content = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Поиск лиц</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                max-width: 600px;
                width: 100%;
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
                font-size: 2em;
            }
            .upload-area {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                background: #f8f9ff;
                margin-bottom: 20px;
                cursor: pointer;
                transition: all 0.3s;
            }
            .upload-area:hover {
                background: #eef0ff;
                border-color: #764ba2;
            }
            .upload-area.dragover {
                background: #e0e7ff;
                border-color: #4f46e5;
            }
            .upload-area.hidden {
                display: none;
            }
            .preview-container {
                position: relative;
                display: none;
                margin: 20px 0;
            }
            .preview-container.visible {
                display: block;
            }
            #preview {
                max-width: 100%;
                max-height: 400px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                display: block;
            }
            .remove-btn {
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(239, 68, 68, 0.9);
                color: white;
                border: none;
                border-radius: 50%;
                width: 36px;
                height: 36px;
                font-size: 20px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            }
            .remove-btn:hover {
                background: rgba(220, 38, 38, 1);
                transform: scale(1.1);
            }
            button {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            button:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
            }
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            #result {
                margin-top: 30px;
                padding: 20px;
                border-radius: 10px;
                display: none;
                animation: slideIn 0.5s;
            }
            #result.success {
                background: #d1fae5;
                border: 2px solid #10b981;
                color: #065f46;
            }
            #result.error {
                background: #fee2e2;
                border: 2px solid #ef4444;
                color: #991b1b;
            }
            .result-name {
                font-size: 1.5em;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .result-distance {
                font-size: 1.1em;
                opacity: 0.8;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
                display: none;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(-10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 Поиск лиц</h1>
            <div class="upload-area" id="uploadArea">
                <p style="font-size: 48px; margin-bottom: 10px;">📷</p>
                <p style="color: #667eea; font-size: 18px; font-weight: 600;">Выберите или перетащите изображение</p>
                <p style="color: #999; margin-top: 5px;">Поддерживаются: JPG, PNG</p>
                <input type="file" id="fileInput" accept="image/*" style="display: none;">
            </div>
            <div class="preview-container" id="previewContainer">
                <button class="remove-btn" id="removeBtn" onclick="removeImage()">✕</button>
                <img id="preview" alt="Preview">
            </div>
            <div class="spinner" id="spinner"></div>
            <button id="searchBtn" onclick="searchFace()" disabled>Найти лицо</button>
            <div id="result"></div>
        </div>

        <script>
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const preview = document.getElementById('preview');
            const previewContainer = document.getElementById('previewContainer');
            const searchBtn = document.getElementById('searchBtn');
            const result = document.getElementById('result');
            const spinner = document.getElementById('spinner');

            uploadArea.addEventListener('click', () => fileInput.click());

            fileInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) handleFile(file);
            });

            // Drag & Drop
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const file = e.dataTransfer.files[0];
                if (file) handleFile(file);
            });

            function handleFile(file) {
                if (!file.type.startsWith('image/')) {
                    alert('Пожалуйста, выберите изображение!');
                    return;
                }

                const reader = new FileReader();
                reader.onload = (e) => {
                    preview.src = e.target.result;
                    uploadArea.classList.add('hidden');
                    previewContainer.classList.add('visible');
                    searchBtn.disabled = false;
                    result.style.display = 'none';
                };
                reader.readAsDataURL(file);
            }

            function removeImage() {
                preview.src = '';
                fileInput.value = '';
                uploadArea.classList.remove('hidden');
                previewContainer.classList.remove('visible');
                searchBtn.disabled = true;
                result.style.display = 'none';
            }

            async function searchFace() {
                const file = fileInput.files[0];
                if (!file) return;

                const formData = new FormData();
                formData.append('file', file);

                searchBtn.disabled = true;
                spinner.style.display = 'block';
                result.style.display = 'none';

                try {
                    const response = await fetch('/search', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (response.ok) {
                        result.className = 'success';
                        result.innerHTML = `
                            <div class="result-name">✅ Найдено совпадение!</div>
                            <div class="result-name">${data.name}</div>
                            <div class="result-distance">Расстояние: ${data.distance.toFixed(4)}</div>
                        `;
                    } else {
                        result.className = 'error';
                        result.innerHTML = `
                            <div class="result-name">❌ Ошибка</div>
                            <div>${data.detail || 'Не удалось найти лицо'}</div>
                        `;
                    }
                } catch (error) {
                    result.className = 'error';
                    result.innerHTML = `
                        <div class="result-name">❌ Ошибка сети</div>
                        <div>${error.message}</div>
                    `;
                } finally {
                    spinner.style.display = 'none';
                    result.style.display = 'block';
                    searchBtn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/search")
async def search_face(file: UploadFile = File(...)):
    """Поиск лица по загруженному изображению"""
    
    if not file.content_type.startswith('image/'): #type: ignore
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")
        
        embedding = DeepFace.represent(img_path=img, model_name="Facenet")[0]["embedding"]  # type: ignore
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT name, (embedding <-> %s::vector) as distance FROM embeddings ORDER BY distance LIMIT 1",
            (embedding_str,)
        )
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if result:
            return JSONResponse({
                "name": result["name"], #type: ignore
                "distance": float(result["distance"]) #type: ignore
            })
        else:
            raise HTTPException(status_code=404, detail="Лица не найдены в базе данных")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Получить статистику базы данных"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as count FROM embeddings")
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return {"total_embeddings": result["count"]} #type: ignore  
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка БД: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
