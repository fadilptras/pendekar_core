from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64

# Mengimpor mesin PENDekar yang sudah kamu buat sebelumnya
from card_det import PendekarPipeline 

app = FastAPI(title="PENDekar API", description="API untuk Restorasi Dokumen Forensik")

# Mengizinkan akses dari aplikasi mobile (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi pipeline
pipeline = PendekarPipeline()

@app.get("/")
def home():
    return {"message": "Server PENDekar berjalan normal. Siap menerima gambar!"}

@app.post("/api/scan")
async def scan_document(
    file: UploadFile = File(...), 
    folder_name: str = Form("Tanpa_Nama")
):
    try:
        print(f"Menerima kasus: {folder_name}")
        
        # 1. Membaca file gambar yang dikirim dari React Native
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "error", "message": "File bukan gambar yang valid."}

        # 2. Memproses gambar menggunakan PENDekar
        print("Memulai proses pemindaian...")
        sukses, hasil_final, pesan_log = pipeline.process_image(img)

        # 3. Mengembalikan hasil ke HP
        if sukses:
            # Mengubah gambar hasil menjadi teks Base64 agar mudah dikirim lewat internet
            _, buffer = cv2.imencode('.jpg', hasil_final)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "status": "success", 
                "message": pesan_log, 
                # Format ini langsung bisa dibaca oleh tag <Image> di React Native
                "image_data": f"data:image/jpeg;base64,{img_base64}" 
            }
        else:
            return {"status": "failed", "message": pesan_log}

    except Exception as e:
        return {"status": "error", "message": str(e)}