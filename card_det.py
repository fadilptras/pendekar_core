import cv2
import numpy as np
from typing import Tuple, Optional

class PendekarPipeline:
    def __init__(self, output_width: int = 856, output_height: int = 540):
        """Resolusi standar output PENDekar."""
        self.width = output_width
        self.height = output_height

    # FUNGSI HELPER
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    # TAHAP 1: SEGMENTASI PRO (Cell 1 & 2)
    def find_card_corners_pro(self, image: np.ndarray):
        total_area = image.shape[0] * image.shape[1]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Strategi 1: Canny Standar
        blur1 = cv2.GaussianBlur(gray, (5, 5), 0)
        edge_std = cv2.Canny(blur1, 70, 200)
        
        # Strategi 2: CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray_clahe = clahe.apply(gray)
        blur2 = cv2.GaussianBlur(gray_clahe, (5, 5), 0)
        edge_clahe = cv2.Canny(blur2, 40, 150)
        
        # Strategi 3: Morphological Gradient
        kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gradient = cv2.morphologyEx(blur1, cv2.MORPH_GRADIENT, kernel_grad)
        _, edge_grad = cv2.threshold(gradient, 50, 255, cv2.THRESH_BINARY)
        
        strategies = [
            ("Standar Canny", edge_std),
            ("CLAHE + Canny", edge_clahe),
            ("Morphological Gradient", edge_grad)
        ]
        
        best_contour = None
        used_strategy = ""
        final_edge_map = None
        
        for name, edged_img in strategies:
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(edged_img, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            
            contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            
            for c in contours:
                area = cv2.contourArea(c)
                if area > (0.10 * total_area):
                    perimeter = cv2.arcLength(c, True)
                    for eps in [0.02, 0.04, 0.06, 0.08, 0.1]:
                        approx = cv2.approxPolyDP(c, eps * perimeter, True)
                        if len(approx) == 4:
                            best_contour = approx
                            break 
                if best_contour is not None:
                    break 
                    
            if best_contour is not None:
                used_strategy = name
                final_edge_map = closed
                break 
                
        if best_contour is None:
            return None, edge_std, "Semua Strategi Gagal"
            
        pts = best_contour.reshape(4, 2)
        return self._order_points(pts), final_edge_map, used_strategy

    # TAHAP 2: GEOMETRI (Cell 3 & 4)
    def apply_geometry_warp(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        pts1 = np.float32(corners)
        pts2 = np.float32([[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(image, matrix, (self.width, self.height))

    # TAHAP 3: MEDIAN FILTER (Cell 5 & 6)
    def apply_median_filter(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.medianBlur(image, kernel_size)

    # TAHAP 4: SEGMENTASI BINER (Cell 7)
    def apply_binarization(self, image: np.ndarray, block_size: int = 15, C: int = 4) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary_image = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, C
        )
        return binary_image

    # TAHAP 5: PEMBERSIHAN TEKS (Cell 8 & 9)
    def enhance_binary_text(self, binary_image: np.ndarray) -> np.ndarray:
        cleaned = cv2.medianBlur(binary_image, 3)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.erode(cleaned, kernel_erode, iterations=1)

    # EKSEKUTOR UTAMA (RUNNER)
    def process_image(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], str]:
        """Menjalankan seluruh pipeline PENDekar secara berurutan."""
        try:
            # 0. Standarisasi Ukuran Awal
            target_width = 800
            ratio = target_width / image.shape[1]
            dim = (target_width, int(image.shape[0] * ratio))
            resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            # Eksekusi TAHAP 1
            corners, _, strategy = self.find_card_corners_pro(resized)
            if corners is None:
                return False, None, "Kartu tidak terdeteksi. Evaluasi latar belakang."

            # Eksekusi TAHAP 2
            warped = self.apply_geometry_warp(resized, corners)

            # Eksekusi TAHAP 3
            filtered = self.apply_median_filter(warped, kernel_size=5)

            # Eksekusi TAHAP 4
            binary = self.apply_binarization(filtered, block_size=15, C=4)

            # Eksekusi TAHAP 5
            final_output = self.enhance_binary_text(binary)

            return True, final_output, f"Sukses direstorasi ({strategy})"

        except Exception as e:
            return False, None, f"Gagal sistem: {str(e)}"

# ==========================================
# BLOK EKSEKUSI (TRIGGER)
# ==========================================
# if __name__ == "__main__":
#     # 1. Panggil / Nyalakan Mesin PENDekar
#     print("Minyiapkan mesin PENDekar Pro...")
#     pipeline = PendekarPipeline()

#     # 2. Tentukan gambar yang mau diuji 
#     # (Pastikan file 'test_sim.jpg' ada di folder yang sama dengan 'card_det.py')
#     nama_file = 'test_sim.jpeg'
#     gambar_uji = cv2.imread(nama_file)

#     if gambar_uji is None:
#         print(f"ERROR: File gambar '{nama_file}' tidak ditemukan di folder ini.")
#     else:
#         print(f"Mulai memproses '{nama_file}'...")
        
#         # 3. Eksekusi proses_image
#         sukses, hasil_final, pesan_log = pipeline.process_image(gambar_uji)

#         # 4. Cetak log ke terminal
#         print(f"LOG SISTEM: {pesan_log}")

#         # 5. Jika sukses, simpan gambar hasilnya agar bisa kamu lihat
#         if sukses:
#             nama_output = 'hasil_uji_terminal.png'
#             cv2.imwrite(nama_output, hasil_final)
#             print(f"BERHASIL! Gambar final telah disimpan sebagai '{nama_output}'")