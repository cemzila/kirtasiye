import sys
import os
import pickle
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as T
from ultralytics import YOLO

from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QGroupBox,
    QMessageBox, QFileDialog
)

# VERİTABANI İŞLEMLERİ

DB_FILE = "reid_db.pkl"
SIMILARITY_THRESHOLD = 0.78

def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Veritabanı yükleme hatası: {e}")
            return []
    return []

def save_db(db):
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)

def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2)


#MODEL YÜKLEME - TAKİP 

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    active_tracks_signal = pyqtSignal(dict)
    detection_log_signal = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._run_flag = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # MobileNetV3 Re-ID Modeli
        try:
            self.embedder = models.mobilenet_v3_small(weights=models.MobileNetV3_Small_Weights.DEFAULT)
        except AttributeError:
            self.embedder = models.mobilenet_v3_small(pretrained=True)
        self.embedder.classifier = torch.nn.Identity()
        self.embedder.eval().to(self.device)

        
        self.yolo_model = YOLO("runs/detect/train/weights/best.pt")

        # Re-ID Ön İşleme
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.db = load_db()
        self.custom_names = {}  
        self.active_crops = {}  

    def get_embedding(self, crop):
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.embedder(tensor).cpu().numpy().flatten()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else None

    def update_db(self, new_db):
        self.db = new_db

    def assign_custom_name(self, track_id, name, crop):
        emb = self.get_embedding(crop)
        if emb is not None:
            self.db.append({"name": name, "embedding": emb})
            save_db(self.db)
            self.custom_names[track_id] = name

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                continue

            results = self.yolo_model.track(
                frame, persist=True, tracker="bytetrack.yaml", conf=0.35, verbose=False
            )

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy().astype(int)

                if results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = np.arange(len(boxes))

                h, w, _ = frame.shape

                for box, track_id, cls in zip(boxes, track_ids, clss):
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                    crop = frame[y1:y2, x1:x2]

                    if crop.size > 0 and crop.shape[0] >= 10 and crop.shape[1] >= 10:
                        self.active_crops[track_id] = crop

                        # Re-ID Eşleştirme Kontrolü
                        if track_id not in self.custom_names:
                            emb = self.get_embedding(crop)
                            if emb is not None and len(self.db) > 0:
                                best_match = None
                                max_sim = 0.0
                                for item in self.db:
                                    sim = cosine_similarity(emb, item["embedding"])
                                    if sim > max_sim:
                                        max_sim = sim
                                        best_match = item["name"]

                                if max_sim >= SIMILARITY_THRESHOLD:
                                    self.custom_names[track_id] = best_match

                        cls_name = self.yolo_model.names[cls]
                        assigned_name = self.custom_names.get(track_id, None)
                        display_label = f"{assigned_name} (ID:{track_id})" if assigned_name else f"{cls_name} (ID:{track_id})"

                        # Log Yayınlama
                        log_data = {
                            "Zaman": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Track_ID": track_id,
                            "Sınıf": cls_name,
                            "Atanan_İsim": assigned_name if assigned_name else "Bilinmiyor"
                        }
                        self.detection_log_signal.emit(log_data)

                        # Bounding Box ve Çizim
                        color = (0, 255, 0) if assigned_name else (255, 165, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.rectangle(frame, (x1, y1 - 25), (x1 + len(display_label) * 11, y1), color, -1)
                        cv2.putText(
                            frame, display_label, (x1 + 5, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                        )

            self.active_tracks_signal.emit(self.active_crops)
            self.change_pixmap_signal.emit(frame)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


# ANA PENCERE

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(" Kırtasiye Ürünleri Takip Sistemi")
        self.resize(1280, 720)

        self.detection_logs = []
        self.active_crops = {}

        self.init_ui()

        # Thread Başlatma
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.active_tracks_signal.connect(self.update_active_tracks)
        self.thread.detection_log_signal.connect(self.add_log)
        self.thread.start()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # Sol Panel (Kamera Akışı)
        left_layout = QVBoxLayout()
        self.video_label = QLabel("Kamera Yükleniyor...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #1e1e1e; color: #ffffff; border-radius: 8px;")
        self.video_label.setMinimumSize(720, 540)
        left_layout.addWidget(self.video_label)

        # Sağ Panel (Kontroller ve Etiketleme)
        right_layout = QVBoxLayout()

        #İsimlendirme Paneli
        naming_group = QGroupBox("Nesne İsimlendirme Paneli")
        naming_layout = QVBoxLayout()

        naming_layout.addWidget(QLabel("Etiketlemek İstediğiniz Track ID:"))
        self.combo_track_id = QComboBox()
        self.combo_track_id.currentIndexChanged.connect(self.on_track_selected)
        naming_layout.addWidget(self.combo_track_id)

        self.crop_preview_label = QLabel("Önizleme")
        self.crop_preview_label.setFixedSize(120, 120)
        self.crop_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.crop_preview_label.setStyleSheet("border: 1px solid #444; background-color: #2b2b2b;")
        naming_layout.addWidget(self.crop_preview_label, alignment=Qt.AlignmentFlag.AlignCenter)

        naming_layout.addWidget(QLabel("Atanacak İsim / Etiket:"))
        self.input_name = QLineEdit()
        self.input_name.setPlaceholderText("Örn: Kırmızı Kalem")
        naming_layout.addWidget(self.input_name)

        self.btn_save = QPushButton("İsmi Kaydet")
        self.btn_save.setStyleSheet("background-color: #2b8a3e; color: white; padding: 8px; font-weight: bold;")
        self.btn_save.clicked.connect(self.save_name)
        naming_layout.addWidget(self.btn_save)

        naming_group.setLayout(naming_layout)
        right_layout.addWidget(naming_group)

        #Veritabanı ve Dışa Aktarım
        db_group = QGroupBox("Veritabanı")
        db_layout = QVBoxLayout()

        self.btn_export = QPushButton("Tespit Geçmişini İndir (CSV)")
        self.btn_export.clicked.connect(self.export_csv)
        db_layout.addWidget(self.btn_export)

        self.btn_clear_db = QPushButton("Veritabanını Temizle")
        self.btn_clear_db.setStyleSheet("background-color: #c92a2a; color: white; padding: 6px;")
        self.btn_clear_db.clicked.connect(self.clear_db)
        db_layout.addWidget(self.btn_clear_db)

        db_group.setLayout(db_layout)
        right_layout.addWidget(db_group)

        #Kayıtlı Re-ID Nesneleri Listesi
        saved_group = QGroupBox("Kayıtlı Nesneler")
        saved_layout = QVBoxLayout()
        self.saved_list_label = QLabel("Veritabanı kontrol ediliyor...")
        self.saved_list_label.setWordWrap(True)
        saved_layout.addWidget(self.saved_list_label)
        saved_group.setLayout(saved_layout)
        right_layout.addWidget(saved_group)

        right_layout.addStretch()

        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=1)

        self.refresh_saved_db_display()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))

    @pyqtSlot(dict)
    def update_active_tracks(self, active_crops):
        self.active_crops = active_crops
        current_selection = self.combo_track_id.currentText()
        
        existing_ids = [self.combo_track_id.itemText(i) for i in range(self.combo_track_id.count())]
        new_ids = [str(tid) for tid in sorted(active_crops.keys())]

        if existing_ids != new_ids:
            self.combo_track_id.blockSignals(True)
            self.combo_track_id.clear()
            self.combo_track_id.addItems(new_ids)
            if current_selection in new_ids:
                self.combo_track_id.setCurrentText(current_selection)
            self.combo_track_id.blockSignals(False)
            self.on_track_selected()

    @pyqtSlot(dict)
    def add_log(self, log):
        self.detection_logs.append(log)

    def on_track_selected(self):
        track_id_str = self.combo_track_id.currentText()
        if track_id_str and track_id_str.isdigit():
            track_id = int(track_id_str)
            if track_id in self.active_crops:
                crop = self.active_crops[track_id]
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_crop.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_crop.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_img)
                self.crop_preview_label.setPixmap(pixmap.scaled(110, 110, Qt.AspectRatioMode.KeepAspectRatio))

    def save_name(self):
        track_id_str = self.combo_track_id.currentText()
        name = self.input_name.text().strip()

        if not track_id_str or not name:
            QMessageBox.warning(self, "Uyarı", "Lütfen bir Track ID seçin ve isim girin.")
            return

        track_id = int(track_id_str)
        if track_id in self.active_crops:
            crop = self.active_crops[track_id]
            self.thread.assign_custom_name(track_id, name, crop)
            QMessageBox.information(self, "Başarılı", f"ID {track_id} için '{name}' ismi kaydedildi!")
            self.input_name.clear()
            self.refresh_saved_db_display()

    def clear_db(self):
        reply = QMessageBox.question(self, "Onay", "Veritabanını silmek istediğinize emin misiniz?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if os.path.exists(DB_FILE):
                os.remove(DB_FILE)
            self.thread.update_db([])
            self.thread.custom_names.clear()
            QMessageBox.information(self, "Bilgi", "Veritabanı sıfırlandı.")
            self.refresh_saved_db_display()

    def export_csv(self):
        if not self.detection_logs:
            QMessageBox.warning(self, "Uyarı", "Henüz kayıtlı tespit bulunmuyor.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "CSV Kaydet", "tespit_gecmisi.csv", "CSV Files (*.csv)")
        if file_path:
            df = pd.DataFrame(self.detection_logs)
            df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Başarılı", f"Kayıtlar başarıyla aktarıldı:\n{file_path}")

    def refresh_saved_db_display(self):
        db = load_db()
        if db:
            names = set(item['name'] for item in db)
            self.saved_list_label.setText("• " + "\n• ".join(names))
        else:
            self.saved_list_label.setText("Veritabanında henüz kayıtlı nesne yok.")

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())