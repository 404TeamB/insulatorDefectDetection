import os
import pandas as pd
from ultralytics import YOLO

class InsulatorInspector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def process_folder(self, input_folder, output_folder, report_name="report.csv"):
        os.makedirs(output_folder, exist_ok=True)
        report_data = []
        files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        for filename in files:
            img_path = os.path.join(input_folder, filename)
            results = self.model.predict(img_path, conf=0.25, verbose=False)
            
            # Сохраняем фото с визуализацией
            save_path = os.path.join(output_folder, f"det_{filename}")
            results[0].save(filename=save_path)
            
            # Заполняем отчет
            if len(results[0].boxes) == 0:
                report_data.append({"filename": filename, "class": "None", "confidence": 0.0})
            else:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    report_data.append({
                        "filename": filename,
                        "class": self.model.names[cls_id],
                        "confidence": round(float(box.conf[0]), 2)
                    })
        
        df = pd.DataFrame(report_data).sort_values(by="filename")
        report_path = os.path.join(output_folder, report_name)
        df.to_csv(report_path, index=False, encoding='utf-8-sig')
        return report_path