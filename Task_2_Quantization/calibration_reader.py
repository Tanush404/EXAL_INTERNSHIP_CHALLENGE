import numpy as np
import os
from PIL import Image
import onnxruntime.quantization as quant

class TinyImageNetCalibrationDataReader(quant.CalibrationDataReader):
    def __init__(self, image_dir, batch_size=10):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.image_list = [os.path.join(image_dir, f) for f in os.listdir(image_dir)[:500]]
        self.cursor = 0

    def get_next(self):
        if self.cursor >= len(self.image_list):
            return None
        
        batch_images = []
        for i in range(self.batch_size):
            if self.cursor < len(self.image_list):
                img_path = self.image_list[self.cursor]
                img = Image.open(img_path).convert('RGB').resize((224, 224), Image.BILINEAR)
                
                # Normalization (ImageNet Standard)
                img_data = (np.array(img).astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                img_data = np.expand_dims(img_data.transpose(2, 0, 1), axis=0).astype(np.float32)
                
                batch_images.append(img_data)
                self.cursor += 1
        
        return {"input": np.concatenate(batch_images)} # 'input' must match model input name