from ultralytics import YOLO

class GunDetector:
    def __init__(self,model_path="model path")
        self.model= YOLO(model_path)

    def detect(self, image):
        results = self.model(image,verbose=False , conf=0.7)
        boxes = results[0].boxes
        gun_centers = []
        
        for box in boxes: 
            x0 , y0, x1, y1 = box.xyxy[0].cpu().numpy()
            center = (int((x0 + x1)/2), int((y0+y1)/ 2))
            gun_centers.append(center)

        annotated_image =results[0].plot()
        return gun_centers, annotated_image
