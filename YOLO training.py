from robotflow import Roboflow

rf = Roboflow(api_key="ZZLFCeb9P1qNDK172Ca")
project=rf.workspace("alanersia").project("gun-detect-v7vj0")
version=project.version(1)
dataset=version.download("yolov8")
from ultralytics import YOLO

model=YOLO("yolov8n")

model.train(
    data="(path) data.yaml",
    name="gun-detect-v4.0",
    epochs= 50
    imgsz=640
    batch=16
    device ="cuda"
)