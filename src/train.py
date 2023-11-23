import yaml
from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")
    model.train(
        # Project
        project="Polar-Owl",
        name="yolov8n",

        # Random Seed parameters
        deterministic=True,
        seed=42,

        # Data & model parameters
        data="config.yaml", 
        save=True,
        save_period=5,
        pretrained=True,
        imgsz=1280,

        # Training parameters
        epochs=20,
        batch=4,
        workers=8,
        val=True,
        device=0,

        # Optimization parameters
        lr0=0.0195,
        patience=3,
        optimizer="SGD",
        momentum=0.957,
        weight_decay=0.0005,
        close_mosaic=5,
    )

if __name__ == '__main__':
    main()
