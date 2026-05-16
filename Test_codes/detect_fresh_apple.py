from ultralytics import YOLO
model=YOLO('/Users/apple/Downloads/best.pt')
results=model.predict(
    source='/Users/apple/Downloads/Apfel-Fuji.jpg'
)
print(results[0].show())