from ultralytics import YOLO
import pandas as pd

model=YOLO('yolov8n-cls.pt')  #loading a yolo nano pretrained model

results=model.train(data= r"path\to\your\Currency_Dataset", epochs=12, imgsz=64)

metrics=results.metrics

data = {
    'epoch': [i + 1 for i in range(len(metrics['train_loss']))],
    'train_loss': metrics['train_loss'],
    'val_loss': metrics['val_loss'],
    'accuracy': metrics['val_acc'],  
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('train_results.csv', index=False)

print("Training results saved to train_results.csv")
