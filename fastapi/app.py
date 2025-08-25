from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
import gdown
import os
from torchvision import transforms

# Конфигурация
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_URL ="https://drive.google.com/uc?id=1_hywEaCjQ8gntsoRAd1ph2h8GcYvHIEg"
MODEL_PATH = ("model1000.pth")
IMAGE_FOLDER = "images"
MASK_FOLDER = "masks"

# Словарь для цветовой карты
id2color = {
    0: (0, 0, 0),        # Фон
    1: (128, 0, 128),    # Дорога
    2: (0, 128, 128),    # Обочина
    3: (128, 0, 0),      # Здания
    4: (0, 0, 128),      # Растительность
    5: (0, 128, 0),      # Машина
    6: (128, 128, 0),    # Человек
    7: (128, 128, 128)   # Знак
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка модели
def load_model():
    checkpoint = torch.load(MODEL_PATH)
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, classes=len(id2color))
    model.load_state_dict(checkpoint)
    model.to(device=DEVICE)
    model.eval()
    return model

model = load_model()
is_processing = False


def create_color_masks(pred_mask_test):
    color_masks = []
    for mask in pred_mask_test:
        mask = mask.squeeze().numpy()  # Преобразуем в numpy и убираем лишнюю размерность
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        for label, color in id2color.items():
            color_mask[mask == label] = color
        
        color_masks.append(color_mask)
    
    return color_masks

# Эндпоинт для проверки состояния сервиса
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Эндпоинт для проверки статуса обработки изображений
@app.get("/status")
def get_status():
    if is_processing:
        return {"status": "processing"}
    return {"status": "completed"}

# Эндпоинт для обработки изображений из папки
@app.post("/process_images/")
async def process_images():
    global is_processing
    is_processing = True
    processed_files = []

    if not os.path.exists(IMAGE_FOLDER):
        is_processing = False
        return JSONResponse(status_code=404, content={"message": "Image folder not found."})

    for filename in os.listdir(IMAGE_FOLDER):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(IMAGE_FOLDER, filename)
            image = Image.open(image_path).convert("RGB")
            image = image.resize((512, 512))
            image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device=DEVICE)

            with torch.inference_mode():
                logit_mask = model(image_tensor)
                prob_mask = logit_mask.softmax(dim=1)
                pred_mask = prob_mask.argmax(dim=1)

            pred_mask_test = pred_mask.detach().cpu()
            color_masks = create_color_masks([pred_mask_test])

            # Сохранение цветной маски
            for i, color_mask in enumerate(color_masks):
                mask_image = Image.fromarray(color_mask)
                mask_image.save(os.path.join(MASK_FOLDER, f"{filename}_mask.png"))

            processed_files.append(filename)

    is_processing = False
    return {"message": "Images processed", "files": processed_files}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

