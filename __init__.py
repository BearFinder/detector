import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2 as cv


model = None
# Transform
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
 

def find(input_file: str, output_file: str, color: tuple = (100, 0, 0)) -> None:
   
    # Image
    bear_coords = []
    FRAME_SIZE = 112
    img = cv.imread(input_file)
    for x in range(0, img.shape[1], FRAME_SIZE // 2):
        # print(f'{x}/{img.shape[1]}')
        for y in range(0, img.shape[0], FRAME_SIZE // 2):
            frame = Image.fromarray(img[y : y + FRAME_SIZE, x : x + FRAME_SIZE, 0:3])
            tensor_frame = IMG_TRANSFORM(frame).unsqueeze(0)
            # Classification
            with torch.no_grad():
                output = model(tensor_frame)
                _, pred = torch.max(output, 1)
            # Detection
            if pred.item() == 0:
                bear_coords.append((x, y))
    # Drawing box's
    for x, y in bear_coords:
        img = cv.rectangle(img, (x, y), (x + FRAME_SIZE, y + FRAME_SIZE), color, 10)
    
    cv.imwrite(output_file, cv.cvtColor(img, cv.COLOR_BGR2RGB))


def init(model_path):
    global model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    

def image_file(input_file: str, output_file: str):
    find(input_file, output_file)


def image_pillow(image_file: str, color: str) -> Image:
    find(image_file, "tmp.png", color=color)
    return Image.open("tmp.png")
