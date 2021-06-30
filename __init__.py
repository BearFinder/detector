from .detect import Finder
from PIL import Image
import io

finder = None

def init(model_path):
    global finder
    finder = Finder(model_path)


def image_json(image_bytes: bytes, width: int, height: int) -> dict:
    image = Image.frombytes('RGB', (width, height), image_bytes, 'raw')
    image.save("tmp.jpg")
    image = finder.find("tmp.jpg")
    raw = io.BytesIO()
    return {"src": image.save(raw, format="jpg").getvalue(), "width": width, "height": height}

def image_pillow(image: Image) -> Image:
    image.save("tmp.jpg")
    image.save("tmp.jpg")
    return finder.find("tmp.jpg")
