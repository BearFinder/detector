from .detect import Finder
from PIL import Image
import io

finder = None


def init(model_path):
    global finder
    finder = Finder(model_path)


def image_file(input_file: str, output_file: str):
    image = finder.find(image_file)
    image.save(output_file)


def image_pillow(image_file: str, color: str) -> Image:
    return finder.find(image_file, color=color)
