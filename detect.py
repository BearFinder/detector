from detecto.utils import read_image
from detecto.core import Dataset
from detecto.core import Model
from PIL import Image, ImageDraw


class Finder:
    def __init__(self, model_file: str, desired_class: str):
        self.desired_class = desired_class
        self.model_file = model_file
        self.model = Model.load(model_file, [desired_class])

    def find(self, image_file: str) -> tuple:
        img = read_image(image_file)
        labels, boxes, scores = self.model.predict(img)
        print(labels, boxes, scores, sep='\n')
        draw_and_save(img_file=image_file, boxes=boxes, file_save_name="out_" + image_file)
        bears = [map(round, react.tolist()) for react in boxes]
        return bears

    # def find_bears(self, image_file: str) -> bytes:
    #    bears, slugs = self.find(image_file)
    #   return b''  # TODO

    def learn(self, path: str, epochs: int, learning_rate: float, lr_step_size: int, verbose: bool):
        MyDataset = Dataset(path)
        model = Model([self.desired_class])
        model.fit(dataset=MyDataset, epochs=epochs, learning_rate=learning_rate, lr_step_size=lr_step_size,
                  verbose=verbose)
        model.save(self.model_file)


def draw_and_save(img_file: str, boxes: list, color: str = "#1700FF", file_save_name="outFile.JPG"):
    source_file = Image.open(img_file).convert("RGBA")
    draw = ImageDraw.Draw(source_file)
    for i in range(boxes.shape[0]):
        box = boxes[i]
        # box[0]: x0, box[1]: y0, box[2]: x1, box[3]: y1
        draw.rectangle([box[0].item(), box[1].item(), box[2].item(), box[3].item()], outline=color, fil=None,
                       width=2)
    source_file.save(file_save_name)


def run():
    Finder("MyModel.pth", "WhiteBear")
