from .utils import *

from detecto.utils import read_image
from detecto.core import Dataset
from detecto.core import Model


class Finder:
    def __init__(self, model_file: str, desired_class: str):
        self.desired_class = desired_class
        self.model_file = model_file
        self.model = Model.load(model_file, [desired_class])

    def find(self, image_file: str) -> tuple:
        img = read_image(image_file)
        labels, boxes, scores = self.model.predict(img)
        print(labels, boxes, scores, sep='\n')
        slugs = [make_slug(b, s) for b, s in zip(labels, scores)]
        bears = [map(round, react.tolist()) for react in boxes]
        return bears, slugs

    def find_bears(self, image_file: str) -> bytes:
        bears, slugs = self.find(image_file)
        return b'' # TODO

    def learn(self, path: str, epochs: int, learning_rate: float, lr_step_size: int, verbose: bool):
        MyDataset = Dataset(path)
        model = Model([self.desired_class])
        model.fit(dataset=MyDataset, epochs=epochs, learning_rate=learning_rate, lr_step_size=lr_step_size, verbose=verbose)
        model.save(self.model_file)

def run():
    Finder("MyModel.pth", "WhiteBear")