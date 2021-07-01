from .utils import draw_boxes
from detecto.utils import read_image
from detecto.core import Dataset
from detecto.core import Model


def fit_boxes(boxes):
    points =  []
    for i, e in enumerate(boxes):
        points.append((e[0], e[1]))
        points.append((e[2], e[3]))
    return sorted(points)


class Finder:
    def __init__(self, model_file: str, desired_class: str="WhiteBear"):
        self.desired_class = desired_class
        self.model_file = model_file
        self.model = Model.load(model_file, [desired_class])

    def find(self, im: str, color: str = "#AA0000") -> tuple:
        img = read_image(im)
        _, boxes, _ = self.model.predict(img)
        return draw_boxes(img_file=im, boxes=boxes, color=color) # file_save_name="out_" + image_file)

    def learn(self, path: str, epochs: int, learning_rate: float, lr_step_size: int, verbose: bool):
        MyDataset = Dataset(path)
        model = Model([self.desired_class])
        model.fit(dataset=MyDataset, epochs=epochs, learning_rate=learning_rate, lr_step_size=lr_step_size,
                  verbose=verbose)
        model.save(self.model_file)

def main():
    f = Finder("MyModel.pth", "WhiteBear")
    f.find(input("filename=")).show()


if __name__ =='__main__':
    main()
