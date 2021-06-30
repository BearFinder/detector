from utils import draw_boxes
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
        _, boxes, _ = self.model.predict(img)
        return draw_boxes(img_file=image_file, boxes=boxes) # file_save_name="out_" + image_file)

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
