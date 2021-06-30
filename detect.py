from os import listdir

from detecto.utils import read_image
from detecto.core import Dataset
from detecto.visualize import show_labeled_image
from detecto.core import Model

basepath = "src/jpegs/"
# MyDataset = Dataset(basepath)
# model = Model(['WhiteBear'])
# model.fit(dataset=MyDataset, epochs=10, learning_rate=0.001, lr_step_size=5, verbose=True)
# model.save("MyModel.pth")
model = Model.load("MyModel.pth", ['WhiteBear'])

for i in listdir(basepath):
    if i.endswith(".JPG"):
        img = read_image(basepath + i)
        labels, boxes, scores = model.predict(img)
        print(labels, boxes, scores, sep='\n')
        show_labeled_image(img, boxes, labels)
