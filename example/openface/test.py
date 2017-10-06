from coremltools.models import MLModel
from PIL import Image

print("Loading model...")
m = MLModel("openface.mlmodel")

test = Image.open("adams.png")

print("Predicting...")

pred = m.predict({"input" : test})

print("Prediction: ", pred)
