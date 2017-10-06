from coremltools.models import MLModel

m = MLModel("openface.mlmodel")
print(m.get_spec())
