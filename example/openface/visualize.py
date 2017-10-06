import pydot # import pydot or you're not going to get anywhere my friend :D
from coremltools.models import MLModel

# first you create a new graph, you do that with pydot.Dot()
graph = pydot.Dot(graph_type='graph')

print("Loading model...")
m = MLModel("openface.mlmodel")

print("Loading layers...")
layers = m.get_spec().neuralNetwork.layers

print("Drawing graph...")
for layer in layers[::-1]: #reverse Order
    for input_name in layer.input:
        edge = pydot.Edge(input_name, layer.name)
        # and we obviosuly need to add the edge to our graph
        graph.add_edge(edge)


graph.write_png('coremlmodel.png')
