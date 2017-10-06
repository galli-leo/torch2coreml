import pydot
from torch.utils.serialization import load_lua

graph = pydot.Dot(graph_type='graph')

m = load_lua("openface.t7")

last_layer = None

num = len(m.modules)

for layer in m.modules[::-1]:
    if last_layer is None:
        last_layer = layer
        continue
    fmt = "{0}_{1}\n{0.output.shape}"
    input_name = fmt.format(layer, num)
    output_name = fmt.format(last_layer, num+1)
    edge = pydot.Edge(input_name, output_name)
    graph.add_edge(edge)
    last_layer = layer
    num -= 1

graph.write_png('torchmodel.png')
