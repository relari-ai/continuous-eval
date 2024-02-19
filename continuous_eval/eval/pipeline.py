from dataclasses import dataclass, field
from typing import Any, List, Optional, Set, Tuple, get_origin

from continuous_eval.eval.dataset import Dataset, DatasetField
from continuous_eval.eval.modules import Module


@dataclass
class ModuleOutput:
    selector: callable = field(default=lambda x: x)
    module: Optional[Module] = None

    def __call__(self, *args: Any) -> Any:
        return self.selector(*args)


@dataclass
class Graph:
    nodes: Set[str]
    edges: Set[Tuple[str, str]]
    dataset_edges: Set[Tuple[DatasetField, str]]


class Pipeline:
    def __init__(self, modules: List[Module], dataset: Dataset) -> None:
        self._modules = modules
        self._dataset = dataset
        self._graph = self._build_graph()

    @property
    def modules(self):
        return self._modules

    @property
    def dataset(self):
        return self._dataset

    def module_by_name(self, name: str) -> Module:
        for module in self._modules:
            if module.name == name:
                return module
        raise ValueError(f"Module {name} not found")

    def get_metric(self, module_name: str, metric_name: str):
        module = self.module_by_name(module_name)
        try:
            metric = [m for m in module.eval if m.name == metric_name][0]
        except IndexError:
            raise ValueError(f"Metric {metric_name} not found in module {module_name}")
        return metric

    def _validate_modules(self):
        names = set()
        for module in self._modules:
            if module.name in names:
                raise ValueError(f"Module {module.name} already exists")
            if module.reference is not None:
                assert (
                    module.reference is not None and module.reference in self._dataset.fields
                ), f"Field {module.reference.name} not found"
            names.add(module.name)
            if self._dataset is not None and module.expected_output is not None:
                if get_origin(module.output) != get_origin(self._dataset.getattr(self, module.expected_output).type):
                    raise ValueError(f"Field {module.output} does not match expected type in the dataset.")

    def _build_graph(self):
        nodes = {m.name: m for m in self._modules}
        edges = set()
        dataset_edges = set()
        for module in self._modules:
            if isinstance(module.input, Module):
                assert module in self._modules, f"Module {module.input.name} not found"
                edges.add((module.input.name, module.name))
            elif isinstance(module.input, DatasetField):
                assert module.input in self._dataset.fields, f"Field {module.input.name} not found"
                dataset_edges.add((module.input, module.name))
            elif isinstance(module.input, tuple):
                for x in module.input:
                    if isinstance(x, Module):
                        assert x in self._modules, f"Module {x.name} not found"
                        edges.add((x.name, module.name))
                    elif isinstance(x, DatasetField):
                        assert x in self._dataset.fields, f"Field {x.name} not found"
                        dataset_edges.add((x, module.name))
            else:
                raise ValueError(f"Invalid input type {module.input}")
        return Graph(nodes, edges, dataset_edges)

    def graph_repr(self, with_type_hints: bool = False):
        repr_str = "graph TD;\n"
        dataset_node_label = "Dataset"
        repr_str += f"    {dataset_node_label}(({dataset_node_label}));\n"
        for edge in self._graph.edges:
            start, end = edge
            if with_type_hints:
                type_hint = type_hint_to_str(self.module_by_name(start).output)
                repr_str += f'    {start}-->|"{type_hint}"|{end};\n'
            else:
                repr_str += f"    {start} --> {end};\n"
        for d_edge in self._graph.dataset_edges:
            dataset_field_name, end_node = d_edge[0].name, d_edge[1]
            # Adding the dataset edge with label
            repr_str += f'    {dataset_node_label} -. "{dataset_field_name}" .-> {end_node};\n'
        return repr_str
