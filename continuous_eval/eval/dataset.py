import json
import typing
from dataclasses import dataclass
from pathlib import Path

import yaml

UUID = str

_SAFE_DICT = {k: v for k, v in typing.__dict__.items() if not k.startswith("__")}
_SAFE_DICT["UUID"] = UUID


@dataclass(frozen=True)
class DatasetField:
    name: str
    type: type
    description: str
    is_ground_truth: bool = False


class Dataset:
    def __init__(self, dataset_path: typing.Union[str, Path]) -> None:
        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)
        assert dataset_path.exists(), f"Dataset folder {dataset_name} does not exist"
        assert (dataset_path / "manifest.yaml").exists(), f"Manifest file not found in {dataset_name}"
        assert (dataset_path / "dataset.jsonl").exists(), f"Dataset file not found in {dataset_name}"
        # Load manifest
        with open(dataset_path / "manifest.yaml", "r") as manifest_file:
            self._manifest = yaml.safe_load(manifest_file)
        # load jsonl dataset
        with open(dataset_path / "dataset.jsonl", "r") as json_file:
            self._data = [json.loads(x) for x in json_file.readlines()]
        # create dynamic properties
        self._fields = list()
        self._create_dynamic_properties()

    @property
    def fields(self):
        return self._fields

    def _create_dynamic_properties(self):
        # Dynamically add a property for each field
        for field_name, field_info in self._manifest["fields"].items():
            try:
                _field = DatasetField(
                    name=field_name,
                    type=eval(field_info["type"], _SAFE_DICT),
                    description=field_info["description"],
                    is_ground_truth=field_info.get("ground_truth", False),
                )
                self._fields.append(_field)
                setattr(self, field_name, _field)
            except:
                raise ValueError(f"Field type {field_info['type']} not supported")

    def filed_types(self, name: str) -> type:
        return getattr(self, name).type

    @property
    def data(self):
        return self._data

    # def get_value(self, field: typing.Union[str, DatasetField], index: int):
    #     if isinstance(field, str):
    #         return self._data[index][field]
    #     elif isinstance(field, DatasetField):
    #         return self._data[index][field.name]
    #     else:
    #         raise ValueError(f"field {field} not recognized")
