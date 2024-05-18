import json
import typing
from dataclasses import dataclass
from pathlib import Path

import yaml

from continuous_eval.eval.types import UID, ToolCall
from continuous_eval.eval.utils import type_hint_to_str

_SAFE_DICT = {k: v for k, v in typing.__dict__.items() if not k.startswith("__")}
_SAFE_DICT["UID"] = UID
_SAFE_DICT["ToolCall"] = ToolCall


@dataclass(frozen=True)
class DatasetField:
    name: str
    type: type = typing.Any  # type: ignore
    description: str = ""
    is_ground_truth: bool = False

    def to_dict(self):
        return {
            "type": type_hint_to_str(self.type),
            "description": self.description,
            "ground_truth": self.is_ground_truth,
        }


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    description: str
    format: str
    license: str
    fields: typing.Dict[str, DatasetField]

    def to_yaml(self):
        return {
            "name": self.name,
            "description": self.description,
            "format": self.format,
            "license": self.license,
            "fields": {field_name: field.to_dict() for field_name, field in self.fields.items()},
        }


class Dataset:
    def __init__(
        self,
        dataset_path: typing.Union[str, Path],
        manifest_path: typing.Optional[typing.Union[str, Path]] = None,
    ) -> None:
        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)
        assert dataset_path.exists(), f"Dataset {dataset_path} does not exist"
        if dataset_path.is_file() and manifest_path is None:
            manifest_path = dataset_path.parent / "manifest.yaml"
        elif dataset_path.is_dir():
            manifest_path = dataset_path / "manifest.yaml"
            dataset_path = dataset_path / "dataset.jsonl"
        else:
            manifest_path = None

        if dataset_path.suffix != ".jsonl":
            raise ValueError(f"Dataset file must be a .jsonl file, found {dataset_path.suffix}")
        # load jsonl dataset
        with open(dataset_path, "r") as json_file:
            self._data = [json.loads(x) for x in json_file.readlines()]
        self._manifest = self._load_or_infer_manifest(manifest_path)
        self._create_dynamic_properties()

    @classmethod
    def from_data(cls, data: typing.List[typing.Dict[str, typing.Any]]):
        dataset = cls.__new__(cls)
        dataset._data = data
        dataset._manifest = dataset._infer_manifest()
        dataset._create_dynamic_properties()
        return dataset

    def save(self, file_path: typing.Union[str, Path], save_manifest: bool = False):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with open(file_path, "w") as json_file:
            for sample in self._data:
                json_file.write(json.dumps(sample) + "\n")

        if save_manifest:
            manifest_path = file_path.parent / "manifest.yaml"
            with open(manifest_path, "w") as manifest_file:
                manifest_file.write(yaml.dump(self._manifest.to_yaml()))

    def _load_or_infer_manifest(self, manifest_path: typing.Optional[Path]) -> DatasetManifest:
        if manifest_path is None or not manifest_path.exists():
            return self._infer_manifest()
        else:
            with open(manifest_path, "r") as manifest_file:
                manifest = yaml.safe_load(manifest_file)
                fields = {
                    field_name: DatasetField(
                        name=field_name,
                        type=eval(field_info["type"], _SAFE_DICT),
                        description=field_info.get("description", ""),
                        is_ground_truth=field_info.get("ground_truth", False),
                    )
                    for field_name, field_info in manifest["fields"].items()
                }
                return DatasetManifest(
                    name=manifest.get("name", ""),
                    description=manifest.get("description", ""),
                    format=manifest.get("format", ""),
                    license=manifest.get("license", ""),
                    fields=fields,
                )

    def _infer_manifest(self):
        fields = dict()
        for sample in self._data:
            for field_name, field_value in sample.items():
                dataset_field = DatasetField(
                    name=field_name,
                    type=type(field_value),
                    description="",
                    is_ground_truth=False,
                )
                if field_name not in fields:
                    fields[field_name] = dataset_field
                else:
                    if fields[field_name].type != dataset_field.type:
                        raise ValueError(
                            f"Field {field_name} has inconsistent types: {fields[field_name]['type']} and {type(field_value).__name__}"
                        )
        return DatasetManifest(
            name="",
            description="",
            format="",
            license="",
            fields=fields,
        )

    def _create_dynamic_properties(self):
        # Dynamically add a property for each field
        for field_name, field_info in self._manifest.fields.items():
            setattr(self, field_name, field_info)

    def filed_types(self, name: str) -> type:
        return getattr(self, name).type

    @property
    def data(self):
        return self._data

    @property
    def name(self):
        return self._manifest.name

    @property
    def description(self):
        return self._manifest.description

    @property
    def format(self):
        return self._manifest.format

    @property
    def license(self):
        return self._manifest.license

    @property
    def fields(self) -> typing.List[DatasetField]:
        return list(self._manifest.fields.values())

    def get_field(self, name: str) -> DatasetField:
        return self._manifest.fields[name]

    def __getitem__(self, key: str):
        return [x[key] for x in self._data]

    def __len__(self):
        return len(self._data)

    def filter(self, fcn: typing.Callable):
        self._data = [x for x in self._data if fcn(x)]
        return self

    def sample(self, size: typing.Union[float, int]):
        import random

        if size > 1 and isinstance(size, int):
            assert size <= len(self._data), f"Sample size {size} is larger than the dataset size {len(self._data)}"
            k = size
        elif size > 0 and size < 1 and isinstance(size, float):
            k = int(len(self._data) * size)
        else:
            raise ValueError(f"Invalid sample size {size}")
        idx = random.choices(range(len(self._data)), k=k)
        idx = sorted(idx)
        self._data = [self._data[i] for i in idx]
        return self
