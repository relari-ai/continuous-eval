import json
from pathlib import Path
from typing import Union

import pandas as pd

from continuous_eval.datatypes import DatumField

_MINIMAL_REQUIRED_COLUMNS = [
    [DatumField.ANSWER, DatumField.QUESTION],
    [DatumField.ANSWER, DatumField.GROUND_TRUTH_ANSWER],
    [DatumField.ANSWER, DatumField.RETRIEVED_CONTEXTS],
    [DatumField.QUESTION, DatumField.RETRIEVED_CONTEXTS],
    [DatumField.RETRIEVED_CONTEXTS, DatumField.GROUND_TRUTH_CONTEXTS],
]


class Dataset(pd.DataFrame):
    def __init__(self, data=None, index=None, columns=None, copy=False):
        super().__init__(data=data, index=index, columns=columns, copy=copy)
        self.validate()

    def iterate(self):
        for _, row in self.iterrows():
            yield row.to_dict()

    def datum(self, index):
        return self.iloc[index].to_dict()

    def to_dict(self, *args, **kwargs):
        if "orient" not in kwargs:
            kwargs["orient"] = "records"
        return super().to_dict(*args, **kwargs)

    def validate(self):
        if len(self) == 0:
            raise ValueError("Dataset is empty")
        if not "question" in self.columns:
            raise ValueError("The dataset should at least question column not found")
        if not any(
            [
                all([col.value in self.columns for col in required_columns])
                for required_columns in _MINIMAL_REQUIRED_COLUMNS
            ]
        ):
            raise ValueError(
                "The dataset should at least have one of the following columns: {}".format(_MINIMAL_REQUIRED_COLUMNS)
            )

        for item in self.values:
            if DatumField.QUESTION.value in self.columns:
                itm = item[self.columns.get_loc(DatumField.QUESTION.value)]
                if not isinstance(itm, str):
                    raise ValueError("Answer must be a string")
            if DatumField.ANSWER.value in self.columns:
                itm = item[self.columns.get_loc(DatumField.ANSWER.value)]
                if not isinstance(itm, str):
                    raise ValueError("Answer must be a string")
            if DatumField.GROUND_TRUTH_ANSWER.value in self.columns:
                itm = item[self.columns.get_loc(DatumField.GROUND_TRUTH_ANSWER.value)]
                if not isinstance(itm, list):
                    raise ValueError("Ground truth answers must be a list of strings")
                for answer in itm:
                    if not isinstance(answer, str):
                        raise ValueError("Ground truth answers must be a list of strings")
            if DatumField.RETRIEVED_CONTEXTS.value in self.columns:
                itm = item[self.columns.get_loc(DatumField.RETRIEVED_CONTEXTS.value)]
                if isinstance(itm, list):
                    for ctx in itm:
                        if not isinstance(ctx, str):
                            raise ValueError("Retrieved context must be a list of strings or a string")
                elif not isinstance(itm, str):
                    raise ValueError("Retrieved context must be a list of strings or a string")
            if DatumField.GROUND_TRUTH_CONTEXTS.value in self.columns:
                itm = item[self.columns.get_loc(DatumField.GROUND_TRUTH_CONTEXTS.value)]
                if not isinstance(itm, list):
                    raise ValueError("Ground truth context must be a list of strings")
                for answer in itm:
                    if not isinstance(answer, str):
                        raise ValueError("Ground truth context must be a list of strings")

    @classmethod
    def from_jsonl(cls, path: Union[str, Path]):
        with open(path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        return cls(data)

    def to_jsonl(self, path: Union[str, Path]):
        with open(path, "w") as f:
            f.write(self.to_json(orient="records", lines=True))
