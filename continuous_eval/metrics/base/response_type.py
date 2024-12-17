import json
import re
from abc import ABC, ABCMeta
from enum import Enum, EnumMeta
from typing import Dict, List, Optional, Type, Union

from json_repair import repair_json

from continuous_eval.utils.types import str_to_type_hint, type_hint_to_str


# Creating a combined metaclass that handles both EnumMeta and ABCMeta features
class _ABCResponseFormatMeta(EnumMeta, ABCMeta):
    pass


class _BaseClass(ABC):
    def serialize(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def deserialize(self):
        raise NotImplementedError("Subclass must implement abstract method")


class ScoringFunction(_BaseClass):
    def score(self, input_val: str):
        raise NotImplementedError("Subclass must implement abstract method")


# Defining a mixin class for the abstract base behavior
class ResponseFormatBaseType(_BaseClass, metaclass=ABCMeta):
    def values(self):
        raise NotImplementedError("Subclass must implement abstract method")

    @property
    def type(self):
        raise NotImplementedError("Subclass must implement abstract method")


# Using the combined metaclass in the Enum declaration properly
class CategoryResponseType(
    ScoringFunction,
    ResponseFormatBaseType,
    Enum,
    metaclass=_ABCResponseFormatMeta,
):
    def __new__(cls, value):
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    @classmethod
    def serialize(self):
        return {"class": self.__name__}

    @classmethod
    def deserialize(self, serialized: Dict):
        return self

    @classmethod
    def score(self, input_val: str):
        cat = [v for v in self.values()]
        input_words = input_val.lower().split()
        first_occurrences = {
            word: input_words.index(word.lower())
            for word in cat
            if word.lower() in input_words
        }
        if first_occurrences:
            return self(min(first_occurrences, key=first_occurrences.get)).value
        raise ValueError("No matching categories found")

    @property
    def type(self):
        return str

    @classmethod
    def values(cls):
        return [
            member.value for member in cls
        ]  # Custom method to return values as strings


ResponseFormat = Union[CategoryResponseType, ResponseFormatBaseType]

###################### Common Format Types ######################


class GoodOrBad(CategoryResponseType):
    GOOD = "good"
    BAD = "bad"


class Boolean(CategoryResponseType):
    TRUE = "true"
    FALSE = "false"


class YesOrNo(CategoryResponseType):
    YES = "yes"
    NO = "no"


class Integer(ScoringFunction, ResponseFormatBaseType):
    def __init__(self, ge: int, le: int):
        assert ge < le, "ge must be less than le"
        self._ge: int = ge
        self._le: int = le

    def serialize(self):
        return {
            "class": self.__class__.__name__,
            "ge": self._ge,
            "le": self._le,
        }

    @classmethod
    def deserialize(cls, serialized: Dict):
        if (
            "class" in serialized
            and "ge" in serialized
            and "le" in serialized
            and serialized["class"] == cls.__name__
        ):
            return cls(serialized["ge"], serialized["le"])

    def __str__(self):
        return f"Integer(ge={self._ge}, le={self._le})"

    def values(self):
        return list(range(self._ge, self._le + 1))

    def score(self, input_val: str):
        num = self._numeric_matcher(input_val)
        if num is None:
            return self.ge
        return max(self._ge, min(self._le, num))

    def _numeric_matcher(self, input_val) -> Optional[float]:
        pattern = r"\d+(?:\.\d+)?"  # Match any number (integer or float)
        matches = re.findall(pattern, input_val)
        if not matches:
            return None
        return max(self._ge, min(self._le, float(matches[0])))

    @property
    def ge(self):
        return self._ge

    @property
    def le(self):
        return self._le

    @property
    def type(self):
        return int  # OpenAI doesn't support Range types in response_format yet

    def weighted_score(self, probabilities: Dict[int, float]):
        return sum(
            (cat - self._ge) * prob for cat, prob in probabilities.items()
        ) / (self._le - self._ge)


class JSON(ScoringFunction):
    def __init__(self, schema: Union[Dict[str, Type], List[Dict[str, Type]]]):
        self.is_list = isinstance(schema, list)
        self.schema: Dict[str, Type] = schema if not self.is_list else schema[0]  # type: ignore
        for key, value in self.schema.items():
            if not isinstance(value, type):
                raise ValueError(
                    f"All values in the schema must be type hints, got {type(value)} for key {key}"
                )

    def serialize(self):
        enc_schema = (
            json.dumps(
                {k: type_hint_to_str(v) for k, v in self.schema.items()},  # type: ignore
                indent=0,
                ensure_ascii=True,
            )
            .replace("\n", "")
            .replace(" ", "")
        )
        return {
            "class": self.__class__.__name__,
            "schema": enc_schema,
            "is_list": self.is_list,
        }

    @classmethod
    def deserialize(cls, serialized: Dict):
        if (
            "class" in serialized
            and "schema" in serialized
            and serialized["class"] == cls.__name__
        ):
            schema = {
                k: str_to_type_hint(v)
                for k, v in json.loads(serialized["schema"]).items()
            }
            _schema = schema if not serialized["is_list"] else [schema]
            return cls(_schema)
        raise ValueError("Invalid serialized JSON")

    def _check_json_obj(self, json_obj: Dict):
        for key in self.schema.keys():
            if key not in json_obj:
                json_obj[key] = None
            else:
                try:
                    json_obj[key] = self.schema[key](json_obj[key])
                except Exception:
                    json_obj[key] = None
        return json_obj

    def score(self, input_val: str):
        json_obj = repair_json(input_val, return_objects=True)  # type: ignore
        if not self.is_list:
            return self._check_json_obj(json_obj)  # type: ignore
        else:
            json_obj = json_obj if isinstance(json_obj, list) else [json_obj]  # type: ignore
            return [self._check_json_obj(item) for item in json_obj]  # type: ignore

    @property
    def type(self):
        return dict  # OpenAI doesn't support Range types in response_format yet


def get_response_format(response_format: Dict[str, str]):
    if "class" not in response_format:
        raise ValueError("Response format must contain class")
    if response_format["class"].startswith("Integer"):
        return Integer.deserialize(response_format)
    elif response_format["class"].startswith("JSON"):
        return JSON.deserialize(response_format)
    elif response_format["class"] == "GoodOrBad":
        return GoodOrBad.deserialize(response_format)
    elif response_format["class"] == "Boolean":
        return Boolean.deserialize(response_format)
    elif response_format["class"] == "YesOrNo":
        return YesOrNo.deserialize(response_format)
    else:
        raise ValueError(f"Unknown category type: {response_format}")
