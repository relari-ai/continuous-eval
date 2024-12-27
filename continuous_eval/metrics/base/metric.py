import inspect
import logging
import os
from abc import ABC, ABCMeta
from collections import defaultdict
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from os import cpu_count
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    _GenericAlias,  # type: ignore
    get_args,
    get_origin,
)

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from tqdm import tqdm

from continuous_eval.utils.telemetry import telemetry
from continuous_eval.utils.types import str_to_type_hint, type_hint_to_str

load_dotenv()

logger = logging.getLogger("Metric")
logger.setLevel(logging.DEBUG)  # Set to lowest level to allow all messages

_DISABLE_MULTIPROCESSING_ENV_VAR = "CONTINUOUS_EVAL_DISABLE_MULTIPROCESSING"


class Arg(BaseModel):
    type: Any = str
    description: str = ""
    is_required: bool = True
    is_ground_truth: bool = False
    default: Any = None

    @field_validator("type")
    def check_type(cls, value):
        if isinstance(value, type):
            return value
        if isinstance(value, TypeVar):
            return value
        if isinstance(
            value, _GenericAlias
        ):  # For types like Union[str, List[str]]
            return value
        return False

    @field_validator("default")
    def check_default(cls, value):
        if value is not None and not isinstance(value, cls.type):
            raise ValueError(f"Default value {value} is not of type {cls.type}")
        return value

    @field_serializer("type")
    def serialize_type(self, type: Any, _info):
        return type_hint_to_str(type)

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            type=str_to_type_hint(data["type"]),
            description=data.get("description", ""),
            is_required=data.get("is_required", True),
            is_ground_truth=data.get("is_ground_truth", False),
            default=data.get("default", None),
        )

    def to_dict(self):
        return {
            "type": type_hint_to_str(self.type),
            "description": self.description,
            "is_required": self.is_required,
            "is_ground_truth": self.is_ground_truth,
            "default": self.default,
        }


class Field(BaseModel):
    type: Any  # a type hint, manually validated
    limits: Optional[Tuple[float, float]] = None
    internal: bool = False
    description: Optional[str] = None
    type_hint: str = "Any"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def post_init(self, __context):
        # Handle standard types
        if isinstance(self.type, type) and hasattr(self.type, "__name__"):
            self.type_hint = self.type.__name__
        # Handle `typing` types
        if hasattr(self.type, "__origin__"):
            origin = get_origin(self.type)
            args = get_args(self.type)
            args_str = ", ".join(
                [
                    arg.__name__ if isinstance(arg, type) else str(arg)
                    for arg in args
                ]
            )
            if origin is not None:
                origin_name = (
                    origin.__name__
                    if hasattr(origin, "__name__")
                    else origin._name
                )
            else:
                origin_name = ""
            self.type_hint = f"{origin_name}[{args_str}]"
        raise ValueError("Invalid type provided")

    @field_serializer("type")
    def serialize_type(self, type: Any, _info):
        return type_hint_to_str(type)


class MetricDecoratorMeta(ABCMeta, type):
    def __new__(cls, name, bases, dct):
        # Skip the Metric class itself
        for attr, method in dct.items():
            if callable(method) and attr == "batch":
                dct[attr] = telemetry.event(
                    name=None,
                    info={"type": "metric", "batch": attr == "batch"},
                )(method)
        return type.__new__(cls, name, bases, dct)


class Metric(ABC, metaclass=MetricDecoratorMeta):
    def __init__(
        self,
        is_cpu_bound: bool = False,
        disable_multiprocessing: bool = False,
        show_progress: bool = True,
    ) -> None:
        super().__init__()
        self._overloaded_params = None
        self.io_bound = not is_cpu_bound
        if (
            disable_multiprocessing
            or os.getenv(_DISABLE_MULTIPROCESSING_ENV_VAR, "false").lower()
            == "true"
        ):
            self.max_workers = None
        else:
            # Compute the number of workers based on the number of cores
            self.max_workers = cpu_count() or 1
            if self.io_bound:
                # If the metric is IO-bound, use a larger number of workers
                self.max_workers = min(32, self.max_workers * 5)
        self.show_progress = show_progress

    def use(self, **kwargs) -> "Metric":
        self._overloaded_params = kwargs
        return self

    @property
    def overloaded_params(self):
        return self._overloaded_params

    def __call__(self, *args, **kwargs):
        telemetry.log_event(
            name=self.name, info={"type": "metric", "batch": False}
        )
        return self.compute(*args, **kwargs)

    def compute(self, **kwargs):
        # Implement this method in the subclass
        raise NotImplementedError()

    def _batch_sequential(
        self, generate_items: Callable[[], Any], tot: int
    ) -> Any:
        return [
            self.__call__(**kwargs)
            for kwargs in tqdm(
                generate_items(),
                desc=self.name,
                disable=not self.show_progress,
                total=tot,
            )
        ]

    def batch(self, **kwargs) -> Any:
        signature = inspect.signature(self.compute)
        arg_names = set(signature.parameters.keys()) - {"kwargs"}
        tot = len(next(iter(kwargs.values())))

        def generate_items():
            for idx in range(tot):
                kw = {key: kwargs[key][idx] for key in arg_names}
                yield kw

        if self.max_workers is None or self.max_workers == 1:
            return self._batch_sequential(generate_items, tot)
        process_pool = (
            ThreadPoolExecutor if self.io_bound else ProcessPoolExecutor
        )
        try:
            with process_pool(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self.__call__, **item)
                    for item in generate_items()
                ]
            results = [
                future.result()
                for future in tqdm(
                    as_completed(futures),
                    total=tot,
                    desc=self.name,
                    disable=not self.show_progress,
                )
            ]
            return results
        except Exception as e:
            logger.warning(f"Processing failed with error: {str(e)}")
            logger.warning("Falling back to sequential processing")
            return self._batch_sequential(generate_items, tot)

    def aggregate(self, results: List[Any]) -> Any:
        # Default implementation
        def sanitize(results: List[Any]) -> List[Any]:
            return [
                {k: v for k, v in r.items() if not isinstance(v, (list, str))}
                for r in results
            ]

        sanitized_results = sanitize(results)
        sums = defaultdict(float)
        counts = defaultdict(int)
        for result in sanitized_results:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    sums[key] += value
                    counts[key] += 1
        means = {
            key: sums[key] / counts[key] for key in sums if counts[key] > 0
        }
        return means

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def schema(self) -> Dict[str, Field]:
        # Implement this method in the subclass
        raise NotImplementedError()

    @property
    def args(self) -> Dict[str, Arg]:
        # Implement this method in the subclass
        raise NotImplementedError()

    @property
    def help(self):
        # Implement this method in the subclass
        return (
            self.__doc__.strip() if self.__doc__ else "No description available"
        )

    def asdict(self):
        return {
            "__class__": self.__class__.__name__,
            "name": self.name,
        }
