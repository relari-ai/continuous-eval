import inspect
import logging
from abc import ABC, ABCMeta
from collections import defaultdict
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)

# from functools import wraps
from os import cpu_count
from typing import _GenericAlias  # type: ignore
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    get_args,
    get_origin,
)

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from tqdm import tqdm

from continuous_eval.utils.telemetry import (
    telemetry_event,
    telemetry_initializer,
)
from continuous_eval.utils.types import str_to_type_hint, type_hint_to_str

# from joblib import Parallel, delayed
# Set multiprocessing to use dill for pickling
# multiprocessing.set_start_method("fork", force=True)

# class DillProcessPoolExecutor(ProcessPoolExecutor):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._call = dill.dumps
#         self._result = dill.loads

logger = logging.getLogger("Metric")
logger.setLevel(logging.DEBUG)  # Set to lowest level to allow all messages


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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("type")
    def check_type(cls, value):
        # Handle standard types
        if isinstance(value, type) and hasattr(value, "__name__"):
            return value.__name__
        # Handle `typing` types
        if hasattr(value, "__origin__"):
            origin = get_origin(value)
            args = get_args(value)
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
            return f"{origin_name}[{args_str}]"
        raise ValueError("Invalid type provided")

    @field_serializer("type")
    def serialize_type(self, type: Any, _info):
        return type_hint_to_str(type)

    @property
    def type_hint(self):
        return str_to_type_hint(self.type)


# class MetricDecoratorMeta(ABCMeta, type):
#     def __new__(cls, name, bases, dct):
#         for attr, method in dct.items():
#             if callable(method) and attr in ["__call__", "batch"]:
#                 dct[attr] = telemetry_event(
#                     name=None, info={"type": "metric", "batch": attr == "batch"}
#                 )(method)
#         return type.__new__(cls, name, bases, dct)


class MetricDecoratorMeta(ABCMeta, type):
    def __new__(cls, name, bases, dct):
        return super().__new__(cls, name, bases, dct)

    def __getattribute__(cls, attr):
        # Dynamically wrap the method with telemetry when accessed
        method = super().__getattribute__(attr)
        if callable(method) and attr in ["__call__", "batch"]:
            return telemetry_event(
                name=None, info={"type": "metric", "batch": attr == "batch"}
            )(method)
        return method


class Metric(ABC, metaclass=MetricDecoratorMeta):
    def __init__(
        self, is_cpu_bound: bool = False, show_progress: bool = True
    ) -> None:
        super().__init__()
        self._overloaded_params = None
        self.io_bound = not is_cpu_bound
        self.max_workers = cpu_count() or 1
        if self.io_bound:
            self.max_workers = min(32, self.max_workers * 5)
        self.show_progress = show_progress

    def use(self, **kwargs) -> "Metric":
        self._overloaded_params = kwargs
        return self

    @property
    def overloaded_params(self):
        return self._overloaded_params

    def __call__(self, **kwargs):
        # Implement this method in the subclass
        raise NotImplementedError()

    def _batch_sequential(
        self, generate_items: Callable[[], Any], tot: int
    ) -> Any:
        logger.info("Using sequential batch processing")
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
        signature = inspect.signature(self.__call__)
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
            with process_pool(
                max_workers=self.max_workers, initializer=telemetry_initializer
            ) as executor:
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

        # try:
        #     results = Parallel(
        #         n_jobs=self.max_workers,
        #         prefer="processes" if not self.io_bound else "threads",
        #         verbose=0,
        #     )(
        #         delayed(self.__call__)(**item)
        #         for item in tqdm(
        #             list(generate_items()),
        #             desc=self.name,
        #             total=tot,
        #             disable=not self.show_progress,
        #         )
        #     )
        #     return results
        # except Exception as e:
        #     logger.warning(f"Parallel processing failed with error: {str(e)}")
        #     logger.warning("Error type:", type(e).__name__)
        #     logger.warning("Falling back to sequential processing")
        #     return self._batch_sequential(generate_items, tot)

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
