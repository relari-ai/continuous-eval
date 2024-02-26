import contextlib
import functools
import json
import logging
import os
import uuid
from functools import lru_cache, wraps
from pathlib import Path

import requests
from appdirs import user_data_dir

_TELEMETRY_ENDPOINT = "http://telemetry.relari.ai/"
_TELEMETRY_ENDPOINT_METRIC = _TELEMETRY_ENDPOINT + "metric"
_TELEMETRY_ENDPOINT_EVENT = _TELEMETRY_ENDPOINT + "event"
_USAGE_REQUESTS_TIMEOUT_SEC = 1
_USER_DATA_DIR_NAME = "continuous_eval"
_DO_NOT_TRACK = "CONTINUOUS_EVAL_DO_NOT_TRACK"
_DEBUG_TELEMETRY = "CONTINUOUS_EVAL_DEBUG_TELEMETRY"
_USER_ID_PREFIX = "ce-"


def _is_server_reachable(url):
    try:
        requests.get(url)
    except requests.ConnectionError:
        return False
    return True


@lru_cache(maxsize=1)
def _do_not_track() -> bool:
    server_reachable = _is_server_reachable(_TELEMETRY_ENDPOINT)
    do_not_track = os.environ.get(_DO_NOT_TRACK, str(False)).lower() == "true"
    return not server_reachable or do_not_track


@lru_cache(maxsize=1)
def _debug_telemetry() -> bool:
    return os.environ.get(_DEBUG_TELEMETRY, str(False)).lower() == "true"


@lru_cache(maxsize=1)
def _get_or_generate_uid() -> str:
    user_id_path = Path(user_data_dir(appname=_USER_DATA_DIR_NAME))
    user_id_path.mkdir(parents=True, exist_ok=True)
    uuid_filepath = user_id_path / "config.json"
    user_id = None
    if uuid_filepath.is_file():
        # try reading the file first
        try:
            user_id = json.load(open(uuid_filepath))["userid"]
        except Exception:
            pass
    if user_id is None:
        user_id = _USER_ID_PREFIX + uuid.uuid4().hex
        try:
            with open(uuid_filepath, "w") as f:
                json.dump({"userid": user_id}, f)
        except Exception:
            pass
    return user_id


class AnonymousTelemetry:
    def __init__(self):
        self.uid = _get_or_generate_uid()
        self._batch_mode = False

    def metric_telemetry(self, fcn):
        @wraps(fcn)
        def wrapper(*args, **kwargs):
            metric = fcn.__qualname__.split(".")[0]
            if metric != "Metric":
                telemetry.log_metric_call(metric)
            return fcn(*args, **kwargs)

        return wrapper

    def batch_metric_telemetry(self, fcn):
        def wrapper(*args, **kwargs):
            metric = fcn.__qualname__.split(".")[0]
            if metric != "Metric":
                self.log_metric_call(metric)
            with self.batch():
                return fcn(*args, **kwargs)

        return wrapper

    @contextlib.contextmanager
    def batch(self):
        old_state = self._batch_mode
        self._batch_mode = True
        try:
            yield self
        finally:
            self._batch_mode = old_state

    def log_metric_call(self, metric: str):
        self._track(
            endpoint=_TELEMETRY_ENDPOINT_METRIC,
            payload={"classname": metric, "uid": self.uid},
        )

    def log_event(self, tag: str, info: str):
        self._track(
            endpoint=_TELEMETRY_ENDPOINT_EVENT,
            payload={
                "tag": tag,
                "info": info,
                "uid": self.uid,
            },
        )

    def _track(self, endpoint, payload: dict):
        if _do_not_track() or self._batch_mode:
            return
        try:
            requests.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=_USAGE_REQUESTS_TIMEOUT_SEC,
            )
        except Exception as e:
            # This way it silences all thread level logging as well
            if _debug_telemetry():
                logging.debug(f"Telemetry error: {e}")


telemetry = AnonymousTelemetry()


def telemetry_event(tag="Unknown"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            telemetry.log_event(tag, info=func.__qualname__)
            result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator
