import io
from pathlib import Path
from typing import Union, Dict
from zipfile import ZipFile
from continuous_eval.eval.dataset import Dataset
import requests
import json
from continuous_eval.utils.telemetry import telemetry


_DATA_RESOURCES = {
    "correctness": {
        "filename": "correctness.jsonl",
        "url": "https://ceevaldata.blob.core.windows.net/examples/correctness.jsonl",
        "type": "dataset",
    },
    "retrieval": {
        "filename": "retrieval.jsonl",
        "url": "https://ceevaldata.blob.core.windows.net/examples/retrieval.jsonl",
        "type": "dataset",
    },
    "faithfulness": {
        "filename": "faithfulness.jsonl",
        "url": "https://ceevaldata.blob.core.windows.net/examples/faithfulness.jsonl",
        "type": "dataset",
    },
    "graham_essays/small/txt": {
        "filename": "graham_essays/208_219_graham_essays.zip",
        "url": "https://ceevaldata.blob.core.windows.net/examples/graham_essays/208_219_graham_essays.zip",
        "type": "zip",
    },
    "graham_essays/small/dataset": {
        "dirname": "graham_essays",
        "url": "https://ceevaldata.blob.core.windows.net/examples/graham_essays/dataset.jsonl",
        "manifest": "https://ceevaldata.blob.core.windows.net/examples/graham_essays/manifest.yaml",
        "type": "dataset",
    },
    "graham_essays/small/results": {
        "filename": "graham_essays/simple_rag_results.json",
        "url": "https://ceevaldata.blob.core.windows.net/examples/graham_essays/simple_rag_results.json",
        "type": "json",
    },
}


def _ensure_destination_filename(filename: Path) -> None:
    if filename.is_dir():
        filename.mkdir(parents=True, exist_ok=True)
    else:
        parent = filename.parent
        parent.mkdir(parents=True, exist_ok=True)


def _download_file(url, destination_filename, force_download=False):
    if not force_download and destination_filename.exists():
        return destination_filename
    _ensure_destination_filename(destination_filename)
    response = requests.get(url)
    if response.status_code == 200:
        content = response.content
        with open(destination_filename, "wb") as file:
            file.write(content)
        return destination_filename
    else:
        raise RuntimeError(f"Could not download {url.split('/')[-1]}")


def _download_and_extract_zip(
    url, destination_dir, force_download=False
) -> Path:
    if (
        not force_download
        and destination_dir.exists()
        and any(destination_dir.iterdir())
    ):
        return destination_dir
    _ensure_destination_filename(destination_dir)
    with requests.get(url, stream=True) as response:
        if response.status_code == 200:
            with ZipFile(io.BytesIO(response.content)) as zip_file:
                zip_file.extractall(destination_dir)
            return Path(destination_dir)
        else:
            raise RuntimeError(f"Could not download {url.split('/')[-1]}")


def example_data_downloader(
    resource: str,
    destination_dir: Path = Path("data"),
    force_download: bool = False,
) -> Union[Path, Dict, Dataset]:  # type: ignore
    if resource not in _DATA_RESOURCES:
        raise ValueError(f"Resource {resource} not found")
    destination_dir.mkdir(parents=True, exist_ok=True)
    res = _DATA_RESOURCES[resource]
    telemetry.log_event("data_downloader", {"resource": resource})
    if res["type"] == "dataset":
        if "dirname" in res:
            out_dir = destination_dir / res["dirname"]
            fname = "dataset.jsonl"
        else:
            out_dir = destination_dir
            fname = res["filename"]
        file = _download_file(
            res["url"], out_dir / fname, force_download=force_download
        )
        if res.get("manifest"):
            manifest_fname = out_dir / "manifest.yaml"
            manifest = _download_file(
                res["manifest"], manifest_fname, force_download=force_download
            )
        else:
            manifest = None
        return Dataset(out_dir / fname, manifest)
    elif res["type"] == "zip":
        out_dir = destination_dir / resource
        return _download_and_extract_zip(
            res["url"], out_dir, force_download=force_download
        )
    elif res["type"] == "json":
        out_fname = destination_dir / res["filename"]
        file = _download_file(
            res["url"], out_fname, force_download=force_download
        )
        with open(file, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown resource type: {res['type']}")
