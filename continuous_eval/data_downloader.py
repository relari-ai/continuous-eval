import io
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import requests

from continuous_eval.utils.telemetry import telemetry

EXAMPLES_DATA_URL = "https://ceevaldata.blob.core.windows.net/examples/"

_DATA_RESOURCES = {
    "correctness": {"filename": "correctness.jsonl", "type": "dataset"},
    "retrieval": {"filename": "retrieval.jsonl", "type": "dataset"},
    "faithfulness": {"filename": "faithfulness.jsonl", "type": "dataset"},
    "graham_essays/small/txt": {
        "filename": "graham_essays/208_219_graham_essays.zip",
        "type": "txt",
    },
    "graham_essays/small/chromadb": {
        "filename": "graham_essays/208_219_chroma_db.zip",
        "type": "chromadb",
    },
}


def _download_file(url, destination_filename, force_download=False):
    if not force_download and destination_filename.exists():
        return destination_filename
    response = requests.get(url)
    if response.status_code == 200:
        content = response.content
        with open(destination_filename, "wb") as file:
            file.write(content)
        return destination_filename
    else:
        raise RuntimeError(f"Could not download {url.split('/')[-1]}")


def _download_and_extract_zip(url, destination_dir, force_download=False) -> Path:
    if not force_download and destination_dir.exists() and any(destination_dir.iterdir()):
        return destination_dir

    with requests.get(url, stream=True) as response:
        if response.status_code == 200:
            with ZipFile(io.BytesIO(response.content)) as zip_file:
                zip_file.extractall(destination_dir)
            return Path(destination_dir)
        else:
            raise RuntimeError(f"Could not download {url.split('/')[-1]}")


def example_data_downloader(
    resource: str, destination_dir: Path = Path("data"), force_download: bool = False
) -> Union[Path, "Chroma"]:  # type: ignore
    assert resource in _DATA_RESOURCES, f"Resource {resource} not found"
    destination_dir.mkdir(parents=True, exist_ok=True)
    res = _DATA_RESOURCES[resource]
    telemetry.log_event("data_downloader", resource)
    if res["type"] == "dataset":
        out_fname = destination_dir / res["filename"]
        file = _download_file(
            EXAMPLES_DATA_URL + res["filename"],
            out_fname,
            force_download=force_download,
        )
        return file
    elif res["type"] == "txt":
        out_dir = destination_dir / resource
        return _download_and_extract_zip(EXAMPLES_DATA_URL + res["filename"], out_dir, force_download=force_download)
    elif res["type"] == "chromadb":
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        out_dir = destination_dir / resource
        _download_and_extract_zip(EXAMPLES_DATA_URL + res["filename"], out_dir, force_download=force_download)
        return Chroma(
            persist_directory=str(out_dir),
            embedding_function=OpenAIEmbeddings(),
        )
