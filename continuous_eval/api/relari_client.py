import json
import os
from typing import Optional

import requests
from dotenv import load_dotenv

from continuous_eval.eval.manager import eval_manager

load_dotenv()


class RelariClient:
    def __init__(self, api_key: Optional[str] = None, url: str = "https://api.relari.ai/api/v1/"):
        self._api_url = url
        if api_key is None:
            self.api_key = os.getenv("RELARI_API_KEY")
        else:
            self.api_key = api_key
        if self.api_key is None:
            raise ValueError("Please set the environment variable RELARI_API_KEY or pass it as an argument.")

        self._headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
        self.valid = self._validate()

    def _validate(self):
        try:
            response = requests.get(self._api_url + "secure/auth", headers=self._headers, timeout=10)
        except requests.exceptions.Timeout:
            exit("Request timed out while trying to validate API key")
        if response.status_code != 200:
            return False
        return True

    def save(self):
        if eval_manager.is_running():
            raise ValueError("Cannot save while evaluation is running")
        evaluation_results = eval_manager.evaluation.results
        try:
            dataset = eval_manager.dataset.data
        except ValueError:
            raise ValueError("Dataset not set")

        payload = {
            "dataset": dataset,
            "results": evaluation_results,
            "metadata": eval_manager.metadata,
        }
        response = requests.post(
            self._api_url + "store",
            headers=self._headers,
            data=json.dumps(payload),
        )
        if response.status_code != 201:
            raise Exception("Failed to save evaluation results")
