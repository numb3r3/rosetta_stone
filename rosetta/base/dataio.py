import os
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data.dataset import Dataset

class BaseDataIO():
    """

    """

    def __init__(
        self,
        file_path: str,
        params: Dict = dict(),
        mode: str = "train"
    ):
    self.file_path = file_path
    self.params = params
    self.mode = mode

    def collate_fn(self, batch, mode: str="train", params):
        """
        A custom collate function that formats the batch as a dictionary where the key is
        the name of the tensor and the value is the tensor itself
        """
        pass