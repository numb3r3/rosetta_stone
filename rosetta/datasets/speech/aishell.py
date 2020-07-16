import json
import os
from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from ...core.dataio import BaseDataIO
from ...data.tokenization import Tokenizer
from ...utils import io
from ...utils.distribute import get_global_rank
from .audio import create_audio_transform


# Batch size will be halfed if the longest wavefile surpasses threshold
HALF_BATCHSIZE_AUDIO_LEN = 800
# Note: Bucketing may cause random sampling to be biased (less sampled for those length > HALF_BATCHSIZE_AUDIO_LEN )
HALF_BATCHSIZE_TEXT_LEN = 150


class AudioDataset(Dataset):
    def __init__(self, data_path: str, **kwargs):
        super().__init__()
        self._data_path = data_path
        self._manifests = []
        with open(data_path, "r", encoding="utf-8") as fin:
            for line in fin:
                manifest = json.loads(line)
                self._manifests.append(manifest)

    def __getitem__(self, index):
        return self._manifests[index]

    def __len__(self):
        return len(self._manifests)


class AiShellDataIO(BaseDataIO):
    def __init__(self, tokenizer_name_or_path: str = "bert-base-chinese", **kwargs):
        self.cache_path = kwargs.get("cache_path", ".cache/")

        self.data_path = kwargs.get(
            "data_path", os.path.join(self.cache_path, "data_aishell")
        )
        self.prepare(
            url="http://www.openslr.org/resources/33/data_aishell.tgz",
            md5sum="2f494334227864a8a8fec932999db9d8",
        )

        # if get_global_rank() <= 0:
        #     self.create_manifest()

        self.tokenizer = Tokenizer.load(tokenizer_name_or_path, do_lower_case=True)
        self.tokenizer.eos_token = "<EOS>"
        self.tokenizer.eos_token_idx = 1

        # self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name_or_path, do_lower_case=True)
        self.audio_transform, _ = create_audio_transform(kwargs["audio_config"])

    def create_manifest(self):
        manifest_path_prefix = os.path.join(self.data_path, "manifest")
        print("Creating manifest %s ..." % manifest_path_prefix)
        json_lines = []
        transcript_path = os.path.join(
            self.data_path, "transcript", "aishell_transcript_v0.8.txt"
        )
        transcript_dict = {}
        with open(transcript_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line == "":
                    continue
                audio_id, text = line.split(" ", 1)
                # remove withespace
                text = "".join(text.split())
                transcript_dict[audio_id] = text

        data_types = ["train", "dev", "test"]
        for dtype in data_types:
            del json_lines[:]
            audio_dir = os.path.join(self.data_path, "wav", dtype)
            for subfolder, _, filelist in sorted(os.walk(audio_dir)):
                for fname in filelist:
                    audio_path = os.path.join(subfolder, fname)
                    audio_id = fname[:-4]

                    # if no transcription for audio then skipped
                    if audio_id not in transcript_dict:
                        continue

                    # audio_data, samplerate = soundfile.read(audio_path)
                    # duration = float(len(audio_data) / samplerate)
                    text = transcript_dict[audio_id]
                    json_lines.append(
                        json.dumps(
                            {
                                "audio_filepath": audio_path,
                                # 'duration': duration,
                                "text": text,
                            },
                            ensure_ascii=False,
                        )
                    )
            manifest_path = manifest_path_prefix + "." + dtype
            with open(manifest_path, "w", encoding="utf-8") as fout:
                for line in json_lines:
                    fout.write(line + "\n")

    def prepare(self, url: str, md5sum: str, **kwargs):
        """Download, unpack and create manifest file."""
        if not os.path.exists(self.data_path):
            filepath = io.download(url, md5sum, self.data_path)
            io.unpack(filepath, self.data_path)
            # unpack all audio tar files
            audio_dir = os.path.join(self.data_path, "wav")
            for subfolder, _, filelist in sorted(os.walk(audio_dir)):
                for ftar in filelist:
                    io.unpack(os.path.join(subfolder, ftar), subfolder, True)
        else:
            print(
                "Skip downloading and unpacking. Data already exists in %s."
                % self.data_path
            )

    def create_dataset(
        self, data_path: str, mode: str = "train", download: bool = True, **kwargs
    ):
        # assert not download, "Download dataset by yourself!"
        assert download or os.path.exists(self.data_path), "cache path does not exist"

        is_train = mode == "train"

        dtype = "train" if mode == "train" else "dev"

        data_path = os.path.join(self.data_path, "manifest.%s" % dtype)

        dt = AudioDataset(data_path)

        return dt

    def collate_fn(
        self, batch, tensor_names=None, mode: str = "train", **kwargs
    ) -> Dict[str, torch.Tensor]:

        # Read batch
        audio_feat, audio_len, lines = [], [], []
        with torch.no_grad():
            for example in batch:
                feat = self.audio_transform(example["audio_filepath"])
                audio_feat.append(feat)
                audio_len.append(len(feat))
                lines.append(example["text"])

        batch_encoding = self.tokenizer.batch_encode_plus(
            lines, add_special_tokens=False, max_length=128, return_lengths=True
        )
        token_ids = [
            torch.LongTensor(x + [self.tokenizer.eos_token_idx])
            for x in batch_encoding["input_ids"]
        ]
        token_len = [l + 1 for l in batch_encoding["length"]]

        teacher_ids = [
            torch.LongTensor([self.tokenizer.pad_token_id] + x)
            for x in batch_encoding["input_ids"]
        ]

        # Zero-padding
        audio_feat = pad_sequence(audio_feat, batch_first=True)
        audio_len = torch.LongTensor(audio_len)
        token_ids = pad_sequence(token_ids, batch_first=True)
        token_len = torch.LongTensor(token_len)
        teacher_ids = pad_sequence(teacher_ids, batch_first=True)
        return (audio_feat, audio_len, token_ids, token_len, teacher_ids)
