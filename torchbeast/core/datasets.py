# Copyright urw7rs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import partial
import os
import PIL
from torchvision.datasets import VisionDataset

from torchvision.datasets.utils import download_file_from_google_drive
from torchvision.datasets.utils import check_integrity, verify_str_arg


class CelebAHQ(VisionDataset):
    base_folder = "CelebAMask-HQ"

    # yapf: disable
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("1badu11NqxGf6qM3PTTooQDJvQbejgbTv", "8a26c2c381f1cfb210d16b54bfd5259e", "CelebAMask-HQ.zip"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]
    # yapf: enable

    def __init__(
        self, root, split="train", target_type="attr", transform=None, download=False,
    ):
        import pandas

        super(CelebAHQ, self).__init__(root, transform=transform, target_transform=None)

        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split = split_map[
            verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))
        ]

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(
            fn("list_eval_partition.txt"),
            delim_whitespace=True,
            header=None,
            index_col=0,
        )
        index = pandas.read_csv(
            fn("CelebA-HQ-to-CelebA-mapping.txt"),
            delim_whitespace=True,
            header=0,
            usecols=["idx", "orig_idx"],
        )

        splits = index["orig_idx"].apply(lambda i: splits.iloc[i])
        index = index["idx"]

        mask = slice(None) if split is None else (splits[1] == split)

        self.filename = index[mask].apply(lambda s: str(s) + ".jpg").values

    def _check_integrity(self):
        for (_, md5, filename) in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            _, ext = os.path.splitext(filename)
            # Allow original archive to be deleted (zip and 7z)
            # Only need the extracted images
            if ext != ".zip" and not check_integrity(fpath, md5):
                return False

        # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "CelebA-HQ-img"))

    def download(self):
        import zipfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        try:
            os.mkdir(os.path.join(self.root, self.base_folder))
        except OSError as error:
            print(error)

        for (file_id, md5, filename) in self.file_list:
            download_file_from_google_drive(
                file_id, os.path.join(self.root, self.base_folder), filename, md5
            )

        with zipfile.ZipFile(
            os.path.join(self.root, self.base_folder, "CelebAMask-HQ.zip"), "r"
        ) as f:
            f.extractall(os.path.join(self.root, self.base_folder))

    def __getitem__(self, index):
        X = PIL.Image.open(
            os.path.join(
                self.root, self.base_folder, "CelebA-HQ-img", self.filename[index]
            )
        )

        if self.transform is not None:
            X = self.transform(X)

        return X

    def __len__(self):
        return len(self.filename)
