# %%
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Preprocess the DAPO-Math-17k dataset to multiturn format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/dapo_math17k")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    data_path = "BytedTsinghua-SIA/DAPO-Math-17k"
    dataset = datasets.load_dataset(data_path, "default")["train"]
    total_size = len(dataset)
    train_indices = range(0, total_size - 100)
    val_indices = range(total_size - 100, total_size)

    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_dir, "val.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
