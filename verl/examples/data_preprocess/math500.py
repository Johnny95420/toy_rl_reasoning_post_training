import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/math")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    data_source = "HuggingFaceH4/MATH-500"
    dataset = datasets.load_dataset(data_source, "default")
    dataset = dataset["test"]

    instruction_following = """Solve the following math problem step by step. The last line of your response should be of the form "Answer: \\boxed{{final_answer}}"(no quote) where final_answer is the answer to the problem.


Question:
{question}


Remember to put your answer on its own line after "Answer:" and with "\\boxed{{final_answer}}" format.
"""

    def process_fn(example, idx):
        ability = example.pop("subject")
        question_raw = example.pop("problem")
        question = instruction_following.format(question=question_raw)

        solution = example.pop("answer")
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": ability,
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "index": idx,
                "question": question_raw,
            },
        }
        return data

    format_dataset = dataset.map(function=process_fn, with_indices=True)
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    format_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
