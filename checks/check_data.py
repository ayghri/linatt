from datasets import load_from_disk
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, logging as hf_logging


ds = load_from_disk("/buckets/datasets/pajama_shards/shard_0/")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
print(ds[0])
dl = DataLoader(ds, batch_size=16)


print(tokenizer.decode(ds[0]["input_ids"]))
for d in dl:
    # print(d["input_ids"].shape)
    print(d["input_ids"])
    break
