from collections import defaultdict
from itertools import islice
import random
from pathlib import Path
import glob
from typing import Iterator, Tuple
import torch
import numpy as np
from more_itertools import chunked


class PreTokDataset(torch.utils.data.IterableDataset):
    def __init__(self, batch_size_for_max_seq_len: int, split: str, max_seq_len: int, max_shards: int=None, chunk_ratios:list[float] = None, increase_chunk_ratio_every_n_shards:int = None):
        """
        Keyword arguments: 
        chunk_ratios: list[float] -- if set, batching is done with variable chunk sizes. The list of floats are used to calculate these variable chunk sizes by multiplying with the `max_seq_len`. Hence, the maximum ratio value (the max value for an element in this list of floats) can be `1.0`. If `chunk_ratios` is specified, we expect to do variable size chunks sometimes less than the `max_seq_len` - this means we can pack more into a single batch. We calculate how much more based on the chunk size (directly proportional). If the chunk size is half of the `max_seq_len` then we can pack twice the number of batches.
        increase_chunk_ratio_every_n_shards: int -- curriculum learning behaviour. If set, values from the chunk ratio are sorted and smallest value is used first. After `n` shards (this variable sets `n`) of sampling, the next smallest value is included in the possible chunk sizes.. and so on. This variable `n` can be higher than the number of shards - this just means we cycle through all shards before we increase the chunk sizes (assuming the training iters are set accordingly to a high value to cycle through all shards and do more)
        batch_size_for_max_seq_len: int -- for chunks of size `max_seq_len`, this is the batch size. But if `chunk_ratios` are specified, we may adjust the batch size based on the chunk size (we'll pack more for smaller chunk sizes)
        max_shards - useful for quick debugging. Simply stop yielding after we reach `max_shards` number of shards
        """
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.max_shards = max_shards
        self.chunk_ratios = chunk_ratios
        self.increase_chunk_ratio_every_n_shards = increase_chunk_ratio_every_n_shards
        self.batch_size_for_max_seq_len = batch_size_for_max_seq_len

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        bin_dir = Path("data/TinyStories_all_data")
        shard_filenames = sorted(glob.glob(str(bin_dir / "*.bin")))
        shard_filenames = (
            shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        )

        rng = random.Random(42)
        num_shards_sampled = 0
        while True:
            rng.shuffle(shard_filenames)
            for shard in islice(shard_filenames, self.max_shards):
                data = np.memmap(shard, dtype=np.uint16, mode="r")
                # `chunk_ratios` is set - we can calculate a variety of chunk sizes based on this ratio
                if self.chunk_ratios is not None:
                    # curriculum set as per increase_chunk_ratio_every_n_shards
                    # at the start num_shards_sampled is 0 leading to num_chunk_ratios_to_use = 1
                    sorted_chunk_ratios = np.sort(np.unique(self.chunk_ratios.round(decimals=4)))
                    if self.increase_chunk_ratio_every_n_shards is not None:
                        num_chunk_ratios_to_use = num_shards_sampled//self.increase_chunk_ratio_every_n_shards + 1
                    else:
                        num_chunk_ratios_to_use = len(sorted_chunk_ratios)
                    chunk_sizes = (self.max_seq_len * sorted_chunk_ratios[:num_chunk_ratios_to_use]).astype(int)
                else:
                # no `chunk_ratios` - we only use one chunk size i.e. the `max_seq_len`
                    chunk_sizes = [self.max_seq_len]
                chunk_start = 0
                data_len = len(data)
                chunks = defaultdict(lambda: [])
                MAX_CHUNKS_FROM_EACH_SHARD = 500
                num_chunks_from_shard = 0
                while chunk_start < data_len:
                    # sample the `chunk_size_selected` every time based on the available `chunk_sizes`
                    # special case happens when we are at the end of the `data` array -- possible chunk_sizes are a subset of the original selection - so we filter the possible chunk sizes to account for it
                    # when chunk_start = 0 and data_len = 10, maximum chunk_size possible is 10 (not inclusive end index)
                    possible_chunk_sizes = [c for c in chunk_sizes if c <= (data_len - chunk_start) ]
                    if possible_chunk_sizes != []:
                        chunk_size_selected = int(np.random.choice(possible_chunk_sizes))
                    else:
                        # instead of throwing away the last chunk in every shard, let's just use it as its own batch
                        # not efficient but only happens once every shard
                        chunk_size_selected = data_len - chunk_start
                    chunk_end = chunk_size_selected + chunk_start
                    chunk = torch.from_numpy(data[chunk_start:chunk_end].astype(np.int64))
                    chunk_start = chunk_end
                    # store the chunks in a dictionary that segregates different chunk sizes
                    chunks[chunk_size_selected].append(chunk)
                    num_chunks_from_shard += 1
                    if num_chunks_from_shard >= MAX_CHUNKS_FROM_EACH_SHARD:
                        break
                
                # iterate through the different chunk sizes and the available chunks of that size
                for chunk_size, chunks_of_same_size in chunks.items():
                    # how much can we pack -- depends on the batch size specified for the maximum sequence length. If the chunk size is less than `max_seq_len`, we can pack a lot more
                    num_chunks_of_same_size = self.max_seq_len // chunk_size * self.batch_size_for_max_seq_len
                    # shuffle the chunks to randomize
                    rng.shuffle(chunks_of_same_size)
                    # `chunked` moves the window by `num_chunks_of_same_size` and gives us a list
                    # if we run out of chunks, we use the leftover in one batch -- leads to less packing but shouldn't happen too frequently, only once per chunk_size per shard
                    # TODO: maybe pack similar `chunk_size` batches across shards?
                    for chunks_in_one_batch in chunked(chunks_of_same_size, num_chunks_of_same_size):
                        # stack the inputs and outputs and return the stacked aka. batched output
                        x_to_stack = []
                        y_to_stack = [] 
                        for chunk in chunks_in_one_batch:
                            # all of these chunks will be of the same size `chunk_size`
                            x = chunk[:-1]
                            y = chunk[1:]
                            x_to_stack.append(x)
                            y_to_stack.append(y)
                        # stack them all
                        X = torch.stack(x_to_stack)
                        Y = torch.stack(y_to_stack)
                        # since this is already stacked, the caller can simply use this as a single batch
                        yield X, Y
                num_shards_sampled += 1


class Task:
    @staticmethod
    def iter_batches(
        batch_size: int, device: str, num_workers: int = 0, **dataset_kwargs
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        ds = PreTokDataset(batch_size_for_max_seq_len=batch_size, **dataset_kwargs)
        # no batch_size = no batching. Because the `PreTokDataset` already batches data with variable sized chunks
        # based on the chunk size, the batch size is modified
        # `batch_size` input to this method assumes constant size batches with `max_seq_len`
        # instead we reduce `max_seq_len` for some batches and appropriately increase batch size
        # we could just directly use the dataset `ds` to iterate but using the dataloader allows us to set number of workers
        # TODO: does `num_workers` have any use now since we are sequentially processing the shards and chunks? Was there ever a point in having this?
        dl = torch.utils.data.DataLoader(
            ds, batch_size=None, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            yield x, y
