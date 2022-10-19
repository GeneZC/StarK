# -*- coding: utf-8 -*-

"""
DataReader -> reading RawData from tsv, json, etc., in general form of Examples. [task-specific]
DataPipeline -> converting Examples to Examples of specific forms, collating Examples as Batches. [model-specific]
"""


from data.readers import (
    SST2Reader,
    MRPCReader,
    STSBReader,
    QQPReader,
    MNLIReader,
    MNLIMMReader,
    QNLIReader,
    RTEReader,
)
from data.pipelines import (
    CombinedDataPipeline,
)


READER_CLASS = {
    "sst2": SST2Reader,
    "mrpc": MRPCReader,
    "stsb": STSBReader,
    "qqp": QQPReader,
    "mnli": MNLIReader,
    "mnlimm": MNLIMMReader,
    "qnli": QNLIReader,
    "rte": RTEReader,
}

def get_reader_class(task_name):
    return READER_CLASS[task_name]


PIPELINE_CLASS = {
    "combined": CombinedDataPipeline,
}

def get_pipeline_class(data_type):
    return PIPELINE_CLASS[data_type]


import torch
from torch.utils.data import IterableDataset


class Dataset(IterableDataset):
    def __init__(self, data, shuffle=True):
        super().__init__()
        self.data = data
        self.shuffle = shuffle
        self.num_instances = len(self.data)

    def __len__(self):
        return self.num_instances

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            for idx in torch.randperm(self.num_instances, generator=generator).tolist():
                yield self.data[idx]
        else:
            for idx in range(self.num_instances):
                yield self.data[idx]

class DistributedDataset(IterableDataset):
    def __init__(self, data, num_replicas=None, rank=None, shuffle=True):
        super().__init__()
        self.data = data
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        # Do ceiling to make the data evenly divisible among devices.
        self.num_instances = math.ceil(len(self.data) / self.num_replicas)
        self.total_num_instances = self.num_instances * self.num_replicas

    def __len__(self):
        return self.num_instances

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            indices = torch.randperm(self.num_instances, generator=generator).tolist()
        else:
            indices = list(range(self.num_instances))

        num_padding_instances = self.total_num_instances - len(indices)
        # Is the logic necessary?
        if num_padding_instances <= len(indices):
            indices += indices[:num_padding_instances]
        else:
            indices += (indices * math.ceil(num_padding_instances / len(indices)))[:num_padding_instances]

        assert len(indices) == self.num_total_instances

        # Subsample.
        indices = indices[self.rank:self.num_total_instances:self.num_replicas]
        assert len(indices) == self.num_instances

        for idx in indices:
            yield self.data[idx]