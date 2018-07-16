from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler

from datasets import boxes_dataset


class BoxesDataLoader(DataLoader):
    """
    Images with boxes data loading
    """

    def __init__(self, config, name):
        super(BoxesDataLoader, self).__init__(
            dataset=boxes_dataset.ImagesWithBoxesDataset(config=config, name=name),
            batch_size=config['data_loader']['batch_size_%s' % name],
            drop_last=config['data_loader']['drop_last']
        )
        self.batch_sampler = BatchSampler(
            SequentialSampler(self.dataset),
            batch_size=self.batch_size,
            drop_last=self.drop_last
        )

        self.config = config
