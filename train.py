import os
import json
import logging
import argparse
import torch

from inference import inference
from model.model import *
from model.loss import *
from model.metric import *
from data_loader import BoxesDataLoader
from trainer import Trainer
from logger import Logger

logging.basicConfig(level=logging.INFO, format='')


def main(config, resume):
    train_logger = Logger()

    print('Create train loader')
    # train_data_loader = BoxesDataLoader(config, name='train')

    model = NatashaDetection(config)
    model.summary()

    loss = eval(config['loss'])

    print('Create trainer')
    # trainer = Trainer(model, loss,
    #                   resume=resume,
    #                   config=config,
    #                   data_loader=train_data_loader,
    #                   train_logger=train_logger)

    print('Start training')
    # trainer.train()

    print('Create test loader')
    test_data_loader = BoxesDataLoader(config, name='test')
    model.eval()
    print('Do inference')
    inference(test_data_loader, model)


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        config = torch.load(args.resume)['config']
    elif args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    assert config is not None

    main(config, args.resume)
