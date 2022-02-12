import argparse

from data import SatelliteData
from unet_training import Trainer, add_arguments


def run(args):
    dataset = SatelliteData('data/data.csv', train=True, use_channels=args.channels)
    val_dataset = SatelliteData('data/data.csv', train=False, use_channels=args.channels)
    trainer = Trainer(dataset, val_dataset, args)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    run(args)
