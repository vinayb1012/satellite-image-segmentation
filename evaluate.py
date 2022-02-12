import argparse

from data import SatelliteData
from unet_training import Trainer, add_arguments


def run(args):
    val_dataset = SatelliteData('data/data.csv', train=False, use_channels=args.channels)
    trainer = Trainer(val_dataset, val_dataset, args)
    trainer.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    run(args)
