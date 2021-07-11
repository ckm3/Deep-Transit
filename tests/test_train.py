import pytest
import deep_transit as dt


def test_train():
    dt.config.DATASET = 'tests/Data'
    dt.config.IMG_DIR = dt.config.DATASET + "/transit-images/"
    dt.config.LABEL_DIR = dt.config.DATASET + "/transit-labels/"

    dt.config.BATCH_SIZE = 2
    dt.config.NUM_EPOCHS = 1

    dt.train()

