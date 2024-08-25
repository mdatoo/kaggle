"""Convert tensorflow records to png and json."""

from glob import glob
from os import makedirs, path, remove
from shutil import rmtree, unpack_archive

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

TRAIN_RECORD_SCHEMA = {
    "id": tf.io.FixedLenFeature([], tf.string),
    "class": tf.io.FixedLenFeature([], tf.int64),
    "image": tf.io.FixedLenFeature([], tf.string),
}

TEST_RECORD_SCHEMA = {
    "id": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
}


def setup() -> None:
    """Create/Clean up directories."""
    rmtree("train", ignore_errors=True)
    rmtree("test", ignore_errors=True)
    makedirs("train")
    makedirs("test")


def extract() -> None:
    """Extract zip."""
    unpack_archive("tpu-getting-started.zip")


def decode_train() -> None:
    """Decode train records."""
    print("Decoding train records and saving images")
    train_labels = {}

    for subset in tqdm(["train", "val"]):
        for tf_record_dataset_path in tqdm(glob(path.join("tfrecords-jpeg-512x512", subset, "**"))):
            tf_record_dataset = tf.data.TFRecordDataset(tf_record_dataset_path)
            for tf_record in tf_record_dataset:
                record = tf.io.parse_single_example(tf_record, TRAIN_RECORD_SCHEMA)
                idx = record["id"].numpy().decode()
                label = record["class"].numpy()
                image = record["image"].numpy()

                train_labels[idx] = {"label": label}
                with open(path.join("train", f"{idx}.png"), "wb") as file:
                    file.write(image)

    print("Saving labels")
    train_labels_df = pd.DataFrame.from_dict(train_labels, orient="index")
    train_labels_df.index.name = "id"
    train_labels_df.to_csv("labels.csv")


def decode_test() -> None:
    """Decode test records."""
    print("Decoding test records and saving images")
    for tf_record_dataset_path in tqdm(glob("tfrecords-jpeg-512x512/test/**")):
        tf_record_dataset = tf.data.TFRecordDataset(tf_record_dataset_path)
        for tf_record in tf_record_dataset:
            record = tf.io.parse_single_example(tf_record, TEST_RECORD_SCHEMA)
            idx = record["id"].numpy().decode()
            image = record["image"].numpy()

            with open(path.join("test", f"{idx}.png"), "wb") as file:
                file.write(image)


def teardown() -> None:
    """Delete temporary files."""
    remove("tpu-getting-started.zip")
    remove("sample_submission.csv")

    rmtree("tfrecords-jpeg-192x192", ignore_errors=True)
    rmtree("tfrecords-jpeg-224x224", ignore_errors=True)
    rmtree("tfrecords-jpeg-331x331", ignore_errors=True)
    rmtree("tfrecords-jpeg-512x512", ignore_errors=True)


def run() -> None:
    """Convert tensorflow records to png and json."""
    setup()

    extract()

    decode_train()
    decode_test()

    teardown()


if __name__ == "__main__":
    run()
