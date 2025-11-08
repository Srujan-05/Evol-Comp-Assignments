import numpy as np
import pandas as pd


def load_dataset_from_csv(file_path: str, train_ratio: float = None, test_ratio: float = None, train_samples: int = None, test_samples: int = None):
    """
    :param file_path: The file path of the dataset.
    :param train_ratio: a value between 0-1, either specify this or the test_ratio.
    :param test_ratio: a value between 0-1, either specify this or the train_ratio.
    :param train_samples: the number of samples is the training data, either specify this or test_samples.
    :param test_samples: the number of samples is the testing data, either specify this or training_samples.
    :return:
    """
    """
    Load dataset from CSV file and split into train/test based on user input.
    Priority:
      1. If user specifies sample counts → use them.
      2. Else if user specifies percentages → use them.
      3. Else → default split 75% train / 25% test.
    """
    df = pd.read_csv(file_path)
    data = np.array(df)
    total_samples = len(data)

    i = 0
    while i < data.shape[1]:
        if np.isnan(data[:, i]).all():
            data = np.delete(data, i, axis=1)
        else:
            i += 1

    if train_samples or test_samples:
        if train_samples:
            test_samples = total_samples - train_samples
        else:
            train_samples = total_samples - train_samples

        train_data = data[:train_samples]
        test_data = data[train_samples:train_samples + test_samples]

    elif train_ratio or test_ratio:
        if train_ratio:
            test_ratio = 1 - train_ratio
        else:
            train_ratio = 1 - test_ratio
        train_count = int(train_ratio * total_samples)
        train_data = data[:train_count]
        test_data = data[train_count:]

    else:
        train_count = int(0.75 * total_samples)
        train_data = data[:train_count]
        test_data = data[train_count:]

    return train_data, test_data

