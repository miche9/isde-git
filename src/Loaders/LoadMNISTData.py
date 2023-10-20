import numpy as np
import pandas as pd


class CLoadMNISTData:
    def __init__(self, filename="https://github.com/unica-isde/isde/raw/master/data/mnist_data.csv", n_samples=None):
        self.filename = filename
        self.n_samples = n_samples

    @property
    def n_samples(self):
        return self._n_samples

    @n_samples.setter
    def n_samples(self, value):
        self._n_samples = value

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        if not isinstance(value, str):
            raise TypeError('value is type string')
        self._filename = value

    def load_mnist_data(self) -> tuple:
        """This function returns MNIST handwritten digits and labels as ndarrays (X, y)."""
        data = pd.read_csv(self.filename)
        data = np.array(data)  # cast pandas dataframe to numpy array
        if self.n_samples is not None:  # only returning the first n_samples
            data = data[:self.n_samples, :]  # I can specify the number of samples I want.
        y = data[:, 0]  # the first colon have the label (take all the raw, take the first colon)
        X = data[:, 1:] / 255.0  # the pixel value goes from 0 to 255, I'm normalizing it to the intervall 0-1
        # for something more complicate is better to rescale the range
        # the rescaling doesn't affect the color of the image
        return X, y