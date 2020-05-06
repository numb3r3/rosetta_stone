import csv


class DataProcessor:
    """Base class for data converters data sets."""

    def get_examples(self, file_names):
        """Gets a collection of `InputExample`."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_csv(cls, input_file: str, delimiter: str = "\t", quotechar: str = None):
        """Reads a csv formated file."""
        with open(input_file, "r") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))
