from unidecode import unidecode  # type: ignore
import pandas as pd


class GenderPredictor:
    """
    A class used to predict the gender based on a given name using a provided dataset.

    Attributes
    ----------
    _data_frame : pandas.DataFrame
        A DataFrame containing names and their corresponding genders.

    Methods
    -------
    __init__(data_frame)
        Initializes the GenderPredictor with a dataset.
    normalize_name(name)
        Normalizes the given name by converting it to lowercase and removing accents.
    is_name_in_data(name)
        Checks if the normalized name exists in the dataset.
    predict(name)
        Predicts the gender of the given name.
        Returns 0 if the name is not in the dataset,
        1 if the name is predicted to be male,
        and 2 if the name is predicted to be female.
    """

    def __init__(self, data_frame_path: str) -> None:
        self._data_frame = pd.read_csv(data_frame_path)

    def normalize_name(self, name: str) -> str:
        """
        Normalizes the given name by converting it to lowercase and removing accents.

        Args:
            name (str): The name to normalize.

        Returns:
            str: The normalized name.
        """

        if name is None:
            name = ""

        return unidecode(name).lower()

    def is_name_in_data(self, name: str) -> bool:
        """
        Checks if the normalized name exists in the dataset
        Args:
            name (str): The name to check.

        Returns:
            bool: Returns True if the name exists in the dataset
        """

        result = self._data_frame[
            self._data_frame["Name"] == self.normalize_name(name)
        ].empty

        return result 

    def predict(self, name) -> int:
        """
                Predicts the gender of a given name.
        Args:
            name (str): The name to predict the gender for.
        Returns:
            int: Returns 0 if the name is not found in the data,
                 1 if the name is predicted to be male,
                 2 if the name is predicted to be female.
        """
        if self.is_name_in_data(name):
            return 0

        if self._data_frame[
            (self._data_frame["Name"] == self.normalize_name(name))
            & (self._data_frame["Gender"] == "M")
        ].empty:
            return 2

        return 1
