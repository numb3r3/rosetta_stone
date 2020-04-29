import os
from abc import ABCMeta, abstractmethod

import torch


class PretrainedModel(metaclass=ABCMeta):
    @abstractmethod
    def init_from_checkpoint(self, checkpoint_file: str):
        """
        """

    def freeze(self, layers):
        """ To be implemented"""
        raise NotImplementedError()

    def _save_config(self, save_dir):
        save_filename = os.path.join(save_dir, "config.json")
        with open(save_filename, "w") as file:
            setattr(self.model.config, "name", self.__class__.__name__)
            setattr(self.model.config, "language", self.language)
            string = self.model.config.to_json_string()
            file.write(string)

    def save(self, save_dir):
        """
        Save the model state_dict and its config file so that it can be loaded again.
        :param save_dir: The directory in which the model should be saved.
        :type save_dir: str
        """
        # Save Weights
        save_name = os.path.join(save_dir, "model.bin")

        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Only save the model it-self
        torch.save(model_to_save.state_dict(), save_name)
        self.save_config(save_dir)
