from abc import ABCMeta, abstractmethod
import os

import torch


class PretrainedModel(metaclass=ABCMeta):
    # @abstractmethod
    # def init_from_checkpoint(self, checkpoint_file: str):
    #     """
    #     """

    def freeze(self, layers):
        """To be implemented."""
        raise NotImplementedError()

    def _save_config(self, save_dir):
        save_filename = os.path.join(save_dir, "config.json")
        with open(save_filename, "w") as file:
            setattr(self.model.config, "name", self.__class__.__name__)
            setattr(self.model.config, "language", self.language)
            string = self.model.config.to_json_string()
            file.write(string)

    def save(self, save_dir):
        """Save the model state_dict and its config file so that it can be
        loaded again.

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

    # pretrained_checkpoint_file = hparams.get("pretrained_bert_model", None)
    # if pretrained_checkpoint_file:
    #     self.init_from_checkpoint(pretrained_checkpoint_file)

    def init_from_checkpoint(self, checkpoint_file: str):
        """The following codes are borrowed from."""
        try:
            state_dict = torch.load(checkpoint_file, map_location="cpu")
        except Exception:
            raise OSError("Unable to load weights from pytorch checkpoint file. ")

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
        # so we need to apply the function recursively.
        def load(module: torch.nn.Module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = self
        has_prefix_module = any(
            s.startswith(self.__class__.base_model_prefix) for s in state_dict.keys()
        )
        if not hasattr(self, self.__class__.base_model_prefix) and has_prefix_module:
            start_prefix = self.__class__.base_model_prefix + "."
        if hasattr(self, self.__class__.base_model_prefix) and not has_prefix_module:
            model_to_load = getattr(self, self.__class__.base_model_prefix)

        load(model_to_load, prefix=start_prefix)

        if self.__class__.__name__ != model_to_load.__class__.__name__:
            base_model_state_dict = model_to_load.state_dict().keys()
            head_model_state_dict_without_base_prefix = [
                key.split(self.__class__.base_model_prefix + ".")[-1]
                for key in self.state_dict().keys()
            ]

            missing_keys.extend(
                head_model_state_dict_without_base_prefix - base_model_state_dict
            )

        if len(missing_keys) > 0:
            self.logger.info(
                "Weights of {} not initialized from pretrained model: {}".format(
                    self.__class__.__name__, missing_keys
                )
            )
        if len(unexpected_keys) > 0:
            self.logger.info(
                "Weights from pretrained model not used in {}: {}".format(
                    self.__class__.__name__, unexpected_keys
                )
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(error_msgs)
                )
            )

        self.tie_weights()  # make sure token embedding weights are still tied if needed
