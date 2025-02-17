from abc import ABC
import os
import pickle
import yaml

import torch


class BaseExperiment(ABC):
    def __init__(self, args, log_dir, external_log_dir=None):
        if external_log_dir is not None:
            self.log_dir = external_log_dir
        else:
            self._init_base_stuff(args, log_dir)

    def run(self):
        self.trainer.train()

    def _init_base_stuff(self, args, log_dir):
        args.pop('self')
        args.pop('__class__')

        name = ""
        for var_name, value in args.items():
            if var_name != "name" and var_name != "self" and not var_name.startswith("__"):
                name += var_name + "_" + str(value).replace(".", "-") + "_"
        if name != "":
            name = name[:-1]
        else:
            name = "experiment"

        self.log_dir = os.path.join(log_dir, name)
        os.makedirs(self.log_dir, exist_ok=True)
        args_yaml_path = os.path.join(self.log_dir, "readable_args.yaml")
        if not os.path.isfile(args_yaml_path):
            # save both in pickle and yaml, the latter is for readability
            with open(os.path.join(self.log_dir, 'args.pkl'), 'wb') as pick:
                pickle.dump(args, pick)

            for k, v in args.items():
                args[k] = str(v)
            with open(args_yaml_path, 'w') as outfile:
                yaml.dump(args, outfile, default_flow_style=False)

    @classmethod
    def reconstruct(cls, log_dir, restore_ckpt=False):
        with open(os.path.join(log_dir, 'args.pkl'), 'rb') as pick:
            args = pickle.load(pick)
            print(args)
        args["external_log_dir"] = log_dir
        experiment = cls(**args)
        if restore_ckpt:
            experiment.trainer.restore_model_from_checkpoint(device='cuda' if torch.cuda.is_available() else 'cpu')
        return experiment