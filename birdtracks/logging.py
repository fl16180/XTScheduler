import os
import copy
from collections import OrderedDict
import torch
import pandas as pd
import numpy as np

from .utils import safe_convert_entries, safe_make_dir


DEFAULT_LOG_DIR = './results'


class BaseTracker:
    def __init__(self, name='log', log_dir=DEFAULT_LOG_DIR, arg_spec=None):
        self.name = name
        self.log_dir = log_dir
        self.arg_spec = arg_spec
        self._safe_make_dir(self.log_dir)

    def _safe_make_dir(self, path):
        safe_make_dir(path)

    def annotate(self, txt):
        with open(os.path.join(self.log_dir, 'info.txt') , 'w') as f:
            f.write(txt)

    def add(self, varname, quantity, disp=False):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError


class StepTracker(BaseTracker):
    def __init__(self, name='log', log_dir=DEFAULT_LOG_DIR, auto_export=True):
        super().__init__(name=name, log_dir=log_dir)
        self.auto_export = auto_export
        self.step_log = OrderedDict()
        self.global_log = {}

    def add(self, t, varname, quantity):
        to_export = False
        quantity = safe_convert_entries(quantity)
        if t not in self.step_log:
            self.step_log[t] = {}
            to_export = True 
        self.step_log[t][varname] = quantity

        if to_export and self.auto_export:
            self.export()

    def add_global(self, varname, quantity):
        quantity = safe_convert_entries(quantity)
        if varname in self.global_log:
            raise ValueError(f'Global value {varname} already exists')
        self.global_log[varname] = quantity
        
        if self.auto_export:
            self.export()

    def export(self):
        out_dict = copy.deepcopy(self.step_log)

        steps = self.step_log.keys()
        global_params = self.global_log.keys()

        step_params = []
        for t in steps:
            for param in self.step_log[t].keys():
                if param not in step_params:
                    step_params.append(param)

        if len(out_dict.keys()) == 0:
            out_dict[0] = {}

        for t in out_dict.keys():
            for globl in global_params:
                out_dict[t][globl] = self.global_log[globl]

        df = pd.DataFrame.from_dict(out_dict, orient='index')

        df = df[list(sorted(step_params)) + list(global_params)]
        df = df.reset_index().rename(columns={'index': 'step'})

        df.to_csv(
            os.path.join(self.log_dir, f'{self.name}.csv'),
            index=False
        )
