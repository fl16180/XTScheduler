import os
import numpy as np
import torch


def safe_convert_entries(item):
    if isinstance(item, (float, int, str)):
        return item

    if isinstance(item, (torch.Tensor, np.ndarray)):
        if item.dim() > 1 or len(item) > 1:
            raise ValueError('Log entry must be a scalar.')
        return item.item()

    raise NotImplementedError('Entry type not supported')


def safe_make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def conjoin_tables(main_df, sub_df, join_cols):
    #     means = means.applymap(lambda x: str(x))
    # stds = stds.applymap(lambda x: str(x))

    def combine_fn(x):
        if x.name not in join_cols:
            return x
        else:
            return r'$' + x + r'_{\ ' + sub_df[x.name] + r'}$'

    df = main_df.apply(combine_fn)


    return df
