


def bold_extreme_values(data, data_max=-1):
    if data == data_max:
        # return "\\bfseries %s" % data
        return "\\mathbf{%s}" % data
    return data


def format_df_table(df, bold_cols, bold_values='max'):
    for col in bold_cols:
        df[col] = df[col].apply(
            lambda x: bold_extreme_values(x, data_max=df[col].max())
        )