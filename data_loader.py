"""
@Time ： 2020/10/22 16:43
@Auth ： 胡怡凡
@File ：data_download.py
@IDE ：PyCharm
"""
from surprise import Dataset, Reader


class SurpriseDataLoader():
    def load_builtin(self, str):
        if str not in ['ml-100k', 'ml-1m', 'jester']:
            raise ValueError('{} not valid.'.format(str))
        else:
            return Dataset.load_builtin(str)

    def load_customdf(self, df, scale=(1, 5)):
        return Dataset.load_from_df(df, Reader(rating_scale=scale))
