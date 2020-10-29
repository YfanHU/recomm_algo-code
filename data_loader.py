"""
@Time ： 2020/10/22 16:43
@Auth ： 胡怡凡
@File ：data_download.py
@IDE ：PyCharm
"""
# this file is used to formulate the famous movie-lens data for testing our models

from surprise import Dataset, Reader
import pandas as pd


class SurpriseDataLoader():
    @staticmethod
    def load_builtin(str):
        if str not in ['ml-100k', 'ml-1m', 'jester']:
            raise ValueError('{} not valid.'.format(str))
        else:
            return Dataset.load_builtin(str)

    def load_customdf(self, df, scale=(1, 5)):
        return Dataset.load_from_df(df, Reader(rating_scale=scale))


class SklearnDataLoader():
    @staticmethod
    def load_builtin(str, classification=False, rating_only=False):
        if str not in ['ml-100k', 'ml-1m', 'jester']:
            raise ValueError('{} not valid.'.format(str))
        else:
            if str == 'ml-100k':
                df_rating_path = './data/ml-100k/ml-100k/u.data'
                df_user_info_path = './data/ml-100k/ml-100k/u.user'
                df_movie_info_path = './data/ml-100k/ml-100k/u.item'
                df_rating = pd.read_csv(df_rating_path, sep='\t',
                                        names=['uid', 'iid', 'rating', 'timestamp'])
                if rating_only:
                    return df_rating[['uid','iid','rating']]
                df_user_info = pd.read_csv(df_user_info_path, sep='|',
                                           names='uid | age | gender | occupation | zip code'.split(' | '))
                df_movie_info = pd.read_csv(df_movie_info_path, sep='|', encoding='latin-1',
                                            names="iid | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western"
                                            .split(' | '))
                df_rating = pd.merge(df_rating, df_user_info, how='left', on='uid')
                df_rating = pd.merge(df_rating, df_movie_info, how='left', on='iid')
                rating = df_rating[['rating']]
                if classification:
                    rating['rating'] = rating['rating'].squeeze().apply(lambda x: 0 if x < 3 else 1)
                return df_rating.drop(['rating'], axis=1), rating
            else:
                pass
