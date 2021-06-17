import pandas as pd

from elliot.evaluation.evaluator import Evaluator
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.folder import build_model_folder


class NeuMF(RecMixin, BaseRecommenderModel):

    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a Fictional recommender.
        :param data: data loader object
        :param path_output_rec_result: path to the directory rec. results
        :param path_output_rec_weight: path to the directory rec. model parameters
        :param args: parameters
        """
        super().__init__(data, config, params, *args, **kwargs)
        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self.evaluator = Evaluator(self._data, self._params)

        self._params.name = self.name

        build_model_folder(self._config.path_output_rec_weight, self.name)
        self._saving_filepath = f'{self._config.path_output_rec_weight}{self.name}/best-weights-{self.name}'

    @property
    def name(self):
        return "NeuMF"

    def train(self):
        recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
        result_dict = self.evaluator.eval(recs)
        self._results.append(result_dict)

    def get_recommendations(self, top_k):
        ratings = self.overwrite_train_dict()
        r = {}
        for u, i_s in ratings.items():
            l = []
            ui = set(i_s.keys())

            for item in ui:
                l.append((item, i_s[item]))
            r[u] = l
        return r

    def overwrite_train_dict(self):
        ml = 'data/ventrella_experiment/testing/NeuRec results/Movielens/NeuMF_16/top_5_predictions_1.tsv'
        db = 'data/ventrella_experiment/testing/NeuRec results/DBbook/NeuMF_16/top_5_predictions_1.tsv'
        column_names = ['userId', 'itemId', 'rating', 'timestamp']
        train_dataframe = pd.read_csv(ml, sep="\t", header=None, names=column_names)
        users = list(train_dataframe['userId'].unique())
        ratings = {}
        for u in users:
            sel_ = train_dataframe[train_dataframe['userId'] == u]
            ratings[u] = dict(zip(sel_['itemId'], sel_['rating']))
        return ratings
