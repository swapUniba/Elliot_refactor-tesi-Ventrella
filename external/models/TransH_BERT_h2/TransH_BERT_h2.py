import pandas as pd

from elliot.evaluation.evaluator import Evaluator
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.folder import build_model_folder


class TransH_BERT_h2(RecMixin, BaseRecommenderModel):

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
        return "TransH+BERT-c2"

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
        topK = self._config.top_k
        ml = 'data/ventrella_experiment/testing/Movielens/[Hybrid2] TransH + BERT/TransH_lastLayerNoStopw/top_' + str(topK) + '_predictions_1.tsv'
        db = 'data/ventrella_experiment/testing/DbBook/[Hybrid2] TransE-TransH + BERT/TransH_lastLayerNoStopw/top_' + str(topK) + '_predictions_1.tsv'
        column_names = ['userId', 'itemId', 'rating', 'timestamp']
        if "movielens" in self._config.data_config.test_path:
            train_dataframe = pd.read_csv(ml, sep="\t", header=None, names=column_names)
        else:
            train_dataframe = pd.read_csv(db, sep="\t", header=None, names=column_names)
        users = list(train_dataframe['userId'].unique())
        ratings = {}
        for u in users:
            sel_ = train_dataframe[train_dataframe['userId'] == u]
            ratings[u] = dict(zip(sel_['itemId'], sel_['rating']))
        return ratings
