'''
========================================================================================================
Copyright 2017, by the California Institute of Technology. ALL RIGHTS RESERVED.
United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology
Transfer at the California Institute of Technology. This software may be subject to U.S. export control laws. By
accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the
responsibility to obtain export licenses, or other export authority as may be required before exporting such
information to foreign countries or providing access to foreign persons.
========================================================================================================
'''

# imports #
import datetime
import dill
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import logging
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import sys
import time
import uuid

from utils import _ElasticSearch

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

__author__ = 'Valentino Constantinou'
__version__ = '0.0.7'
__license__ = 'Apache License, Version 2.0'


# resources #
class Models(object):
    '''
    This class facilitates model training, testing, storage, and deployment, and defines the
    SKLearn model class which facilitates the use of scikit-learn models for making predictions.
    '''

    def __init__(self, server_config=None):
        if server_config:
            self.server_config = server_config
            self.connection_models = server_config.es_connection_models
            self.index = _ElasticSearch().Models(server_config)

    @staticmethod
    def _id_generator(model):
        if model.id is None:
            t = uuid.uuid4()
            model.id = t.hex

        return model.id

    def _check_path(self, path):
        if 'Contents' in self.server_config.s3_connection.list_objects(
                Bucket=self.server_config.sys_config.s3_bucket).keys():
            for key in self.server_config.s3_connection.list_objects(Bucket=self.server_config.sys_config.s3_bucket)['Contents']:
                if key['Key'] == path:
                    logging.warning('Model path ' + path + ' already exists in S3 bucket. Please specify another path.', UserWarning)
                    sys.exit()
        else:
            logging.info('No pickled objects in bucket.')

    def _load(self, src):
        '''
        Allows for the retrieval of a previously trained model via Amazon S3 or the preloaded
        pickle_jar of models and encoders.
        :param src:
        :return:
        '''

        # use preloaded dictionary
        if self.server_config.sys_config.models_preload_pickles:
            return self.server_config.sys_config.pickle_jar[src]

        # use global s3_connection and s object
        response = self.server_config.s3_connection.get_object(Bucket=self.server_config.sys_config.s3_bucket, Key=src)
        response_body = response['Body'].read()
        p_obj = dill.loads(response_body)

        return p_obj

    def _store(self, model, model_path, encoder_path, encoder, override=False):
        '''
        Store a model and encoders via Amazon S3.
        :param model:
        :param model_path:
        :param encoder_path:
        :param encoder:
        :return:
        '''

        # check if model path or encoder path already exist in S3
        if override is False:
            self._check_path(model_path)
            self._check_path(encoder_path)
            for i in model.independent:
                if 'generator_path' in i.keys():
                    self._check_path(i['generator_path'])

        with open('tmp/' + model_path, 'wb') as m:
            model.model_path = model_path
            model.id = self._id_generator(model)
            dill.dump(model.model, m, protocol=dill.HIGHEST_PROTOCOL)

        if not model_path or not encoder_path:
            logging.warning('Must store model and encoder prior to deployment.', UserWarning)

        elif all(v is not None for v in [encoder_path, encoder]):
            model.encoder_path = encoder_path
            model.encoder_type = encoder.__class__.__name__

            text_vectorizers = ['CountVectorizer', 'TfidfVectorizer']
            if model.encoder_type == 'LabelEncoder':
                model.encoder_data = encoder.classes_
            elif model.encoder_type in text_vectorizers:
                model.encoder_data = encoder

            with open('tmp/' + encoder_path, 'wb') as e:
                dill.dump(model.encoder_data, e, protocol=dill.HIGHEST_PROTOCOL)

        # use global s3 connection object and s object
        self.server_config.s3_connection.upload_file('tmp/' + model_path, self.server_config.sys_config.s3_bucket,
                                                     model_path)
        if all(v is not None for v in [encoder_path, encoder]):
            self.server_config.s3_connection.upload_file('tmp/' + encoder_path, self.server_config.sys_config.s3_bucket,
                                                         encoder_path)
        for i in model.independent:
            if 'generator_path' in i.keys():
                self.server_config.s3_connection.upload_file('tmp/' + i['generator_path'],
                                                             self.server_config.sys_config.s3_bucket,
                                                             i['generator_path'])

        # remove things
        os.remove('tmp/' + model_path)
        if encoder_path:
            os.remove('tmp/' + encoder_path)
        for i in model.independent:
            if 'generator_path' in i.keys():
                os.remove('tmp/' + i['generator_path'])

        self.index.update(model)

    def _deploy(self, model):
        '''
        Sets a model stored in Elasticsearch to an active "deployed" state for use in henosis. Model
        parameters are updated in SKlearn for now. Although this function simply calls the
        :return:
        '''

        self.index.update(model)

    @staticmethod
    def _train(model, X, y, balance=None, encoder=None):
        '''
        Trains a model and returns the results.
        :param model:
        :param X:
        :param Y:
        :param balance:
        :param encoder:
        :return:
        '''
        m = model.fit(X, y)
        start = time.time()
        y_score = m.predict(X)
        end = time.time()
        train_accuracy = accuracy_score(y_score, y)
        # https://stackoverflow.com/questions/31421413/how-to-compute-precision-recall-accuracy-and-f1-score-for-the-multiclass-case
        train_recall = recall_score(y_score, y, average='weighted')
        train_precision = precision_score(y_score, y, average='weighted')
        train_f1 = f1_score(y_score, y, average='weighted')
        train_results = {
            'accuracy': train_accuracy,
            'recall': train_recall,
            'precision': train_precision,
            'f1': train_f1
        }

        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%S')
        time_spent = end - start

        return train_results, timestamp, time_spent, balance

    @staticmethod
    def _test(model, X, y):
        '''
        Tests a model and returns the results.
        :param model:
        :param X:
        :param Y:
        :param load:
        :return:
        '''
        y_test = np.array(y)
        start = time.time()
        y_score = model.predict(X)
        end = time.time()
        test_accuracy = accuracy_score(y_score, y)
        # https://stackoverflow.com/questions/31421413/how-to-compute-precision-recall-accuracy-and-f1-score-for-the-multiclass-case
        test_recall = recall_score(y_score, y, average='weighted')
        test_precision = precision_score(y_score, y, average='weighted')
        test_f1 = f1_score(y_score, y, average='weighted')
        test_results = {
            'accuracy': test_accuracy,
            'recall': test_recall,
            'precision': test_precision,
            'f1': test_f1
        }

        if np.unique(y_test).shape != np.unique(y_score).shape:
            print('Classes ' + str(np.setdiff1d(y_test, y_score)) + ' in test data but not in predicted values.')

        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%S')
        time_spent = end - start

        return test_results, timestamp, time_spent

    class SKModel(object):
        '''
        This class facilitates training, testing, storage, and deployment of
        scikit-learn models.
        '''

        def __init__(self, estimator=None, name=None, encoder=None):
            self.estimator = estimator
            self.name = name
            self.id = None
            self.deployed = False
            self.call_count = 0
            self.last_call = None
            self.recommendation_threshold = 0.0
            self.train_results = None
            self.test_results = None
            self.dependent = None
            self.independent = None
            self.classes = None
            self.model = None
            self.encoder = encoder
            self.tpr = None
            self.fpr = None
            self.roc_auc = None
            self.model_path = None
            self.encoder_path = None
            self.encoder_type = None
            self.train_timestamp = None
            self.train_time = None
            self.train_data_balance = None
            self.test_timestamp = None
            self.test_time = None

        def train(self, data):
            self.model = OneVsRestClassifier(self.estimator).fit(data.X_train, data.y_train)
            self.dependent = data.dependent
            independent_vars = []
            for i in data.independent:
                independent_vars.append({"name": i})
            self.independent = independent_vars

            train_results, timestamp, train_time, train_data_balance = Models()._train(self.model, data.X_train,
                                                                                       data.y_train,
                                                                                       balance=data.balance,
                                                                                       encoder=self.encoder)

            self.train_results = train_results
            self.train_timestamp = timestamp
            self.train_time = train_time
            self.train_data_balance = train_data_balance

        def test(self, data):

            if self.model is None:
                logging.warning('Model not yet specified. Please train model or load one from disk.', UserWarning)

            test_results, timestamp, test_time = Models()._test(self.model, data.X_test, data.y_test)

            self.test_results = test_results
            self.test_timestamp = timestamp
            self.test_time = test_time

        def predict(self, X):

            if self.model is None:
                logging.warning('Model not yet specified. Please train model or load one from disk.', UserWarning)

            y_pred = self.model.predict(X)

            return y_pred

        def predict_proba(self, X):

            if self.model is None:
                logging.warning('Model not yet specified. Please train model or load one from disk.', UserWarning)

            Y_pred_proba = self.model.predict_proba(X)

            return Y_pred_proba

        def store(self, model_path, server_config, encoder_path=None, encoder=None, override=False):

            Models(server_config=server_config)._store(
                model=self,
                model_path=model_path,
                encoder_path=encoder_path,
                encoder=encoder,
                override=override
            )
            logging.info('Model stored successfully.')

        def deploy(self, server_config, deploy=False, recommendation_threshold=0.0):

            if not self.model_path or not self.encoder_path:
                logging.warning('Must store model and encoder prior to deployment.', UserWarning)
            else:
                self.deployed = deploy
                self.recommendation_threshold = recommendation_threshold
                Models(server_config=server_config)._deploy(
                    model=self
                )
                logging.info('Model deployed successfully and will be available after the next server restart.')

        def tag_generator(self, func, output_var, input_vars, generator_path=None):
            if generator_path:
                generator_path = generator_path
            else:
                generator_path = func.__name__ + '.pickle'
            if not isinstance(self.independent, list):
                logging.warning('Independent variables not defined as a list.', UserWarning)
                sys.exit()
            for i in self.independent:
                if i['name'] == output_var:
                    i['inputs'] = input_vars
                    i['generator_path'] = generator_path
            with open('tmp/' + generator_path, 'wb') as g:
                dill.dump(func, g, protocol=dill.HIGHEST_PROTOCOL)
            logging.info('Generator tagged successfully.')


class Data(object):
    '''
    This class facilitates the loading, storage, and preparation of data for modeling purposes.
    '''

    def __init__(self):
        self.all = None
        self.balance = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.dependent = None
        self.independent = None

    def load(self, csv_path=None):

        '''
        Load data for models in the recommender system from a variety of data sources. Currently supports
        local CSV files and MySQL databases.
        :param src:
        :param path:
        :param host:
        :param db:
        :param un:
        :param pw:
        :param query:
        :return:

        @TODO:
        - elasticsearch
        - postgres
        - excel
        - json # maybe just have this for elastic?
        - sas
        '''

        if csv_path:
            self.all = pd.read_csv(csv_path)

        if self.all is not None:
            logging.info('Data loaded successfully.')
        else:
            logging.warning('Error loading data.', UserWarning)

        return self.all

    @staticmethod
    def store(dataframe, csv_path=None):

        '''
        Store data loaded in the recommender system to local disk.
        :param target:
        :param dataframe:
        :param local:
        :return:
        '''

        dataframe.to_csv(csv_path)
        logging.info('Data stored successfully.')

    def test_train_split(self, X, y, share_train=0.8, stratify=None, balance=None, X_label=None):

        '''
        Create testing and training splits from the provided data. If balance is not None,
        balances data by upsampling or downsampling (upsample, downsample) using RandomSampling.
        Requires the imbalanced-learn library.
        :param X:
        :param Y:
        :param share_train:
        :param stratify:
        :param balance:
        :param mod:
        :param min_val:
        :return:
        '''

        if X.shape[0] != y.shape[0]:
            logging.warning('X and Y are not the same length.', UserWarning)

        # set aside X_test and y_test so that test data is not upsample or downsample data
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=(1. - share_train),
            stratify=stratify
        )

        self.dependent = y.name
        if X_label:
            self.independent = X_label
        else:
            self.independent = list(X.columns.values)
        self.balance = balance

        if balance == 'upsample':
            ros = RandomOverSampler()
            X_resample, y_resample = ros.fit_sample(X_train, y_train)
        elif balance == 'downsample':
            rus = RandomUnderSampler()
            X_resample, y_resample = rus.fit_sample(X_train, y_train)
        else:
            X_resample = X
            y_resample = y

        self.X_train, X_test, self.y_train, y_test = train_test_split(
            X_resample,
            y_resample,
            test_size=(1. - share_train),
            stratify=stratify
        )


