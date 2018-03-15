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
import boto3
from collections import OrderedDict
import datetime
import dill
from flask import Flask, make_response, render_template, session
from flask_cors import CORS
from flask_restful import Api, reqparse, request, Resource
from functools import wraps
from gevent.wsgi import WSGIServer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import itertools
import json
import logging
import numpy as np
import operator
import os
import pandas as pd
import pymssql
import requests
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
import sys
import time
import uuid
import warnings
import yaml


__author__ = 'Valentino Constantinou'
__version__ = '0.0.5'
__license__ = 'Apache License, Version 2.0'


# resources #
class _Auth:
    '''
    This class enables simple authentication for routes using the credentials specified in config.yaml.
    '''

    @staticmethod
    def check_auth(req_username, req_password, config_username=None, config_password=None):
        if config_username is not None and config_password is not None:
            username_bool = req_username == config_username
            password_bool = req_password == config_password
            if username_bool is True and password_bool is True:
                return True
            return False
        return True

    @staticmethod
    def authenticate():
        message = {'message': "Incorrect username and/or password or username/password not needed. "
                              "Please authenticate or retry without username/password."}
        resp = make_response(json.dumps(message), 401)
        return resp

    @staticmethod
    def requires_auth(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            auth = request.authorization
            config = args[0].server_config.sys_config.api_auth
            if not auth:
                pass
            elif not _Auth.check_auth(auth.username, auth.password, config[0], config[1]):
                return _Auth.authenticate()
            return f(*args, **kwargs)

        return decorated


class _Recommendations(Resource):
    '''
    This class is used to interface with the recommender system framework and provide recommendations for fields.
    It accepts a flat JSON of form fields and their values. Empty fields should be passed to the recommender
    system with the specified api.missing tag in config.yaml.

    Example request:
    curl http://localhost:5005/api/v0.1/predict -d "formData={'badgeNumber': '1234', 'title': '999999',
    'description': '999999', 'projectName': '999999', 'specificEnvironment': '999999'}"
    '''

    def __init__(self, server_config):
        self.server_config = server_config

    @_Auth.requires_auth
    def get(self):
        ts_in = time.time()
        parser = reqparse.RequestParser()
        # print(reqparse.request.json) # should not be none, or allow acceptance of this
        parser.add_argument('formData')
        args = parser.parse_args()
        # ensure proper JSON
        formatted = json.loads(args['formData'].replace("'", '"'))
        r = _FormProcessor(self.server_config).get_recommendations(formatted)
        response = make_response(json.dumps(r[0]), r[1])
        cookies = reqparse.request.cookies
        session_manager = _SessionManager(self.server_config)
        response_cookie = session_manager.cookie_manager(cookies, response)
        if response_cookie:
            ts_out = time.time()
            session_manager.log_request(response_cookie, start_time=ts_in, end_time=ts_out)
            return response_cookie
        else:
            return response


class _ModelInfo(Resource):
    '''
    This class is used to interface with the recommender system framework and provide model information based
    on passed filters: currently supports 'deployed' (true/false) and 'dependent' (name of variable) fields.
    Values of passed fields must be passed as strings and lowercase.

    Example request:
    curl http://localhost:5005/api/v0.1/models -d "modelInfo={'dependent': 'projectname', 'deployed': 'true'}"
    '''

    def __init__(self, server_config):
        self.server_config = server_config

    @_Auth.requires_auth
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('modelInfo')
        args = parser.parse_args()
        # ensure proper JSON
        if args['modelInfo'] is not None:
            formatted = json.loads(args['modelInfo'].replace("'", '"'))
            # obtain the models that match the provided attributes
            r = _ElasticSearch().Models(self.server_config).get(**formatted)
        else:
            r = _ElasticSearch().Models(self.server_config).get()
        response = make_response(json.dumps(r[0]), r[1])
        return response


class _RequestLogs(Resource):
    '''
    This class is used to interface with the recommender system framework and provide request log information based
    on passed filters. Values of passed fields must be passed as strings and lowercase.

    Example request:
    curl http://localhost:5005/api/v0.1/requestlog -d "requestInfo={'responseStatusCode': 200}"
    '''

    def __init__(self, server_config):
        self.server_config = server_config

    @_Auth.requires_auth
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('requestInfo')
        args = parser.parse_args()
        # ensure proper JSON
        if args['requestInfo'] is not None:
            formatted = json.loads(args['requestInfo'].replace("'", '"'))
            # obtain the request logs that match the provided attributes
            r = _ElasticSearch().Requests(self.server_config).get(**formatted)
        else:
            r = _ElasticSearch().Requests(self.server_config).get()
        response = make_response(json.dumps(r[0]), r[1])
        return response


class _MultiColumnLabelEncoder:
    '''
    This class wraps scikit-learn's LabelEncoder() so that it can be used on multiple
    columns at the same time, e.g. when fitting to X. Many thanks to
    Price Hardman: https://stackoverflow.com/a/30267328/5441252
    '''

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''

        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)

        return output

    def fit_transform(self, X, y=None):

        return self.fit(X, y).transform(X)


class _FormProcessor(object):
    '''
    This class accepts form data from the /recommend API endpoint and:

    1: determines which fields are empty according to some pre-defined string (999999 in this case).
    2: checks to see if we have a model that can help predict that field. If yes, then
    2a: see if the provided form json has the fields we need to predict
    3: if yes, then
    3a: load the pickled model from disk
    3b: use the values from the form to make a prediction
    4: return the predicted value
    5: add that predicted value to the scope

    '''

    def __init__(self, server_config):
        self.server_config = server_config
        self.missing = server_config.sys_config.api_missing
        self.missing_fields = []
        self.index = _ElasticSearch().Models(server_config)
        self.model_functions = Models(server_config)
        self.combinations = []
        self.predictions = {}
        self.predictions_proba = {}
        self.predict_probabilities = server_config.sys_config.models_predict_probabilities
        self.top_n = server_config.sys_config.models_top_n
        self.models_queried = []
        self.text_vectorizers = ['CountVectorizer', 'TfidfVectorizer']

    @staticmethod
    def most_common(l):
        m = max(set(l), key=l.count)
        return [m]

    def generator_check(self, form_data, independent_var):
        # if the generated variable is not in the form, load the generator and populate the form
        if independent_var['name'] not in form_data.keys() and 'generator_path' in independent_var.keys() and set(independent_var['inputs']) <= set(
                form_data.keys()):
            g = self.model_functions._load(independent_var['generator_path'])
            # need way to handle multiple inputs for functions
            input_vals = [form_data[inp] for inp in independent_var['inputs']]
            return g(*input_vals)
        return None

    def conditions_met(self, deployed_flag, form_data, k, y, X):
        if deployed_flag is True and form_data[k] == self.missing and y == k and set(X) <= set(form_data.keys()):
            return True
        return False

    def load_vectorizer(self, input_data, encoder_type, encoder_path):
        if encoder_type == 'LabelEncoder' and encoder_path is not None:
            encoder = LabelEncoder()
            encoder.classes_ = self.model_functions._load(encoder_path)
            input_data = _MultiColumnLabelEncoder().fit_transform(input_data)
        elif encoder_type in self.text_vectorizers and encoder_path is not None:
            if encoder_type == 'CountVectorizer':
                encoder = CountVectorizer()
            else:
                encoder = TfidfVectorizer()
            encoder.vocabulary_ = self.model_functions._load(encoder_path)
            # gather text from columns and place in single list for processing if not empty
            X_t = []
            for i in input_data.columns.values:
                if input_data[i][0] != self.missing:
                    X_t.append(input_data[i][0])
            t = np.asarray([" ".join(str(s) for s in list(X_t))])
            input_data = encoder.transform(t)
        return input_data

    def predict_majority(self, model, input_data, k):
        # make the prediction
        p = model.predict(input_data)
        # add the list of predictions if a key doesn't exist
        if k not in self.predictions.keys():
            self.predictions[k] = []
        # add this prediction to the list of all predictions for this field
        self.predictions[k].append(p[0])

    def predict_proba(self, model, input_data, k):
        # make the prediction
        p = model.predict_proba(input_data)
        # organize the probabilities
        probas = {}
        for p, c in zip(p[0], model.classes_):
            probas[c] = p
        # store the probabilities
        if k not in self.predictions_proba.keys():
            self.predictions_proba[k] = probas
        else:
            # note that probabilities will no longer sum to 1
            for p in probas:
                if k in self.predictions_proba.keys():
                    # create a list of probabilities for each class which we will later average
                    if self.predictions_proba[k][p].__class__.__name__ == 'float64':
                        proba = self.predictions_proba[k][p]
                        self.predictions_proba[k][p] = []
                        self.predictions_proba[k][p].append(proba)
                    self.predictions_proba[k][p].append(probas[p])

    def sort_votes(self):
        for k in self.predictions.keys():
            self.predictions[k] = self.most_common(self.predictions[k])

    def sort_proba(self):
        # for each key we want to predict
        for k in self.predictions_proba.keys():
            # create a top_n list to append to
            self.predictions[k] = []
            # average the predicted probabilities
            for cls in self.predictions_proba[k]:
                self.predictions_proba[k][cls] = np.mean(self.predictions_proba[k][cls])
            # sort and fetch top_n as list and return
            sorted_probas = OrderedDict(
                sorted(self.predictions_proba[k].items(), key=operator.itemgetter(1), reverse=True))
            top_n_slice = itertools.islice(sorted_probas.items(), 0, self.top_n)
            for key, value in top_n_slice:
                self.predictions[k].append(key)

    def predict(self, model, input_data, k):
        # make predictions based on config
        if self.predict_probabilities:
            self.predict_proba(model, input_data, k)
        else:
            self.predict_majority(model, input_data, k)

    def process_form(self, form_data, models):
        # process the form and store dep/indp model combinations for which we have models for, return prediction
        iter_list = itertools.product(form_data.keys(), models)
        for k, m in iter_list:
            # get independent variable names
            independent_vars = []
            for i in m['_source']['independent']:
                independent_vars.append(i['name'])
                gen_vals = self.generator_check(form_data, i)
                if gen_vals is not None:
                    form_data[i['name']] = gen_vals
            # ensure form field is missing and that an appropriate model is available and deployed for use
            if self.conditions_met(m['_source']['deployed'], form_data, k, m['_source']['dependent'], independent_vars):
                self.missing_fields.append(k)
                # process the input data, ensure order
                input_data = pd.DataFrame(
                    [[form_data[k] for k in independent_vars]],
                    columns=independent_vars
                )
                input_data = self.load_vectorizer(input_data, m['_source']['encoderType'], m['_source']['encoderPath'])
                # load the estimator
                pred_m = self.model_functions._load(m['_source']['modelPath'])
                # make predictions
                self.predict(pred_m, input_data, k)
                # add to the callCount, lastCall and update
                m['_source']['callCount'] += 1
                m['_source']['lastCall'] = datetime.datetime.fromtimestamp(time.time()).strftime(
                    '%Y-%m-%dT%H:%M:%S')
                self.models_queried.append(m['_id'])
                self.index.update(update_dict=m['_source'])
        if self.predict_probabilities:
            self.sort_proba()
        else:
            self.sort_votes()

    def get_recommendations(self, form_data):
        try:
            # retrieve any deployed models from the index
            response = self.index.get(deployed=True)
            models = response[0]['models']
            # self.index.models_last_refresh = response[0]['models_last_refresh']
            # print(self.index.models_last_refresh)
        except Exception as e:
            logging.warning(e)
            warnings.warn(e)
            return {'description': 'Error retrieving models from Elasticsearch.'}, 400

        try:
            self.process_form(form_data, models)
            return {
                       'predictions': self.predictions,
                       'modelsQueried': self.models_queried,
                       'description': 'Model ids and form field predictions provided by the available models.'
                   }, 200
        except Exception as e:
            logging.warning(e)
            warnings.warn(e)
            return {'description': 'Error processing models and their predictions.'}, 400


class _SessionManager:

    def __init__(self, server_config):
        self.server_config = server_config
        self.session = session
        self.connection_requests = server_config.es_connection_requestlog
        self.index = _ElasticSearch().Requests(server_config)
        self.sessionId = None
        self.sessionExpireDate = None
        self.timeIn = None
        self.timeOut = None
        self.timeElapsed = None
        self.responseCode = None
        self.responseDescription = None

    def cookie_manager(self, cookies, response):
        try:
            ts = time.time()
            ts_expire = ts + (60 * self.server_config.sys_config.api_session_expiration)
            timestamp_expire = datetime.datetime.fromtimestamp(ts_expire).strftime('%Y-%m-%dT%H:%M:%S')
            if 'session_id' not in cookies.keys():
                idc = uuid.uuid4().hex
            else:
                idc = cookies['session_id']
            self.sessionId = idc
            if idc not in self.session.keys():
                logging.info('Session ' + idc + ' not in session.')
                idc = uuid.uuid4().hex
                self.session[idc] = {
                    'expTimestamp': timestamp_expire
                }
                session[idc] = self.session[idc]
                self.sessionId = idc
                logging.info('Session ' + idc + ' now in session.')
            elif ts > time.mktime(time.strptime(session[idc]['expTimestamp'], '%Y-%m-%dT%H:%M:%S')):
                sessions = [self.session, session]
                logging.info('Session ' + idc + ' has expired.')
                for sesh in sessions:
                    try:
                        sesh.pop(idc)
                    except KeyError:
                        pass
                idc = uuid.uuid4().hex
                self.session[idc] = {
                    'expTimestamp': timestamp_expire
                }
                session[idc] = self.session[idc]
                self.sessionId = idc
                logging.info('Session ' + idc + ' now in session.')
            else:
                logging.info('Session ' + idc + ' already in session.')
            self.sessionExpireDate = timestamp_expire
            self.responseCode = response.status_code
            self.responseDescription = response.status
            response.set_cookie('session_id', idc, expires=timestamp_expire)
            return response
        except Exception as e:
            logging.warning('Issue with cookie manager.', RuntimeError)
            logging.warning(e, RuntimeError)
            return None

    def log_request(self, response, start_time=None, end_time=None):
        if all(v is not None for v in [start_time, end_time]):
            self.timeIn = start_time
            self.timeOut = end_time
            self.timeElapsed = end_time - start_time

        r = json.loads(response.get_data().decode('utf8'))

        timestamp_in = datetime.datetime.fromtimestamp(self.timeIn).strftime('%Y-%m-%dT%H:%M:%S')
        timestamp_out = datetime.datetime.fromtimestamp(self.timeOut).strftime('%Y-%m-%dT%H:%M:%S')

        request_log = {
            "sessionId": self.sessionId,
            "sessionExpireDate": self.sessionExpireDate,
            "timeIn": timestamp_in,
            "timeOut": timestamp_out,
            "timeElapsed": self.timeElapsed,
            "responseStatusCode": self.responseCode,
            "responseDescription": self.responseDescription,
            "recommendations": r['predictions'],
            "missingFields": list(r['predictions'].keys()),
            "modelsQueried": r['modelsQueried'],
        }

        self.index.store(request_log)


class _ElasticSearch:

    class Common:

        @staticmethod
        def build_query(kwargs, operation="must"):
            q = {
                "query": {
                    "constant_score": {
                        "filter": {
                            "bool": {
                                operation: []
                            }
                        }
                    }
                }
            }
            for k in kwargs:
                if isinstance(k, dict):
                    for d_k in k:
                        q['query']['constant_score']['filter']['bool']['must'].append({"term": {d_k: k[d_k]}})
                else:
                    q['query']['constant_score']['filter']['bool']['must'].append({"term": {k: kwargs[k]}})
            return q

    class Connection(object):

        def __init__(self):
            self.host = None
            self.index = None
            self.type = None
            self.headers = None
            self.ssl_verify = True
            self.auth = None
            self.search_size = 10000

        def config(self, host=None, index=None, type=None, ssl_verify=True, auth=None, search_size=10000):
            self.host = host
            self.index = index
            self.type = type
            self.ssl_verify = ssl_verify
            self.auth = auth
            self.search_size = search_size

            return self

        @staticmethod
        def validate_connection_property(property_dict):
            for p in property_dict:
                if property_dict[p] is None:
                    logging.warning('Property ' + p + ' must not be None', UserWarning)
                    sys.exit()

        @staticmethod
        def request_status_parser(request):

            if request.status_code == 200:
                logging.info('Connection to Elasticsearch resource successful.')
                warnings.warn('Connection to Elasticsearch resource successful.')
                try:
                    logging.info(
                        json.dumps(
                            request.json(),
                            indent=4
                        )
                    )
                except json.decoder.JSONDecodeError:
                    pass
            else:
                logging.warning('Error establishing connection to Elasticsearch resource.', UserWarning)
                warnings.warn('Error establishing connection to Elasticsearch resource.', UserWarning)
                logging.warning('Status code: ' + str(request.status_code))
                warnings.warn('Status code: ' + str(request.status_code))

        def test_host(self, host=None):

            test_h = self.host
            if host:
                test_h = host

            connection_properties = {
                'HOST': test_h
            }

            self.validate_connection_property(connection_properties)

            r = requests.get(test_h, verify=self.ssl_verify, auth=self.auth)

            self.request_status_parser(r)

        def test_index(self, host=None, index=None):

            test_h = self.host
            test_i = self.index
            if host:
                test_h = host
            if index:
                test_i = index

            connection_properties = {
                'HOST': test_h,
                'INDEX': test_i
            }

            self.validate_connection_property(connection_properties)

            test_endpoint = test_h + test_i

            r = requests.get(test_endpoint, verify=self.ssl_verify, auth=self.auth)

            self.request_status_parser(r)

        def test_type(self, host=None, index=None, type=None):

            test_h = self.host
            test_i = self.index
            test_t = self.type
            if host:
                test_h = host
            if index:
                test_i = index
            if type:
                test_t = type

            connection_properties = {
                'HOST': test_h,
                'INDEX': test_i,
                'TYPE': test_t
            }

            self.validate_connection_property(connection_properties)

            test_endpoint = test_h + test_i + test_t

            r = requests.get(test_endpoint, verify=self.ssl_verify, auth=self.auth)

            self.request_status_parser(r)

        def test(self, host=None, index=None, type=None):

            test_h = self.host
            test_i = self.index
            test_t = self.type
            if host:
                test_h = host
            if index:
                test_i = index
            if type:
                test_t = type

            tests = [
                self.test_host(host=test_h),
                self.test_index(host=test_h, index=test_i),
                self.test_type(host=test_h, index=test_i, type=test_t)
            ]

            for t in tests:
                try:
                    t
                except Exception as e:
                    logging.warning(e)

    class Requests(object):

        def __init__(self, server_config):

            self.host = server_config.es_connection_requestlog.host
            self.index = server_config.es_connection_requestlog.index
            self.headers = server_config.es_connection_requestlog.headers
            self.ssl_verify = server_config.es_connection_requestlog.ssl_verify
            self.auth = server_config.es_connection_requestlog.auth
            self.search_size = server_config.es_connection_requestlog.search_size

        def get(self, **kwargs):

            q = _ElasticSearch.Common.build_query(kwargs)

            r = requests.get(
                self.host + self.index + '/_search?size=' + str(self.search_size),
                verify=self.ssl_verify,
                auth=self.auth,
                json=q
            ).json()

            if r['_shards']['failed'] > 0:
                logging.warning(r)
                warnings.warn(r)
                return {'description': 'One or more Elasticsearch shards failed.'}, 400
            else:
                return {
                           'requests': r['hits']['hits'],
                           'description': 'All requests in the Elasticsearch index.'
                       }, 200

        def store(self, request_info):
            r = requests.post(
                self.host + self.index + '/requestlog',
                json=request_info,
                verify=self.ssl_verify,
                auth=self.auth
            ).json()

            try:
                if r['_shards']['failed'] > 0:
                    logging.warning(r)
                    warnings.warn(r)
                    return {
                               'response': r,
                               'description': 'Error updating request log ' + r['_id'] + '.'
                           }, 400
                else:
                    return {'description': 'Request log ' + r['_id'] + ' indexed successfully.'}, 200
            except KeyError:
                return {
                           'response': r,
                           'description': 'Error indexing request log ' + r['_id'] + '.'
                       }, 400

        def count(self):
            r = requests.get(
                self.host + self.index + '/_count',
                verify=self.ssl_verify,
                auth=self.auth
            ).json()

            if r['_shards']['failed'] > 0:
                logging.warning(r)
                warnings.warn(r)
                return {
                           'response': r,
                           'description': 'Error updating model ' + r['_id'] + '.'
                       }, 400
            else:
                return {'count': r['count'], 'description': 'The number of request logs indexed.'}, 200

    class Models(object):

        def __init__(self, server_config):

            self.host = server_config.es_connection_models.host
            self.index = server_config.es_connection_models.index
            self.headers = server_config.es_connection_models.headers
            self.ssl_verify = server_config.es_connection_models.ssl_verify
            self.auth = server_config.es_connection_models.auth
            self.search_size = server_config.es_connection_models.search_size

        def get(self, **kwargs):

            q = _ElasticSearch.Common.build_query(kwargs)

            r = requests.get(
                self.host + self.index + '/_search?size=' + str(self.search_size),
                verify=self.ssl_verify,
                auth=self.auth,
                json=q
            ).json()

            if r['_shards']['failed'] > 0:
                logging.warning(r)
                warnings.warn(r)
                return {'description': 'One or more Elasticsearch shards failed.'}, 400
            else:
                return {
                           'models': r['hits']['hits'],
                           'description': 'Models that match the specified search criteria in the Elasticsearch index.'
                       }, 200

        def update(self, model=None, update_dict=None):

            if model:
                model_info = {
                    "callCount": model.call_count,
                    "lastCall": model.last_call,
                    "dependent": model.dependent,
                    "independent": model.independent,
                    "deployed": model.deployed,
                    "id": model.id,
                    "lastTrainedDate": model.train_timestamp,
                    "lastTestedDate": model.test_timestamp,
                    "modelPath": model.model_path,
                    "modelType": str(model.model),
                    "encoderPath": model.encoder_path,
                    "encoderType": model.encoder_type,
                    "testAccuracy": model.test_results["accuracy"],
                    "testPrecision": model.test_results["precision"],
                    "testRecall": model.test_results["recall"],
                    "testF1": model.test_results["f1"],
                    "trainAccuracy": model.train_results["accuracy"],
                    "trainPrecision": model.train_results["precision"],
                    "trainRecall": model.train_results["recall"],
                    "trainF1": model.train_results["f1"],
                    "trainTime": model.train_time,
                    "testTime": model.test_time,
                    "f1_threshold": model.f1_threshold,
                    "trainDataBalance": model.train_data_balance
                }

                r = requests.put(
                    self.host + self.index + '/model/' + str(model.id),
                    json=model_info,
                    verify=self.ssl_verify,
                    auth=self.auth
                ).json()

                try:
                    if r['_shards']['failed'] > 0:
                        logging.warning(r)
                        warnings.warn(r)
                        return {
                                   'response': r,
                                   'description': 'Error updating model ' + r['_id'] + '.'
                               }, 400
                    else:
                        return {'description': 'Model ' + r['_id'] + ' updated successfully.'}, 200
                except KeyError:
                    return {
                               'response': r,
                               'description': 'Error updating model ' + r['_id'] + '.'
                           }, 400

            if update_dict:
                r = requests.put(
                    self.host + self.index + '/model/' + str(update_dict['id']),
                    json=update_dict,
                    verify=self.ssl_verify,
                    auth=self.auth
                ).json()

                if r['result'] == 'updated':
                    return {'description': 'Model ' + r['_id'] + ' updated successfully.'}, 200
                else:
                    return {
                               'response': r,
                               'description': 'Error updating model ' + r['_id'] + '.'
                           }, 400

        def count(self):
            r = requests.get(
                self.host + self.index + '/_count',
                verify=self.ssl_verify,
                auth=self.auth
            ).json()

            if r['_shards']['failed'] > 0:
                logging.warning(r)
                warnings.warn(r)
                return {
                           'response': r,
                           'description': 'Error updating model ' + r['_id'] + '.'
                       }, 400
            else:
                return {'count': r['count'], 'description': 'The number of models indexed.'}, 200


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
                    logging.warning('Model path already exists in S3 bucket. Please specify another path.', UserWarning)
                    warnings.warn('Model path already exists in S3 bucket. Please specify another path.', UserWarning)
                    sys.exit()
        else:
            logging.warning('No picked objects in bucket.', UserWarning)
            warnings.warn('No picked objects in bucket.', UserWarning)

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
        if not override:
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
            warnings.warn('Must store model and encoder prior to deployment.', UserWarning)

        elif all(v is not None for v in [encoder_path, encoder]):
            model.encoder_path = encoder_path
            model.encoder_type = encoder.__class__.__name__

            text_vectorizers = ['CountVectorizer', 'TfidfVectorizer']
            if model.encoder_type == 'LabelEncoder':
                model.encoder_data = encoder.classes_
            elif model.encoder_type in text_vectorizers:
                model.encoder_data = encoder.vocabulary_

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
            self.f1_threshold = 0.0
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
                warnings.warn('Model not yet specified. Please train model or load one from disk.', UserWarning)

            test_results, timestamp, test_time = Models()._test(self.model, data.X_test, data.y_test)

            self.test_results = test_results
            self.test_timestamp = timestamp
            self.test_time = test_time

        def predict(self, X):

            if self.model is None:
                logging.warning('Model not yet specified. Please train model or load one from disk.', UserWarning)
                warnings.warn('Model not yet specified. Please train model or load one from disk.', UserWarning)

            y_pred = self.model.predict(X)

            return y_pred

        def predict_proba(self, X):

            if self.model is None:
                logging.warning('Model not yet specified. Please train model or load one from disk.', UserWarning)
                warnings.warn('Model not yet specified. Please train model or load one from disk.', UserWarning)

            Y_pred_proba = self.model.predict_proba(X)

            return Y_pred_proba

        def store(self, model_path, server_config, encoder_path=None, encoder=None, override=None):

            Models(server_config=server_config)._store(
                model=self,
                model_path=model_path,
                encoder_path=encoder_path,
                encoder=encoder,
                override=override
            )
            logging.info('Model stored successfully.')

        def deploy(self, server_config, deploy=False, f1_threshold=0.0):

            if not self.model_path or not self.encoder_path:
                logging.warning('Must store model and encoder prior to deployment.', UserWarning)
                warnings.warn('Must store model and encoder prior to deployment.', UserWarning)
            else:
                self.deployed = deploy
                self.f1_threshold = f1_threshold
                Models(server_config=server_config)._deploy(
                    model=self
                )
                logging.info('Model deployed successfully.')

        def tag_generator(self, func, output_var, input_vars, generator_path=None):
            if generator_path:
                generator_path = generator_path
            else:
                generator_path = func.__name__ + '.pickle'
            if not isinstance(self.independent, list):
                logging.warning('Independent variables not defined as a list.', UserWarning)
                warnings.warn('Independent variables not defined as a list.', UserWarning)
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

    def load(self, csv_path=None, db_type=None, host=None, db=None, un=None, pw=None, query=None, **kwargs):

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
        if db_type in ['MySQL', 'mysql', 'MySql', 'Mysql', 'MYSQL', 'mySql', 'mySQL']:
            conn = pymssql.connect(host=host, user=un, password=pw, database=db)
            self.all = pd.read_sql(query, conn)

        if self.all is not None:
            logging.info('Data loaded successfully.')
        else:
            logging.warning('Error loading data.', UserWarning)
            warnings.warn('Error loading data.', UserWarning)

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
            warnings.warn('X and Y are not the same length.', UserWarning)

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


class _LoadConfig:

    def __init__(self, yaml_path):
        self.config_yaml_path = yaml_path
        self.elasticsearch_host = None
        self.elasticsearch_index = None
        self.elasticsearch_index_models = None
        self.elasticsearch_index_requestlog = None
        self.elasticsearch_verify = None
        self.elasticsearch_auth = None
        self.api_index = None
        self.api_version = None
        self.api_missing = None
        self.api_session_expiration = None
        self.api_auth = None
        self.api_secret = None
        self.models_preload_pickles = None
        self.pickle_jar = None
        self.models_predict_probabilities = None
        self.models_top_n = None
        self.s3_region = None
        self.s3_bucket = None
        self.s3_key = None
        self.s3_secret = None

    def config(self):

        '''
        Loads recommender system parameters from config.yaml.
        :param src:
        :return:
        '''

        with open(self.config_yaml_path, 'r') as c:
            try:
                config_params = yaml.load(c)
                logging.info(config_params)

                # None in yaml file is read as string but needs to be converted
                config_params['elasticsearch']['auth'] = None if config_params['elasticsearch']['auth'] == 'None' else \
                    config_params['elasticsearch']['auth']

                self.elasticsearch_host = config_params['elasticsearch']['host']
                self.elasticsearch_index_models = config_params['elasticsearch']['models']['index']
                self.elasticsearch_index_requestlog = config_params['elasticsearch']['request_log']['index']
                self.elasticsearch_verify = config_params['elasticsearch']['verify']
                self.elasticsearch_auth = config_params['elasticsearch']['auth']
                self.api_index = config_params['api']['index']
                self.api_version = config_params['api']['version']
                self.api_missing = config_params['api']['missing']
                self.api_session_expiration = config_params['api']['session_expiration']
                self.api_auth = config_params['api']['auth']
                self.api_secret = config_params['api']['secret']
                self.models_preload_pickles = config_params['models']['preload_pickles']
                self.models_predict_probabilities = config_params['models']['predict_probabilities']
                self.models_top_n = config_params['models']['top_n']
                self.s3_region = config_params['aws_s3']['region']
                self.s3_bucket = config_params['aws_s3']['bucket']
                self.s3_key = config_params['aws_s3']['key']
                self.s3_secret = config_params['aws_s3']['secret']

                return self

            except yaml.YAMLError as exc:
                logging.info(exc, RuntimeWarning)
                logging.warning('Error reading config.yaml', RuntimeWarning)

    def elasticsearch(self, es_index=None):
        if es_index:
            self.elasticsearch_index = es_index
        connection = _ElasticSearch().Connection()
        connection.config(
            host=self.elasticsearch_host,
            index=self.elasticsearch_index,
            ssl_verify=self.elasticsearch_verify,
            auth=self.elasticsearch_auth
        )

        return connection

    def aws_s3(self):
        client = boto3.client(
            's3',
            region_name=self.s3_region,
            aws_access_key_id=self.s3_key,
            aws_secret_access_key=self.s3_secret
        )
        logging.info('Connection to AWS S3 resource successful.')

        return client

    def preload_pickles(self, server_config):
        pickles = {}
        if 'Contents' in server_config.s3_connection.list_objects(Bucket=server_config.sys_config.s3_bucket).keys():
            for key in server_config.s3_connection.list_objects(Bucket=server_config.sys_config.s3_bucket)['Contents']:
                response = server_config.s3_connection.get_object(Bucket=server_config.sys_config.s3_bucket,
                                                                  Key=key['Key'])
                response_body = response['Body'].read()
                p_obj = dill.loads(response_body)
                pickles[key['Key']] = p_obj
            self.pickle_jar = pickles
            logging.info('Successfully loaded models, encoders, and feature generator objects.')
        else:
            logging.warning('No picked objects in bucket.', UserWarning)
            warnings.warn('No picked objects in bucket.', UserWarning)


class Server:

    def __init__(self):
        self.sys_config = None
        self.es_connection_models = None
        self.es_connection_requestlog = None
        self.s3_connection = None
        self.port = None
        self.base_url = None

    def config(self, yaml_path):
        self.sys_config = _LoadConfig(yaml_path=yaml_path)
        self.sys_config.config()
        self.es_connection_models = self.sys_config.elasticsearch(self.sys_config.elasticsearch_index_models)
        self.es_connection_requestlog = self.sys_config.elasticsearch(self.sys_config.elasticsearch_index_requestlog)
        self.s3_connection = self.sys_config.aws_s3()
        if self.sys_config.models_preload_pickles:
            self.sys_config.preload_pickles(self)
        self.base_url = self.sys_config.api_index + self.sys_config.api_version

        return self

    def run(self, routes=None, api_resources=None, port=5005):
        self.port = port

        # add custom templates
        if routes:
            template_dir = None
            static_dir = None
            for r in routes:
                # clean this ish up eventually, no like.
                if static_dir is None:
                    static_dir = r['static_directory']
                if template_dir is None:
                    template_dir = r['template_directory']
                if r['static_directory'] != static_dir:
                    logging.warning('All static directories must be the same for custom templates.', UserWarning)
                    sys.exit()
                if r['template_directory'] != template_dir:
                    logging.warning('All template directories must be the same for custom templates.', UserWarning)
                    sys.exit()
            app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
            for r in routes:
                exec("""\n@app.route('""" + r['route'] + """')\ndef """ + r[
                    'function_name'] + """():\n    return render_template('""" + r['template'] + """')\n""")
            logging.info('Routes added successfully.')
        else:
            app = Flask(__name__)

        # configure cross-origin requests
        CORS(app, resources={r"/" + self.sys_config.api_index + "/*": {"origins": "*"}})

        # add API resources
        api = Api(app)
        api.add_resource(_Recommendations, self.base_url + '/recommend', resource_class_kwargs={'server_config': self})
        api.add_resource(_ModelInfo, self.base_url + '/models', resource_class_kwargs={'server_config': self})
        api.add_resource(_RequestLogs, self.base_url + '/requestlogs', resource_class_kwargs={'server_config': self})
        if api_resources:
            for r in api_resources:
                api.add_resource(r['class'], self.sys_config.api_index + self.sys_config.api_version + r['endpoint'])

        # add app secret key from config
        app.secret_key = self.sys_config.api_secret

        # spin up app
        logging.info('Running server on port: ' + str(self.port))
        app.debug = True
        server = WSGIServer(("", self.port), app)
        server.serve_forever()
