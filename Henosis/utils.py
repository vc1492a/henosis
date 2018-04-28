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
import datetime
import dill
from flask import session
import json
import logging
import requests
import sys
import time
import uuid
import yaml

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

__author__ = 'Valentino Constantinou'
__version__ = '0.0.10'
__license__ = 'Apache License, Version 2.0'


# resources #
class _LoadConfig:

    def __init__(self, config_yaml_path):
        self.config_yaml_path = config_yaml_path
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
        self.models_refresh_pickles = None
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
                self.models_refresh_pickles = config_params['models']['refresh_pickles']
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
            logging.info('No pickled objects in bucket.')

        return pickles


class Connect:

    def __init__(self):
        self.config_yaml_path = None
        self.sys_config = None
        self.es_connection_models = None
        self.es_connection_requestlog = None
        self.s3_connection = None
        self.port = None
        self.routes = None
        self.api_resources = None
        self.base_url = None

    def config(self, config_yaml_path, server=False):
        self.config_yaml_path = config_yaml_path
        self.sys_config = _LoadConfig(config_yaml_path=self.config_yaml_path)
        self.sys_config.config()
        self.es_connection_models = self.sys_config.elasticsearch(self.sys_config.elasticsearch_index_models)
        self.es_connection_requestlog = self.sys_config.elasticsearch(self.sys_config.elasticsearch_index_requestlog)
        self.s3_connection = self.sys_config.aws_s3()
        if self.sys_config.models_preload_pickles is True and server is True:
            self.sys_config.preload_pickles(self)
        self.base_url = self.sys_config.api_index + self.sys_config.api_version

        return self


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
            "modelsUsed": r['modelsUsed'],
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
                logging.warning('Status code: ' + str(request.status_code))

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
                    "recommendationThreshold": model.recommendation_threshold,
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
                return {
                           'response': r,
                           'description': 'Error updating model ' + r['_id'] + '.'
                       }, 400
            else:
                return {'count': r['count'], 'description': 'The number of models indexed.'}, 200



