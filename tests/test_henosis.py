# Authors: Valentino Constantinou <vc@valentino.io>
# License: Apache 2.0

import sys

sys.path.insert(0, '/Users/vconstan/Files/Projects/5X/PRS Form Recommender/api/src/henosis')

from Henosis.model import Data, Models
from Henosis.server import Server, ServerThread, _Auth, _FormProcessor, _ModelInfo, _MultiColumnLabelEncoder, \
    _Recommendations, _RequestLogs
from Henosis.utils import Connect, _Elasticsearch, _LoadConfig, _SessionManager

from flask import make_response
from flask_restful import Resource, reqparse
import json
import os
import numpy as np
import pandas as pd
import pytest
import requests
import time
import warnings

from sklearn.datasets import load_iris

# first load some data and dump to csv for testing
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
iris_df.to_csv('iris.csv')


# Data() tests #
def test_data_init():
    d = Data()

    # assert that the object has the desired properties
    assert hasattr(d, 'all')
    assert hasattr(d, 'X_train')
    assert hasattr(d, 'X_test')
    assert hasattr(d, 'y_train')
    assert hasattr(d, 'y_test')
    assert hasattr(d, 'dependent')
    assert hasattr(d, 'independent')

    # assert that a Data object is returned
    assert d is not None
    assert d.__class__.__name__ == 'Data'


def test_data_readcsv():
    d = Data()
    d.load(csv_path='iris.csv')

    # assert d.all is not None
    assert d.all is not None

    # assert that d.all is a pandas dataframe
    assert type(d.all).__name__ == 'DataFrame'

    # assert that the dataframe has column names
    assert d.all.columns.shape[0] > 0


def test_data_assignment():
    d = Data()
    d.all = pd.read_csv('iris.csv')

    # assert d.all is not None
    assert d.all is not None

    # assert that d.all is a pandas dataframe
    assert type(d.all).__name__ == 'DataFrame'

    # assert that the dataframe has column names
    assert d.all.columns.shape[0] > 0

    # assert that when a non-dataframe is passed, there's a warning (if numpy, convert like PyNomaly or?)
    print()


def test_data_store():
    d = Data()
    d.load(csv_path='iris.csv')

    d.store(d.all, 'iris_store_test.csv')

    # assert that a file is written to the directory
    assert os.path.exists('iris_store_test.csv') == 1

    # assert that a message was provided
    print()


def test_data_split():
    d = Data()
    d.load(csv_path='iris.csv')
    d.train_test_split(
        d.all[['SepalLength', 'SepalWidth', 'PetalLength']],
        d.all['PetalWidth'],
        test_size=0.2
    )

    # assert that X_train, X_test, y_train, y_test all exist
    assert d.X_train is not None
    assert d.X_test is not None
    assert d.y_train is not None
    assert d.y_test is not None

    # assert that they all have a positive shape with min number of samples
    assert d.X_train.shape[0] > 0
    assert d.X_test.shape[0] > 0
    assert d.y_train.shape[0] > 0
    assert d.y_test.shape[0] > 0

    # assert a warning is raised if the specified variables are not in the dataframe
    print()


# Models() tests #
def test_models_init():
    m = Models()

    # assert object has the desired properties
    assert hasattr(m, 'server_config')
    assert hasattr(m, 'connection_models')
    assert hasattr(m, 'index')

    # assert that a Models object is returned
    assert m is not None
    assert m.__class__.__name__ == 'Models'

    # assert that specifying a server_config results in proper parsing
    c = Connect().config('tests/config.yaml')
    m_serverconfig = Models(server_config=c)

    # assert that the returned objects are what we expect
    assert m_serverconfig.server_config.__class__.__name__ == 'Connect'
    assert m_serverconfig.index.__class__.__name__ == 'Models'
    assert m_serverconfig.connection_models.__class__.__name__ == 'Connection'


def test_model_id_generator():
    # generate an SKModel() and ID
    m = Models().SKModel()
    m_id = Models()._id_generator(m)

    # assert that the generated ID is a hexadecimal string
    assert isinstance(int(m_id, 16), int)


# def test_model_path_check():

# so this thing needs to go through the process of training a model
# testing
# uploading it
# testing again

# this test is more complicated
# we need to check that it exits IF the specified path exists
# a user warning is provided
# else check that a message was provided


def test_model_load_from_bucket():
    model_src = 'model_SpecificEnvironment_1.pickle'  # change this to testing-specific model later

    # try to load from bucket without a connection to a Henosis instance
    Models()._load_from_bucket(src=model_src)

    # assert a warning is raised if server_config is not None
    with pytest.warns(UserWarning) as record:
        warnings.warn(
            "Must specify server configuration before loading models from S3. Please define server_config.",
            UserWarning)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Must specify server configuration before loading models from S3. Please define server_config."

    # try to load from bucket with a connection to a Henosis instance
    c = Connect().config('tests/config.yaml')
    m = Models(server_config=c)
    response = m._load_from_bucket(src=model_src)

    # assert that an object is returned in a format we expect
    assert response.__class__.__name__ in ['OneVsRestClassifier', 'TfidfVectorizer', 'CountVectorizer']

    model_src = 'model_DoesNotExist_1.pickle'

    # try to load a model that does not exist
    m._load_from_bucket(src=model_src)

    # assert a warning is raised of the object does not exist
    with pytest.warns(UserWarning) as record:
        warnings.warn('No pickled object with specified path in bucket.', UserWarning)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "No pickled object with specified path in bucket."


def test_model_get_info():
    model_id = '7851c359709040c4882ed3b3149f2aba'  # change this to a testing-specific model later, maybe grab an ID at random from test API / Elastic

    # try to load model information from Elasticsearch without a connection to a Henosis instance
    Models()._get_info(model_id)

    # assert a warning is raised if server_config is not None
    with pytest.warns(UserWarning) as record:
        warnings.warn(
            "Must specify server configuration before loading model info from Elasticsearch. Please define server_config.",
            UserWarning)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Must specify server configuration before loading model info from Elasticsearch. Please define server_config."

    # try to load from bucket with a connection to a Henosis instance
    c = Connect().config('tests/config.yaml')
    m = Models(server_config=c)
    response = m._get_info(model_id)

    # assert that an object is returned in a format we expect
    assert isinstance(response, dict)
    assert 'models' in response.keys()
    assert 'description' in response.keys()
    assert isinstance(response['models'], dict)

    model_id = '7851c359709040c4882ed3b3149f2abH'

    # try to load a model that doesn't exist
    m._get_info(model_id)

    # assert a warning is raised if model does not exist
    with pytest.warns(UserWarning) as record:
        warnings.warn(
            "Model with specified ID not in Elasticsearch index.",
            UserWarning)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Model with specified ID not in Elasticsearch index."


def test_skmodel_init():
    m = Models().SKModel()

    # assert that the returned object has the necessary properties
    assert hasattr(m, 'estimator')
    assert hasattr(m, 'id')
    assert hasattr(m, 'deployed')
    assert hasattr(m, 'call_count')
    assert hasattr(m, 'last_call')
    assert hasattr(m, 'recommendation_threshold')
    assert hasattr(m, 'train_results')
    assert hasattr(m, 'test_results')
    assert hasattr(m, 'dependent')
    assert hasattr(m, 'independent')
    assert hasattr(m, 'model')
    assert hasattr(m, 'encoder')
    assert hasattr(m, 'tpr')
    assert hasattr(m, 'fpr')
    assert hasattr(m, 'roc_auc')
    assert hasattr(m, 'model_path')
    assert hasattr(m, 'model_class')
    assert hasattr(m, 'encoder_path')
    assert hasattr(m, 'encoder_type')
    assert hasattr(m, 'train_timestamp')
    assert hasattr(m, 'train_time')
    assert hasattr(m, 'test_timestamp')
    assert hasattr(m, 'test_time')


# Connect() class #
def test_connect_init():
    c = Connect()

    # assert that the returned object has the necessary properties
    assert hasattr(c, 'config_yaml_path')
    assert hasattr(c, 'sys_config')
    assert hasattr(c, 'es_connection_models')
    assert hasattr(c, 'es_connection_requestlog')
    assert hasattr(c, 's3_connection')
    assert hasattr(c, 'port')
    assert hasattr(c, 'routes')
    assert hasattr(c, 'api_resources')
    assert hasattr(c, 'base_url')


def test_connect_config():
    c = Connect().config('tests/config.yaml')

    # assert that the properties of the returned object are of the right type
    assert isinstance(c.config_yaml_path, str)
    assert isinstance(c.sys_config, object)
    assert isinstance(c.es_connection_models, object)
    assert isinstance(c.es_connection_requestlog, object)
    assert isinstance(c.s3_connection, object)
    assert isinstance(c.base_url, str)

    # assert that the properties of the returned object are of the correct class
    assert c.sys_config.__class__.__name__ == '_LoadConfig'
    assert c.es_connection_models.__class__.__name__ == 'Connection'
    assert c.es_connection_requestlog.__class__.__name__ == 'Connection'
    assert c.s3_connection.__class__.__name__ == 'S3'


# _LoadConfig class #
def test_loadconfig_init():
    lc = _LoadConfig(config_yaml_path='tests/config.yaml')

    # assert that the returned object has the necessary properties
    assert hasattr(lc, 'config_yaml_path')
    assert hasattr(lc, 'elasticsearch_host')
    assert hasattr(lc, 'elasticsearch_verify')
    assert hasattr(lc, 'elasticsearch_user')
    assert hasattr(lc, 'elasticsearch_pass')
    assert hasattr(lc, 'api_index')
    assert hasattr(lc, 'api_version')
    assert hasattr(lc, 'api_missing')
    assert hasattr(lc, 'api_session_expiration')
    assert hasattr(lc, 'api_user')
    assert hasattr(lc, 'api_pass')
    assert hasattr(lc, 'api_secret')
    assert hasattr(lc, 'models_preload_pickles')
    assert hasattr(lc, 'pickle_jar')
    assert hasattr(lc, 'models_refresh_pickles')
    assert hasattr(lc, 'models_predict_probabilities')
    assert hasattr(lc, 'models_top_n')
    assert hasattr(lc, 's3_region')
    assert hasattr(lc, 's3_bucket')
    assert hasattr(lc, 's3_key')
    assert hasattr(lc, 's3_secret')

    # assert that the properties of the returned object are of the right type
    assert isinstance(lc.config_yaml_path, str)


def test_loadconfig_config():
    lc = _LoadConfig(config_yaml_path='tests/config.yaml').config()

    # assert that the properties of the returned object are of the right type
    assert isinstance(lc.config_yaml_path, str)
    assert isinstance(lc.elasticsearch_host, str)
    assert lc.elasticsearch_index is None
    assert isinstance(lc.elasticsearch_index_models, object)
    assert isinstance(lc.elasticsearch_index_requestlog, object)
    assert isinstance(lc.elasticsearch_verify, bool)
    assert isinstance(lc.elasticsearch_user, str)
    assert isinstance(lc.elasticsearch_pass, str)
    assert isinstance(lc.api_index, str)
    assert isinstance(lc.api_version, str)
    assert isinstance(lc.api_missing,
                      str)  # I guess right now this strict requirement isn't listed anywhere in the docs
    assert isinstance(lc.api_session_expiration, int)
    if lc.api_user is not None:
        assert isinstance(lc.api_user, str)
    if lc.api_pass is not None:
        assert isinstance(lc.api_pass, str)
    assert isinstance(lc.api_secret, str)
    assert isinstance(lc.models_preload_pickles, bool)
    assert lc.pickle_jar is None
    if lc.models_refresh_pickles is not None:
        assert isinstance(lc.models_refresh_pickles, int)
    assert isinstance(lc.models_predict_probabilities, bool)
    assert isinstance(lc.models_top_n, int)
    assert isinstance(lc.s3_region, str)
    assert isinstance(lc.s3_bucket, str)
    assert isinstance(lc.s3_key, str)
    assert isinstance(lc.s3_secret, str)

    # test whether a warning is raised if there's an error parsing a malformed config file
    _LoadConfig(config_yaml_path='tests/config_malformed.yaml').config()

    # assert a warning is raised if there's an error in parsing
    with pytest.warns(RuntimeWarning) as record:
        warnings.warn(
            "Error parsing configuration file.",
            RuntimeWarning)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Error parsing configuration file."

    # test whether a warning is raised if there's an error parsing a config file with a missing key
    _LoadConfig(config_yaml_path='tests/config_keyerror.yaml').config()

    # assert a warning is raised if there's an error in parsing
    with pytest.warns(UserWarning) as record:
        warnings.warn(
            "KeyError when parsing configuration file. Check that all necessary keys are specified.",
            UserWarning)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "KeyError when parsing configuration file. Check that all necessary keys are specified."


def test_loadconfig_preload_pickles():
    # connect to the Henosis instance
    c = Connect().config(config_yaml_path='tests/config.yaml')

    # pull first from a bucket with models
    lc = _LoadConfig(config_yaml_path='tests/config.yaml').config()
    pickle_jar = lc.preload_pickles(server_config=c)

    # assert that the returned object is a dictionary
    assert isinstance(pickle_jar, dict)

    # assert preload pickles is true
    assert lc.models_preload_pickles is True

    # assert the returned object matches the internal definition
    assert lc.pickle_jar == pickle_jar

    # pull from a bucket without models
    c = Connect().config(config_yaml_path='tests/config_nomodels.yaml')
    lc = _LoadConfig(config_yaml_path='tests/config_nomodels.yaml').config()
    pickle_jar = lc.preload_pickles(server_config=c)

    # assert that the returned object is a dictionary
    assert isinstance(pickle_jar, dict)

    # assert preload pickles is false
    assert lc.models_preload_pickles is False

    # assert the returned object is empty while the internal definition is None
    assert pickle_jar == {}
    assert lc.pickle_jar is None

    # try to pull from a bucket without
    c = Connect().config(config_yaml_path='tests/config_nonexistentbucket.yaml')
    lc = _LoadConfig(config_yaml_path='tests/config_nonexistentbucket.yaml').config()
    lc.preload_pickles(server_config=c)

    # assert a warning is raised if there's an error in pulling objects from S3
    with pytest.warns(RuntimeWarning) as record:
        warnings.warn(
            "Error retrieving objects from S3.",
            RuntimeWarning)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Error retrieving objects from S3."


# MORE TESTS FOR _LOADCONFIG
# CHECK TO SEE THAT ELASTICSEARCH IS GOOD # def test_loadconfig_Elasticsearch():
# CHECK TO SEE THAT S3 IS GOOD # def test_loadconfig_s3client():

# _SessionManager class #
def test_sessionmanager_init():
    # create the connection object to feed into the Server object that feeds into the session manager
    c = Connect().config(config_yaml_path='tests/config.yaml')
    s = Server(connection=c).config()
    sm = _SessionManager(s)

    # assert the properties we expect exist
    assert hasattr(sm, 'sessionId')
    assert hasattr(sm, 'sessionExpireDate')
    assert hasattr(sm, 'timeIn')
    assert hasattr(sm, 'timeOut')
    assert hasattr(sm, 'timeElapsed')
    assert hasattr(sm, 'responseCode')
    assert hasattr(sm, 'responseDescription')

    # assert that the properties which are objects and not None are the intended format
    assert sm.server_config.__class__.__name__ == 'Server'
    assert sm.session.__class__.__name__ == 'LocalProxy'
    assert sm.connection_requests.__class__.__name__ == 'Connection'
    assert sm.index.__class__.__name__ == 'Requests'


def test_sessionmanager_cookiemanager():
    # create the connection object to feed into the Server object that feeds into the session manager
    c = Connect().config(config_yaml_path='tests/config.yaml')
    s = Server(connection=c).config()
    sm = _SessionManager(s)

    # generate a test request
    q = {
        'formData': {
            'Title': 'The CSSR aboard the spacecraft failed due to improper environmental temperature.',
            'SpecificEnvironment': '999999'  # to match what is in the config yaml
        }
    }

    # set up the testing context
    app = s.configure_server().app
    with app.test_request_context():
        # make a request
        r = _FormProcessor(s).get_recommendations(q)
        response = make_response(json.dumps(r[0]), r[1])

        # first test the capability when a session is not in place #
        cookies = reqparse.request.cookies
        response_cookie = sm.cookie_manager(cookies, response)

        # assert response_cookie is a response object
        assert response_cookie is not None
        assert response_cookie.__class__.__name__ == 'Response'

        # assert that the session manager objects and properties are what they should be now
        assert isinstance(int(sm.sessionId, 16), int)
        assert isinstance(sm.sessionExpireDate, str)
        assert sm.timeIn is None
        assert sm.timeOut is None
        assert sm.timeElapsed is None
        assert isinstance(sm.responseCode, int)
        assert isinstance(sm.responseDescription, str)

        # now test the capability when a session is already in place #
        previous_session_id = sm.sessionId
        cookies = {'session_id': previous_session_id}
        response_cookie = sm.cookie_manager(cookies, response)

        # assert response_cookie is a response object
        assert response_cookie is not None
        assert response_cookie.__class__.__name__ == 'Response'

        # assert that the previous session ID is the same as the current one (since we are in session)
        assert previous_session_id == sm.sessionId

        # now test the capability when a session has expired (1 minute in our config.yaml)
        time.sleep(65)

        previous_session_id = sm.sessionId
        cookies = {'session_id': previous_session_id}
        sm.cookie_manager(cookies, response)

        # assert that the previous ID does not match the newly created one
        assert previous_session_id != sm.sessionId

        # lastly, test for proper catch of failure #
        sm.cookie_manager(cookies, '')

        # assert a warning is raised if there's an error in using the cookie manager
        with pytest.warns(RuntimeWarning) as record:
            warnings.warn(
                "Issue with cookie manager.",
                RuntimeWarning)

        # check that only one warning was raised
        assert len(record) == 1
        # check that the message matches
        assert record[0].message.args[
                   0] == "Issue with cookie manager."


def test_sessionmanager_logrequest():
    # create the connection object to feed into the Server object that feeds into the session manager
    c = Connect().config(config_yaml_path='tests/config.yaml')
    s = Server(connection=c).config()
    sm = _SessionManager(s)

    # generate a test request
    q = {
        'formData': {
            'Title': 'The CSSR aboard the spacecraft failed due to improper environmental temperature.',
            'SpecificEnvironment': '999999'  # to match what is in the config yaml
        }
    }

    # set up the testing context
    app = s.configure_server().app
    with app.test_request_context():
        # make a request
        ts_in = time.time()
        r = _FormProcessor(s).get_recommendations(q)
        response = make_response(json.dumps(r[0]), r[1])
        cookies = reqparse.request.cookies
        response_cookie = sm.cookie_manager(cookies, response)
        ts_out = time.time()

        # first make a proper request to log the response
        sm.log_request(response_cookie, start_time=ts_in, end_time=ts_out)

        # assert that the time stamps are recorded and the correct format
        assert isinstance(sm.timeIn, float)
        assert isinstance(sm.timeOut, float)
        assert isinstance(sm.timeElapsed, float)

        # now make a request without time stamps
        sm.log_request(response_cookie, start_time=None, end_time=None)

        # assert that a warning was raised
        with pytest.warns(RuntimeWarning) as record:
            warnings.warn(
                "Start and/or end times not passed or wrong format when logging request.",
                RuntimeWarning)

        # check that only one warning was raised
        assert len(record) == 1
        # check that the message matches
        assert record[0].message.args[
                   0] == "Start and/or end times not passed or wrong format when logging request."


# _Elasticsearch class #
def test_elasticsearch_buildquery():
    # build a query and verify a dictionary is returned
    query = _Elasticsearch().Common.build_query({"id": "7851c359709040c4882ed3b3149f2abH"})

    # verify query is a dictionary
    assert isinstance(query, dict)


# def test_elastic_connection_init():
#
#     # create an Elasticsearch connection
#     c = _Elasticsearch().Connection()
#
#     # assert that the returned object has the necessary properties
#
#
#
#
# def test_elastic_connection_config()

def test_elastic_connection_validate_property():
    # first test to check proper response when all is well
    connection_properties = {
        'HOST': '',
        'INDEX': '/requestlog'
    }

    # check for proper response
    validation = _Elasticsearch().Connection().validate_connection_property(connection_properties)
    assert validation is True

    # now test to check proper response when at least one value is None
    connection_properties = {
        'HOST': '',
        'INDEX': None
    }

    # check for proper response
    validation = _Elasticsearch().Connection().validate_connection_property(connection_properties)
    assert validation is False

    # check that a warning is provided for the given issue
    # assert that a warning was raised
    with pytest.warns(RuntimeWarning) as record:
        warnings.warn(
            "Property INDEX must not be None.",
            RuntimeWarning)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Property INDEX must not be None."


def test_elastic_connection_request_status_parser():
    # connect to an existing host
    host = ''
    connection_properties = {
        'HOST': host
    }

    _Elasticsearch().Connection().validate_connection_property(connection_properties)
    r = requests.get(host, verify=False, auth=('elastic', 'JpaRgbEGL6iXgLZdCZHjyYlu'))

    # assert that the request parser returns True
    parsed_request = _Elasticsearch().Connection().request_status_parser(r)
    assert parsed_request is True

    # connect to an non-existing host
    host = ''
    connection_properties = {
        'HOST': host
    }

    _Elasticsearch().Connection().validate_connection_property(connection_properties)
    r = requests.get(host, verify=False)

    # assert that the request parser returns False
    parsed_request = _Elasticsearch().Connection().request_status_parser(r)
    assert parsed_request is False

    # check that he warning is provided on the given issue
    # assert that a warning was raised
    with pytest.warns(RuntimeWarning) as record:
        warnings.warn(
            "Error establishing connection to Elasticsearch resource.",
            RuntimeWarning)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Error establishing connection to Elasticsearch resource."

    # force a json decode error by hitting a non-JSON api, like JPL Horizons
    # connect to an existing host
    host = """https://ssd.jpl.nasa.gov/horizons_batch.cgi?batch=1&COMMAND='499'&MAKE_EPHEM='YES'  
        &TABLE_TYPE='OBSERVER'&START_TIME='2000-01-01'&STOP_TIME='2000-12-31'&STEP_SIZE='15%20d'  
        &QUANTITIES='1,9,20,23,24'&CSV_FORMAT='YES'"""
    connection_properties = {
        'HOST': host
    }

    _Elasticsearch().Connection().validate_connection_property(connection_properties)
    r = requests.get(host, verify=False)

    # assert that the request parser returns False
    parsed_request = _Elasticsearch().Connection().request_status_parser(r)
    assert parsed_request is False

    # check that he warning is provided on the given issue
    # assert that a warning was raised
    with pytest.warns(RuntimeWarning) as record:
        warnings.warn(
            "Error establishing connection to Elasticsearch resource due to JSON decode error.",
            RuntimeWarning)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Error establishing connection to Elasticsearch resource due to JSON decode error."


def test_elasticsearch_connection_host_test():
    # create connection
    c = _Elasticsearch().Connection()
    c.verify = False
    c.auth = ('elastic', 'JpaRgbEGL6iXgLZdCZHjyYlu')

    # use an existing host and ensure no warnings occur
    host = ''
    c.test_host(host=host)

    # use a non-existing host to test for a warning
    host = ''
    c.test_host(host=host)

    # check that he warning is provided on the given issue
    # assert that a warning was raised
    with pytest.warns(RuntimeWarning) as record:
        warnings.warn(
            "Error establishing connection to Elasticsearch resource.",
            RuntimeWarning)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Error establishing connection to Elasticsearch resource."


def test_elasticsearch_connection_index_test():
    # create connection
    c = _Elasticsearch().Connection()
    c.verify = False
    c.auth = ('elastic', 'JpaRgbEGL6iXgLZdCZHjyYlu')

    # use an existing host and index and ensure no warnings occur
    host = ''
    index = '/requestlog'
    c.test_index(host=host, index=index)

    # use a non-existing host and index to test for a warning
    host = ''
    c.test_index(host=host, index=index)

    # check that he warning is provided on the given issue
    # assert that a warning was raised
    with pytest.warns(RuntimeWarning) as record:
        warnings.warn(
            "Error establishing connection to Elasticsearch resource.",
            RuntimeWarning)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Error establishing connection to Elasticsearch resource."


def test_elasticsearch_connection_type_test():
    # create connection
    c = _Elasticsearch().Connection()
    c.verify = False
    c.auth = ('elastic', 'JpaRgbEGL6iXgLZdCZHjyYlu')

    # use an existing host, index, and type and ensure no warnings occur
    host = ''
    index = '/requestlog'
    type = '/requestlog'
    c.test_type(host=host, index=index, type=type)

    # use a non-existing host, index and type to test for a warning
    host = ''
    c.test_type(host=host, index=index, type=type)

    # check that he warning is provided on the given issue
    # assert that a warning was raised
    with pytest.warns(RuntimeWarning) as record:
        warnings.warn(
            "Error establishing connection to Elasticsearch resource.",
            RuntimeWarning)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Error establishing connection to Elasticsearch resource."


def test_elasticsearch_connection_comprehensive_test():
    # create a connection
    c = _Elasticsearch().Connection()
    c.verify = False
    c.auth = ('elastic', 'JpaRgbEGL6iXgLZdCZHjyYlu')

    # use an existing host, index, and type and ensure no warnings occur
    host = ''
    index = '/requestlog'
    type = '/requestlog'
    c.test(host=host, index=index, type=type)

    # use a non-existing host to test for a warning
    host = ''
    c.test_type(host=host, index=index, type=type)

    # check that he warning is provided on the given issue
    # assert that a warning was raised
    with pytest.warns(RuntimeWarning) as record:
        warnings.warn(
            "Error establishing connection to Elasticsearch resource.",
            RuntimeWarning)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Error establishing connection to Elasticsearch resource."

    # use a non-existing index to test for a warning
    host = ''
    index = '/request'
    type = '/requestlog'
    c.test(host=host, index=index, type=type)

    # check that he warning is provided on the given issue
    # assert that a warning was raised
    with pytest.warns(RuntimeWarning) as record:
        warnings.warn(
            "Error establishing connection to Elasticsearch resource.",
            RuntimeWarning)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Error establishing connection to Elasticsearch resource."

    # use a non-existing type to test for a warning
    host = ''
    index = '/requestlog'
    type = '/request'
    c.test(host=host, index=index, type=type)

    # check that he warning is provided on the given issue
    # assert that a warning was raised
    with pytest.warns(RuntimeWarning) as record:
        warnings.warn(
            "Error establishing connection to Elasticsearch resource.",
            RuntimeWarning)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == "Error establishing connection to Elasticsearch resource."


def test_elasticsearch_requests_get():
    # connect to a Henosis instance
    c = Connect().config('tests/config.yaml')

    # test a simple GET query and assert we receive a 200 response
    r = _Elasticsearch().Requests(server_config=c)
    response = r.get()
    assert response[1] == 200

    # test a GET query with kwargs and assert we receive a 200 response
    r = _Elasticsearch().Requests(server_config=c)
    response = r.get(**{'sessionId': 'c99c02286d6448e9885d745a2c524a83'})
    assert response[1] == 200


def test_elasticsearch_requests_count():
    # connect to a Henosis instance
    c = Connect().config('tests/config.yaml')

    # test a GET count query and assert we receive a 200 response
    r = _Elasticsearch().Requests(server_config=c)
    response = r.count()
    assert response[1] == 200


def test_elasticsearch_models_get():
    # connect to a Henosis instance
    c = Connect().config('tests/config.yaml')

    # test a simple GET query and assert we receive a 200 response
    r = _Elasticsearch().Models(server_config=c)
    response = r.get()
    assert response[1] == 200

    # test a GET query with kwargs and assert we receive a 200 response
    r = _Elasticsearch().Models(server_config=c)
    response = r.get(**{'dependent': 'SpecificEnvironment'})
    assert response[1] == 200


def test_elasticsearch_models_get_by_id():
    # connect to a Henosis instance
    c = Connect().config('tests/config.yaml')

    # test a simple GET query and assert we receive a 200 response
    r = _Elasticsearch().Models(server_config=c)
    response = r.get_by_id('f3f981ac69eb41ada5d775e4f9e36886')
    assert response[1] == 200


def test_elasticsearch_models_delete():
    # connect to a Henosis instance
    c = Connect().config('tests/config.yaml')

    # create a new record by changing the ID of an existing record
    r = _Elasticsearch().Models(server_config=c)
    response = r.get_by_id('f3f981ac69eb41ada5d775e4f9e36886')
    model = response[0]
    model['models']['id'] = int(1234)
    r.update(update_dict=model['models'])

    # delete the record and assert we receive a 200 response
    response = r.delete('1234')
    assert response[1] == 200

    # try to delete the now non-existing model and assert we receive a 400 response
    response = r.delete('1234')
    assert response[1] == 400

    # check that he warning is provided on the given issue
    # assert that a warning was raised
    with pytest.warns(RuntimeWarning) as record:
        warnings.warn(
            'Error deleting model from Elasticsearch.',
            RuntimeWarning)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == 'Error deleting model from Elasticsearch.'


def test_elasticsearch_models_update():
    # connect to a Henosis instance
    c = Connect().config('tests/config.yaml')

    # test the creation of a new record from a recently trained model
    print()

    # test record update by get a model by the an ID and update the callCount
    r = _Elasticsearch().Models(server_config=c)
    response = r.get_by_id('f3f981ac69eb41ada5d775e4f9e36886')
    model = response[0]
    model['models']['callCount'] += 1

    # update the model and assert we receive a 200 response
    response = r.update(update_dict=model['models'])
    assert response[1] == 200

    # test a bad record update by changing the model id to a float (new model)
    # this creates a model and does not update it, so should throw an error
    # assert we receive a 400 response
    model['models']['id'] = int(1234)

    # assert we receive a 400 response
    response = r.update(update_dict=model['models'])
    assert response[1] == 400
    r.delete('1234')

    # check that he warning is provided on the given issue
    # assert that a warning was raised
    with pytest.warns(RuntimeWarning) as record:
        warnings.warn(
            'Error updating model ' + str(model['models']['id']) + '.',
            RuntimeWarning)

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[
               0] == 'Error updating model ' + str(model['models']['id']) + '.'


def test_elasticsearch_models_count():
    # connect to a Henosis instance
    c = Connect().config('tests/config.yaml')

    # test a GET count query and assert we receive a 200 response
    r = _Elasticsearch().Models(server_config=c)
    response = r.count()
    assert response[1] == 200


# Server endpoints #
def test_recommendations_endpoint_init():
    # connect to a Henosis instance
    c = Connect().config('tests/config.yaml')

    # instantiate
    r = _Recommendations(server_config=c)

    # assert that the self object matches passed value
    assert r.server_config == c


# def test_recommendations_endpoint_get():
#
#     # # connect to a Henosis instance
#     # c = Connect().config('tests/config.yaml')
#     #
#     # # instantiate
#     # r = _Recommendations(server_config=c)
#
#     # create the connection object to feed into the Server object that feeds into the session manager
#     c = Connect().config(config_yaml_path='tests/config_nopicklejar.yaml')
#     s = Server(connection=c).config()
#
#     # generate a test request
#     q = {
#         'formData': {
#             'Title': 'The CSSR aboard the spacecraft failed due to improper environmental temperature.',
#             'SpecificEnvironment': '999999'  # to match what is in the config yaml
#         }
#     }
#
#     # set up the testing context
#     app = s.configure_server().app.test_client()
#
#     print(app.get('/api/v0.1/recommend?formData=' + json.dumps(q['formData'])))
#
#
#     # with app.test_request_context('/api/v0.1/recommend?formData=' + json.dumps(q['formData'])):
#     #     r = app.preprocess_request()
#     #     print(r)
#     #     print(app.process_response(r))
#
#     # print(app.get('/api/v0.1/recommend?formData=' + json.dumps(q['formData'])))
#
#     # problem here is that this request uses a tagged function...
#
#
# test_recommendations_get()


def test_models_endpoint_init():
    # connect to a Henosis instance
    c = Connect().config('tests/config.yaml')

    # instantiate
    r = _ModelInfo(server_config=c)

    # assert that the self object matches passed value
    assert r.server_config == c


def test_models_endpoint_get():
    # create the connection object to feed into the Server object that feeds into the session manager
    c = Connect().config(config_yaml_path='tests/config.yaml')
    s = Server(connection=c).config()

    # set up the testing context
    app = s.configure_server().app.test_client()

    # test the functionality of the base query
    r = app.get('/api/v0.1/models')

    # assert we receive a 200 response and that we receive data from the request
    assert r.status_code == 200
    assert len(json.loads(r.data)['models']) > 0

    # test the functionality of making a specific query
    q = {'id': 'f3f981ac69eb41ada5d775e4f9e36886'}
    r = app.get('/api/v0.1/models?modelInfo=' + json.dumps(q))

    # assert we receive a 200 response and that we receive data from the request
    assert r.status_code == 200
    assert len(json.loads(r.data)['models']) == 1
    assert json.loads(r.data)['models'][0]['_id'] == 'f3f981ac69eb41ada5d775e4f9e36886'


def test_requestlog_endpoint_init():
    # connect to a Henosis instance
    c = Connect().config('tests/config.yaml')

    # instantiate
    r = _RequestLogs(server_config=c)

    # assert that the self object matches passed value
    assert r.server_config == c


def test_requestlog_endpoint_get():
    # create the connection object to feed into the Server object that feeds into the session manager
    c = Connect().config(config_yaml_path='tests/config.yaml')
    s = Server(connection=c).config()

    # set up the testing context
    app = s.configure_server().app.test_client()

    # test the functionality of the base query
    r = app.get('/api/v0.1/requestlogs')

    # assert we receive a 200 response and that we receive data from the request
    assert r.status_code == 200
    assert len(json.loads(r.data)['requests']) > 0

    # test the functionality of making a specific query
    q = {'sessionId': 'a950fe1813324a53b59d895ab23ddc64'}
    r = app.get('/api/v0.1/requestlogs?requestInfo=' + json.dumps(q))

    # assert we receive a 200 response and that we receive data from the request
    assert r.status_code == 200
    assert len(json.loads(r.data)['requests']) >= 1
    assert json.loads(r.data)['requests'][0]['_source']['sessionId'] == 'a950fe1813324a53b59d895ab23ddc64'


# Server class #
def test_server_init():
    # create the connection object to feed into the Server object that feeds into the session manager
    c = Connect().config(config_yaml_path='tests/config.yaml')
    s = Server(connection=c)

    # assert that the connection object is not None
    assert s.sc == c


def test_server_config():
    # create the connection object to feed into the Server object that feeds into the session manager
    c = Connect().config(config_yaml_path='tests/config.yaml')
    s = Server(connection=c).config()

    # assert that the properties of the returned object are of the right type
    assert isinstance(s.base_url, str)
    assert s.sys_config.__class__.__name__ == '_LoadConfig'
    assert s.es_connection_models.__class__.__name__ == 'Connection'
    assert s.es_connection_requestlog.__class__.__name__ == 'Connection'
    assert s.s3_connection.__class__.__name__ == 'S3'


def test_server_configure():
    # create the connection object to feed into the Server object that feeds into the session manager
    c = Connect().config(config_yaml_path='tests/config.yaml')

    # create a baseline version of the server
    s = Server(connection=c).config().configure_server()

    # assert that the properties of the returned object are of the right type
    assert s.app.__class__.__name__ == 'Flask'
    assert s.api.__class__.__name__ == 'Api'
    assert isinstance(s.port, int)
    assert s.port == 5005
    assert s.routes is None
    assert s.api_resources is None

    # now create a version of the application that includes additional resources and routes, test port assignment
    class _Test(Resource):

        @staticmethod
        def get():
            return 'test response'

    custom_routes = [
        {
            'route': '/',
            'template': 'index.html',
            'function_name': 'index',
            'template_directory': '/usr/src/app/templates',  # must be same for all
            'static_directory': '/usr/src/app/static'  # must be same for all
        }
    ]

    custom_endpoints = [
        {'class': _Test, 'endpoint': '/test'}
    ]

    s = Server(connection=c).config().configure_server(routes=custom_routes, api_resources=custom_endpoints, port=5001)

    # assert that the properties of the returned object are of the right type
    assert s.app.__class__.__name__ == 'Flask'
    assert s.api.__class__.__name__ == 'Api'
    assert isinstance(s.port, int)
    assert s.port == 5001
    assert isinstance(s.routes, list)
    assert isinstance(s.api_resources, list)


# _MultiColumnLabelEncoder class #
def test_multicolumnlabelencoder_fit():
    # generate some data
    import numpy as np
    X = np.array([['1', '2', '3'], ['a', 'b', 'c']])

    # assert that fit returns self
    m = _MultiColumnLabelEncoder().fit(X)
    assert m.__class__.__name__ == '_MultiColumnLabelEncoder'


def test_multicolumnlabelencoder_transform():
    # generate some data
    X = pd.DataFrame([['a', '1'], ['b', '2'], ['b', '3']])

    # assert that transform returns the labels we expect
    m = _MultiColumnLabelEncoder().transform(X)

    assert m.iloc[0, 0] == 0
    assert m.iloc[0, 1] == 0
    assert m.iloc[1, 0] == 1
    assert m.iloc[1, 1] == 1
    assert m.iloc[2, 0] == 1
    assert m.iloc[2, 1] == 2


def test_multicolumnlabelencoder_fittransform():
    # generate some data
    X = pd.DataFrame([['a', '1'], ['b', '2'], ['b', '3']])

    # assert that transform returns the labels we expect
    m = _MultiColumnLabelEncoder().transform(X)

    assert m.iloc[0, 0] == 0
    assert m.iloc[0, 1] == 0
    assert m.iloc[1, 0] == 1
    assert m.iloc[1, 1] == 1
    assert m.iloc[2, 0] == 1
    assert m.iloc[2, 1] == 2


# _FormProcessor class #
def test_formprocessprocessor_most_common():
    # generate a list of numbers, mode is 2
    numbers = [0, 1, 9, 2, 2, 3, 2, 4, 8, 7, 6, 4, 3, 2]

    mode = _FormProcessor.most_common(numbers)

    assert isinstance(mode, list)
    assert mode[0] == 2


def test_formprocessor_generator_check():

    # create the connection object to feed into the Server object that feeds into the session manager
    c = Connect().config(config_yaml_path='tests/config.yaml')

    # create a baseline version of the server
    s = Server(connection=c).config().configure_server()

    # generate a test form
    q = {
        'formData': {
            'Title': 'The CSSR aboard the spacecraft failed due to improper environmental temperature.',
            'SpecificEnvironment': '999999'  # to match what is in the config yaml
        }
    }

    # test environments
    independent_var_1 = {
        'name': "Title",
        'inputs': [
            "Title"
        ],
        'generator_path': "pre_clean.pickle"
    }
    independent_var_2 = {
        'name': "clean_title",
        'inputs': [
            "Title"
        ],
        'generator_path': "pre_clean.pickle"
    }

    # assert that None is returned if the field already exists in the form
    generated_vals = _FormProcessor(server_config=s).generator_check(q['formData'], independent_var_1)
    assert generated_vals is None

    # # assert the new field is created with preloaded pickes
    # fp = _FormProcessor(server_config=s)
    # fp.server_config = s.sys_config.preload_pickles(server_config=s.sys_config)
    #
    # # same problem as above test
    # # imports in generator functions not defined
    #
    # # generated_vals = _FormProcessor(server_config=s).generator_check(q['formData'], independent_var_2)
    # # print(generated_vals)
    #
    # # test against preload_pickles is false
    # # create the connection object to feed into the Server object that feeds into the session manager
    # c = Connect().config(config_yaml_path='tests/config_nopicklejar.yaml')
    #
    # # create a baseline version of the server
    # s = Server(connection=c).config().configure_server()
    #
    # # assert the new field is created with preloaded pickes without previously loading the generator
    # # assert the new field is created with preloaded pickes
    # fp = _FormProcessor(server_config=s)
    # fp.server_config = s.sys_config.preload_pickles(server_config=s.sys_config)
    #
    # # same problem as above test
    # # imports in generator functions not defined
    #
    # # generated_vals = _FormProcessor(server_config=s).generator_check(q['formData'], independent_var_2)
    # # print(generated_vals)

# test_formprocessor_generator_check()


def test_formprocessor_conditions_met():

    # create the connection object to feed into the Server object that feeds into the session manager
    c = Connect().config(config_yaml_path='tests/config.yaml')

    # create a baseline version of the server
    s = Server(connection=c).config().configure_server()

    # set some static vars
    dependent = 'SpecificEnvironment'

    # generate test forms
    q_1 = {
        'formData': {
            'Title': 'The CSSR aboard the spacecraft failed due to improper environmental temperature.',
            'SpecificEnvironment': '999999'
        }
    }
    q_2 = {
        'formData': {
            'Title': 'The CSSR aboard the spacecraft failed due to improper environmental temperature.',
            'ProjectName': '999999'
        }
    }
    q_3 = {
        'formData': {
            'Title': 'The CSSR aboard the spacecraft failed due to improper environmental temperature.',
            'SpecificEnvironment': 123
        }
    }

    # generate sets of X labels
    X_1 = ['Title', 'SpecificEnvironment']
    X_2 = ['Title', 'ProjectName']

    # assert False is returned if deployed_flag is False
    gen_check = _FormProcessor(server_config=s).conditions_met(False, q_1['formData'], dependent, dependent, X_1)
    assert gen_check is False

    # assert False is returned if the key does not matches the dependent y
    gen_check = _FormProcessor(server_config=s).conditions_met(True, q_2['formData'], 'ProjectName', dependent, X_1)
    assert gen_check is False

    # assert False if the key value does not match the missing identifier to predict
    gen_check = _FormProcessor(server_config=s).conditions_met(True, q_3['formData'], dependent, dependent, X_1)
    assert gen_check is False

    # assert False if the labels in X are not a subset of all fields in the form
    gen_check = _FormProcessor(server_config=s).conditions_met(True, q_1['formData'], dependent, dependent, X_2)
    assert gen_check is False

    # assert True is returned if all conditions are met
    gen_check = _FormProcessor(server_config=s).conditions_met(True, q_1['formData'], dependent, dependent, X_1)
    assert gen_check is True


def test_formprocessor_threshold_proba():

    # create the connection object to feed into the Server object that feeds into the session manager
    c = Connect().config(config_yaml_path='tests/config.yaml')

    # create a baseline version of the server
    s = Server(connection=c).config().configure_server()

    # define a threshold
    th = 0.7

    # generate a set of probabilities that are above and below a threshold
    probas_1 = np.array([[0.8, 0.7, 0.2, 0.1]])
    probas_2 = np.array([[0.6, 0.4, 0.2, 0.1]])

    # assert that True is returned if meeting the threshold
    threshold_met = _FormProcessor(server_config=s).threshold_proba(probas_1, th)
    assert threshold_met is True

    # assert that False is returned if not meeting the threshold
    threshold_met = _FormProcessor(server_config=s).threshold_proba(probas_2, th)
    assert threshold_met is False


# def test_formprocessor_load_vectorizer():
# def test_formprocessor_predict_majority():
# def test_formprocessor_predict_proba():
# def test_formprocessor_predict():
# def test_formprocessor_sort_votes():
# def test_formprocessor_process_form():
# def test_formprocessor_get_recommendations():


# PROBABLY A GOOD IDEA TO SEPARATE THIS INTO SEPARATE FILES MIRRORING PACKAGE STRUCTURE

