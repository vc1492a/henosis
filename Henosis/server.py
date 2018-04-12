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
from gevent import monkey
monkey.patch_thread()
monkey.patch_select(aggressive=False)
from collections import OrderedDict
import datetime
from flask import Flask, make_response, render_template
from flask_cors import CORS
from flask_restful import Api, reqparse, request, Resource
from functools import wraps
from gevent.wsgi import WSGIServer
import itertools
import json
import logging
import multiprocessing as mp
import numpy as np
import operator
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import subprocess
import sys
import threading
import time

from Henosis.model import Models
from Henosis.utils import Connect, _ElasticSearch, _SessionManager

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
queue = None

__author__ = 'Valentino Constantinou'
__version__ = '0.0.8'
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

    def conditions_met(self, model_path, deployed_flag, form_data, k, y, X):
        if deployed_flag is True and form_data[k] == self.missing and y == k and set(X) <= set(form_data.keys()):
            if self.server_config.sys_config.models_preload_pickles is True:
                if model_path in self.server_config.sys_config.pickle_jar.keys():
                    return True
                return False
            else:
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
            encoder = self.model_functions._load(encoder_path)
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

    def threshold_proba(self, probas, threshold):
        # grabs the indices of the top N values
        max_probs_ind = np.argpartition(probas[0], -self.top_n)[-self.top_n:]
        max_probs = probas[0][max_probs_ind]
        m = np.max(max_probs)
        logging.info('Max bucket probability: %s', m)
        logging.info('Probability threshold: %s', threshold)
        if np.max(max_probs) > threshold:
            return True
        return False

    def predict_proba(self, model, input_data, k, recommendation_threshold):
        # make the prediction
        p = model.predict_proba(input_data)
        # check for recommendationThreshold compliance
        if self.threshold_proba(p, recommendation_threshold) is True:
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
        else:
            logging.warning('Model predictions for field %s did not meet minimum recommendation (confidence) threshold.', k)

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

    def predict(self, model, input_data, k, recommendation_threshold):
        # make predictions based on config
        if self.predict_probabilities:
            self.predict_proba(model, input_data, k, recommendation_threshold)
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
            if self.conditions_met(m['_source']['modelPath'], m['_source']['deployed'], form_data, k, m['_source']['dependent'], independent_vars):
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
                self.predict(pred_m, input_data, k, m['_source']['recommendationThreshold'])
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
            return {'description': 'Error processing models and their predictions.'}, 400


class ServerThread(threading.Thread):

    def __init__(self, app, port=5005):
        self.port = port
        threading.Thread.__init__(self)
        self.srv = WSGIServer(("", self.port), app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        logging.info('Starting threaded server.')
        self.srv.serve_forever()

    def shutdown(self):
        logging.info('Stopping threaded server.')
        self.srv.stop()


def start_server(app, port=5005):
    global server
    server = ServerThread(app, port=port)
    server.start()


def stop_server():
    global server
    server.shutdown()


class Server:

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

    def config(self, config_yaml_path):
        sc = Connect().config(config_yaml_path, server=True)
        self.config_yaml_path = sc.config_yaml_path
        self.sys_config = sc.sys_config
        self.es_connection_models = sc.sys_config.elasticsearch(sc.sys_config.elasticsearch_index_models)
        self.es_connection_requestlog = sc.sys_config.elasticsearch(sc.sys_config.elasticsearch_index_requestlog)
        self.s3_connection = sc.sys_config.aws_s3()
        self.base_url = sc.sys_config.api_index + sc.sys_config.api_version

        return self

    def reload_pickles(self, start):
        seconds = 60. * self.sys_config.models_refresh_pickles
        t = threading.Timer(seconds, self.reload_pickles, [False])
        t.start()
        logging.info('Pickled objects preloading every %s minutes according to reload schedule.', self.sys_config.models_refresh_pickles)
        global queue
        if start is False:
            logging.info('Restarting Henosis.')
            queue.put("reload pickles")

    def start_queue(self, q):
        # set up the restart queue
        global queue
        queue = q
        # periodically refresh models if specified
        if self.sys_config.models_refresh_pickles is not None and self.sys_config.models_preload_pickles is True:
            self.reload_pickles(True)

    def start_server(self, routes=None, api_resources=None, port=5005):

        self.port = port
        self.routes = routes
        self.api_resources = api_resources

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

        # serve the app
        start_server(app, port=self.port)

    def run(self, routes=None, api_resources=None, port=5005):

        # create a Queue that will be used to trigger a restart
        q = mp.Queue()

        # start the threaded server
        t_server = threading.Thread(target=self.start_server, args=[routes, api_resources, port])
        t_server.start()
        t_restart = threading.Thread(target=self.start_queue, args=[q])
        t_restart.start()

        # watching queue, if there is no call than sleep, otherwise break and restart app
        while True:
            if q.empty():
                time.sleep(1)
            else:
                t_server.join()
                stop_server()
                break

        # restart app using same start arguments
        args = [sys.executable] + [sys.argv[0]]
        subprocess.call(args)




