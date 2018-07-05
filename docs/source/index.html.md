---
title: API Reference

language_tabs: # must be one of https://git.io/vQNgJ
  - shell
  - python

search: true
---

# About

## Introduction

Henosis is a cloud-native, lightweight Python-based recommender framework
that facilitates providing recommendations to users of applications, like this:

<br/>

![PRS Demo Gif](https://i.imgur.com/5mzhJzm.gif)

<br/>

Henosis is in **active development** and is released as alpha software.
Henosis is being developed at the [NASA Jet Propulsion Laboratory](https://jpl.nasa.gov/) (JPL)
using user-driven development (UDD) with generous support from the Office of Safety and
Mission Success (5X).

[![PyPi](https://img.shields.io/badge/pypi-0.0.10-green.svg)](https://pypi.python.org/pypi/Henosis/0.0.10)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## How Does Henosis Work?

This framework brings together model training and testing,
storage and deployment, and querying under one framework that data
scientists and developers can collectively use to provide
recommendations to users. Henosis provides data scientists with a straight-forward interface in
which to train, test, store, and deploy categorical machine learning
models. For developers, it provides an API that can be used
to provide form field recommendations to users of enterprise
applications.

### For Data Scientists

Henosis is intended to work well with data scientists' modeling workflow,
and can be used to create training and test splits, fit models, and deploy
models to a running Henosis instance (see example).

> For Data Scientists

```python
from Henosis.model import Data, Models

# read in data from a csv
d = Data()
d.load(csv_path='file.csv')
print(d.all.head())

# split data
d.test_train_split(
    d.all[X_vars],
    d.all[y_var],
    share_train=0.8
)

# fit a model (use any categorical scikit-learn model)
# this is where you do your "magic"!
m = Models().SKModel(MultinomialNB(alpha=0.25))
m.train(d)
m.test(d)

# print results
print(m.train_results)
print(m.test_results)

# connect to the deployed Henosis instance by referencing a configuration file
from Henosis.utils import Connect
s = Connect().config(config_yaml_path='config.yaml')

# store the model in S3 and Elasticsearch
m.store(
    server_config=s,
    model_path='model_' + y_var + '_1.pickle',
    encoder_path='encoder_' + y_var + '_1.pickle',
    encoder=count_vect
)

# deploy a model for use
m.deploy(
    server_config=s,
    deploy=True
)
```

More detailed information and examples are available in the rest of the documentation.

### For Developers

Running a Henosis instance for serving recommendations requires first setting
up an Elasticsearch index (details in the configuration section) and an
Amazon S3 bucket. Following that, deploying an instance of Henosis for
making recommendations is as easy as placing the following in a Python file (see example).

> For Developers

```python
from Henosis.server import Connect, Server

# run the server
c = Connect().config(config_yaml_path='config.yaml')
s = Server(c).config()
s.run()
```

Once a Henosis instance is running, developers can query for recommendations,
model information, and API request information from the available Henosis API (REST)
endpoints.

### The Internal Structure

There are several classes
that facilitate the interaction between data, scikit-learn models, and a
REST API that provides recommendations and other information.

<br/>
<br/>

<!-- <img src="https://i.imgur.com/EXUh0cx.png" style="max-width: 650px;"/> -->

![Henosis Flowchart](https://i.imgur.com/EXUh0cx.png)

<br/>
<br/>

Henosis classes (bold borders) interface with the data scientist or statistician,
developers querying for recommendations, and trained models. When queried for
recommendations, Henosis references model information in Elasticsearch,
loads appropriate models from AWS S3, and serves recommendations by using
data available in the query (REST API request).

## Requirements

- Python 3.6+ (untested on lower versions) with the following packages:
    - boto3
    - dill
    - Flask
    - Flask-Cors
    - Flask-HTTPAuth
    - Flask-RESTful
    - gevent
    - imbalanced-learn
    - Jinja2
    - jwt
    - gevent
    - numpy
    - pandas
    - PyYAML
    - requests
    - scikit-learn
- A working Amazon Web Services (AWS) S3 bucket, along with:
    - AWS key
    - AWS secret
- A running Elasticsearch 6+ server (untested on lower versions). You'll need to
create an index and specify a mapping (documentation is provided and helpful
scripts are available in the *scripts* directory).

## Motivation

In October 2017, our data science team at the [NASA Jet Propulsion Laboratory](https://jpl.nasa.gov/) (JPL)
was approached by the Office of Safety and Mission Success (5X) to improve
processes for reporting in the Problem Reporting System (PRS). PRS is an internal
tool that allows engineers to submit Problem Failure Reports (PFRs) and Incident
Surprise, Anomaly reports (ISAs), which document pre-launch test failures and
post-launch operational anomalies experienced by spacecraft. These reports not
only serve as a record of past problems but also of past solutions to the problems described.

Despite their value, the reports contained within the PRS are costly to fill out and submit.
With dozens of textual, categorical, and other inputs in the forms, the PFRs and ISAs
draw valuable time away from mission staff to the annotation of internal forms — time
better spent with spacecraft operations and mission work. A solution was needed that
would reduce the time needed to file reports in PRS while ensuring ease of use for
users already familiar with the current PRS system, such as a recommendation system
for form fields. What was needed from a data science and IT operations perspective
was a straightforward process to deploy a simple recommendation system for use in
enterprise applications containing categorical form inputs (like dropdown menus).

While the initial effort focused on a single internal use case, Henosis was developed
as a generalized, open-source framework; it is freely available for use and easy to use
in many different use cases.

# Configuration

Henosis is configured using a locally stored **config.yaml** file. This configuration
file contains parameters associated with Henosis API configuration, AWS S3 bucket
specification, Elasticsearch configuration and Henosis model deployment settings,
and is referenced when storing and deploying models as well as
when deploying a Henosis instance for use in testing or production.

## Henosis Configuration

The **config.yaml** file is not required for model training and testing, but *is*
required for model storage and deployment as well as running a Henosis instance.

> config.yaml

```yaml

# ElasticSearch
elasticsearch:
  host: 'https://<host>:<port>'
  verify: False
  user: '<username>'
  pw: '<password>'
  models:
    index: '/model'
  request_log:
    index: '/requestlog'

# AWS S3
aws_s3:
  region: '<region>'
  bucket: '<bucket>'
  key: '<aws_key>'
  secret: '<aws_secret>'

# Api
api:
  host: 'http://localhost:5005'
  index: '/api'
  version: '/v0.1'
  missing: '999999'
  session_expiration: 10
  user: '<username>'
  pw: '<password>'

# Models
models:
  preload_pickles: True
  refresh_pickles: 1440 # in minutes
  predict_probabilities: True
  top_n: 3

```

Each top level of the config file describes the specifications for each of the resources
used or available from Henosis. A few notes on some of the specifications:

- elasticsearch
    - host: the endpoint of your Elasticsearch server.
    - verify: tells Henosis whether or not to use SSL verification when making requests to
    the specified elasticsearch index. Set to *True* to use SSL verification.
    - auth: an optional username and password for connecting to Elasticsearch. If
    not using Elasticsearch auth, specify *None*.
    - models / request_log: the names of the indices used in Henosis for model and request
    log storage. *Default parameters recommended.*
- AWS S3:
    - region: the AWS region where your S3 bucket is located.
    - bucket: the name of your AWS S3 bucket.
    - key: your AWS S3 key.
    - secret: your AWS S3 secret.
- api
    - host: the internal host used by Henosis to provide information through its own API.
    - index: the name of your API following the host. *Default parameters recommended.*
    - version: the version of your API.
    - missing: this is the key that informs Henosis whether or not a form field needs a recommendation.
    This value is passed to Henosis in requests for recommendations.
    - session_expiration: defines how long a session lasts when a user starts a new session
    (series of requests), in minutes.
    - auth: simple authentication parameters for the Henosis API. *Use of simple authentication
    for the Henosis API is recommended.*
- models
    - preload_pickles: tells Henosis to store models in S3 in application memory versus
    pulling from S3 for each request. Defaults to True, highly recommended.
    - refresh_pickles: The number of minutes between refreshing models in memory if not set to None and *preload_pickles* set to True.
    - predict_probabilities: if True, uses bagging to average model probabilities for each class.
    Else, provides the majority vote.
    - top_n: the number of recommendations to provide for each form field (defaults to 1 if
    predict_probabilities is set to False).


## Configuring Elasticsearch

Henosis utilizes Elasticsearch to identify which models are available for use,
when models have the appropriate data for making recommendations, and to
keep track of models as they are used to make recommendations for users.
Additionally, a separate Elasticsearch index is used to keep track of requests
made to the Henosis API which can be used to track system and user behavior.

The Elasticsearch indices that store this information **must** first be
created and mapped before storing and deploying models in Henosis and
prior to running a Henosis instance.

> Create the Models Index

```shell
curl -XPUT '<your_host>/model' -H 'Content-Type: application/json' -d '{
    "settings" : {
        "index" : {
            "number_of_shards" : 3,
            "number_of_replicas" : 2
        }
    }
}'
```

> Specify the Models Mapping

```shell
curl -XPUT '<your_host>/model/_mapping/model' -H 'Content-Type: application/json' -d '{
    "model": {
    "properties": {
      "dependent": {
        "type": "keyword"
      },
      "independent": {
        "type": "nested",
        "properties": {
          "name": {
            "type": "keyword"
          },
          "inputs": {
            "type": "text"
          },
          "generatorPath": {
            "type": "text"
          }
        }
      },
      "id": {
        "type": "text"
      },
      "lastTrainedDate": {
        "type": "date"
      },
      "trainRecall": {
        "type": "float"
      },
      "trainPrecision": {
        "type": "float"
      },
      "trainAccuracy": {
        "type": "float"
      },
      "trainF1": {
        "type": "float"
      },
      "trainTime": {
        "type": "float"
      },
      "trainDataBalance": {
        "type": "text"
      },
      "lastTestedDate": {
        "type": "date"
      },
      "testRecall": {
        "type": "float"
      },
      "testPrecision": {
        "type": "float"
      },
      "testAccuracy": {
        "type": "float"
      },
      "testF1": {
        "type": "float"
      },
      "testTime": {
        "type": "float"
      },
      "recommendationThreshold": {
        "type": "float"
      },
      "deployed": {
        "type": "boolean"
      },
      "callCount": {
        "type": "integer"
      },
      "lastCall": {
        "type": "date"
      },
      "modelPath": {
        "type": "text"
      },
      "modelType": {
        "type": "text"
      },
      "encoderPath": {
        "type": "text"
      },
      "encoderType": {
        "type": "text"
      },
    }
  }
}'
```

> Create the RequestLog Index

```shell
curl -XPUT '<your_host>/requestlog' -H 'Content-Type: application/json' -d '{
    "settings" : {
        "index" : {
            "number_of_shards" : 3,
            "number_of_replicas" : 2
        }
    }
}'
```

> Specify the RequestLog Mapping

```shell
curl -XPUT '<your_host>/requestlog/_mapping/requestlog' -H 'Content-Type: application/json' -d '{
    "requestlog": {
    "properties": {
      "sessionId": {
        "type": "keyword"
      },
      "sessionExpireDate": {
        "type": "date"
      },
      "timeIn": {
        "type": "date"
      },
      "timeOut": {
        "type": "date"
      },
      "timeElapsed": {
        "type": "float"
      },
      "missingFields": {
        "type": "keyword"
      },
      "recommendations": {
        "type": "nested",
        "properties": {
          "fieldName": {
            "type": "keyword"
          }
        }
      },
      "modelsQueried": {
        "type": "keyword"
      },
      "modelsUsed": {
        "type": "keyword"
      },
      "modelsWithheld": {
        "type": "keyword"
      },
      "responseStatusCode": {
        "type": "integer"
      },
      "responseDescription": {
        "type": "text"
      }
    }
  }
}'
```


# Modeling

Henosis is being developed using user-driven-development (UDD), and is intended to
work easily in a data scientist or statistician's workflow.
Two classes, *Data* and *Models*, are used to work with your data and
scikit-learn models that form the basis of the form recommendation system.

## Data

The Data object facilitates the loading, saving, and splitting of
data into training and testing splits. The Data object has the following
attributes:

- _all_: The entire dataframe present in the Data object.
- _balance_: Indicates whether or not the data has been upsampled, downsampled, or neither.
- _dependent_: The dependent variable to be used in modeling after data is split.
- _independent_: The independent variables used in modeling. Must be manually specified when using Numpy arrays.
- _X_train_: The independent variable(s) data to be used in model training.
- _X_test_: The independent variable(s) data to be used in model testing.
- _y_train_: The dependent variable data to be used in model training.
- _y_test_: The dependent variable data to be used in model testing.

### Load Data

Henosis supports loading data from local storage, specifically CSV files, and
can be called after loading the *Data* class as an object.

> Load Data

```python
from Henosis.model import Data
d = Data()
d.load(csv_path='data.csv')
# d.all is a Pandas DataFrame
print(d.all.columns.values)
```

### Train / Test Splits

The Henosis Data class can also be used to split your data into training
and testing sets for use in modeling. Upsampling and downsampling is
also an option when splitting your data using [imbalanced-learn](http://contrib.scikit-learn.org/imbalanced-learn/stable/index.html)'s
random oversampling or undersampling methods.

> Split Data for Training and Testing

```python
d.test_train_split(
    df[X_var],
    df[y_var],
    share_train=0.8,
    balance='upsample' # or 'downsample', not required
)
```

Similarly, if using scikit-learn's *CountVectorizer* or *TfidfVectorizer*, a **X_label**
must be specified (see example).

> Split Data with CountVectorizer or TfidfVectorizer

```python
d.test_train_split(
    X_train, # your count or tf-idf matrix
    df[y_var],
    share_train=0.8,
    X_label=['cleanText'] # needed for anything that uses the countvectorizer or tf or tf/idf vectorizers
)
```

### Store Data

Henosis supports storing your dataset to local disk as a CSV file.

> Store Data

```python
d.store(df, csv_path='df.csv')
```

> Accessing Training and Testing Data

```python
print(d.y_train)
print(d.X_train)
```

```python
print(d.y_test)
print(d.X_test)
```

## Models

The Henosis Models object is used jointly with categorical scikit-learn
models to train, test, store, and deploy models for use with form fields.
While a specified **config.yaml** file is not needed to get started with
the train and test methods, storing and deploying models to a running
Henosis instance requires configuration.


**Note:** Henosis supports categorical scikit-learn models, but
does not support regression models or collaborative filtering at this time.
Additionally, it is necessary to ensure that variable names are *identical*
between models (training data) and the form data sent in each request. This
is how Henosis identifies when the data necessary to make a recommendation
is available for each model (e.g. if your variable is named 'projectName' when
training your model, send the form field for project name as 'projectName' in the API request).

### Defining Models

Defining a model in Henosis is as easy as passing a scikit-learn model
into the Henosis models SKModel class. Simply pass any categorical
scikit-learn model into the SKModel class.

> Define Model

```python
from Henosis.model import Models
m = Models().SKModel(MultinomialNB(alpha=1e-10))
```

The SKModel object has the following attributes:

- _call_count_: The number of times the model has been called as a result of an API request.
- _dependent_: The name of the variable being modeled.
- _deployed_: A boolean which indicates whether or not the model is deployed and available for use.
- _encoder_: The scikit-learn encoder used in encoding independent variables (optional).
- _encoder_path_: The file path of the encoder in Amazon S3.
- _encoder_type_: The type of encoder (e.g. TfidfVectorizer).
- _estimator_: The scikit-learn estimator used in modeling.
- _fpr_: The false positive rate for each class in the dependent variable.
- _independent_: The name(s) of the variable(s) used in modeling the dependent.
- _last_call_: The date and time of the last time the model was called as a result of an API request.
- _model_: The scikit-learn model used in modeling.
- _model_path_: The file path of the model in Amazon S3.
- _roc_auc_: The ROC AUC for each class in the dependent variable.
- _test_results_: The overall test results for the model.
- _test_time_: The amount of time needed (in seconds) to test the model.
- _test_timestamp_: The date and time of when the model was last tested.
- _tpr_: The true positive rate for each class in the dependent variable.
- _train_data_balance_: Indicates whether the data was upsampled, downsampled, or neither.
- _train_results_: The overall train results for the model.
- _train_time_: The amount of time needed (in seconds) to test the model.
- _train_timestamp_: The sate and time of when the model was last trained.


### Train

Once training and testing splits have been created using the Data class
and the model has been specified,
training a model is just one function call away. Simply call *m.train(d)*.

> Train Model

```python
m.train(d)
# show the training results
print(json.dumps(m.train_results, indent=4))
```

### Test

Similarly, testing a model is also one function call away by calling
*m.test(d)*.

> Test Model

```python
m.test(d)
# show the testing results
print(json.dumps(m.test_results, indent=4))
```

### Tagging Feature Generators

Henosis has built-in functionality to handle feature generation for
data provided in the request for recommendations, creating new features
from incoming data for use in the available models. For example, a
form may contain raw text that can be used to make predictions for a
particular form field, but this raw text may need to be processed before
use in the models. The feature tagging functionality can be used to
generate new features from the data provided in the recommendation
request itself by associating a Python function with a particular field.

> Define a Function for Generating a New Feature

```python
# a function that processes text, as an example
def clean_text(s):
    import re
    import html
    import string
    import unicodedata
    # use regular expressions to remove html tags from text
    s = re.sub('<[^<]+?>', '', s)  # some strings are of float type?
    # replace HTML entities with proper string encodings
    s = html.unescape(s)
    # if no space after end of sentence punctuation, add a space (for later tokenization)
    s = re.sub(r'(\[.?!,/])', r' ', s)
    # remove punctuation from the text
    translator = str.maketrans("", "", string.punctuation)
    s = str(s.translate(translator))
    # ensure all text is lowercase
    s = " ".join([token.lower() for token in s.split()])
    # replace any unicode characters with ASCII representation.
    s = str(unicodedata.normalize('NFKD', s).encode('ascii', 'ignore'), encoding='utf-8')
    if len(s) == 0:
        s = '<empty>'

    return s
```

> Tag that Function for a Field and Tie to a Model

```python
m.tag_generator(clean_text, 'cleanText', ['title'])
print(m.independent) # to ensure new feature is present
```

After tagging feature generators, Henosis will process incoming data used to
generate new features using the function in the tag and adds that new
field to the data used to provide recommendations. .

**Note:** If using external libraries in your feature generation, this must
be defined as an import *within the function*. In addition, the deployed
Henosis instance must also have the libraries used in any tagged features (functions)
installed on that machine to function properly. This process must be
repeated for each model which uses features generated from the incoming data.

### Store

Storing models with Henosis requires a specified **config.yaml** file with
the correct AWS S3 and Elasticsearch specifications. When calling *m.store()*,
Henosis stores your trained model to AWS S3 and stores information about that
model in Elasticsearch for use when models are queried from the API.

When storing models, the Henosis *Server* class is first called to load
the configuration specified in **config.yaml** into Henosis. Then,
the model can be stored by referencing this configuration.

> Store Model

```python
# NOTE: if this is your first time loading the config yaml, set preload pickles to False in the config file
from Henosis.utils import Connect
s = Connect().config(config_yaml_path='config.yaml')
```

```python
m.store(
    server_config=s,
    model_path='model_variableOne_1.pickle', # make sure not to overwrite an old file
    encoder_path='encoder_variableOne_1.pickle', # make sure not to overwrite an old file
    encoder=count_vect, # only needed for CountVectorizer and TfidfVectorizer data
    override=True # to overwrite old AWS S3 files. Does not overwrite previous Elasticsearch entries.
)
```

*encoder_path* and *encoder* are optional and only needed if variable
encoders are used as part of the training process, such as when
using CountVectorizer or TfidfVectorizer when training models.

### Load Model

It may be useful to load a model from a running Henosis instance to
change its deployment status or work with locally.

> Load Model

```python
# NOTE: if this is your first time loading the config yaml, set preload pickles to False in the config file
from Henosis.utils import Connect
s = Connect().config(config_yaml_path='config.yaml')
```

```python
# now specify a model ID and pass the connection object
m = Models().SKModel().load_model('bbe4830f5eb24a73a299b720eef58ccd', s)
```

Note: Retrain the model to obtain *tpr*, *fpr*, and *roc_auc* attributes.

### Load Generators

It may also be useful to load a function used to generate new data or
process request data for modeling, such as `clean_text` in
*Tagging Feature Generators*. Generator functions associated with a model
are returned in a list.

> Load Generators

```python
# NOTE: if this is your first time loading the config yaml, set preload pickles to False in the config file
from Henosis.utils import Connect
s = Connect().config(config_yaml_path='config.yaml')
```

```python
# now specify a model ID and pass the connection object
generators = Models().SKModel().load_generators('bbe4830f5eb24a73a299b720eef58ccd', s)
```

### Deploy

When deploying models, reference the same *s* object created
as part of the model storage process.

> Deploy Model

```python
m.deploy(
    server_config=s,
    deploy=True,
    recommendation_threshold=0.1
)
```

The *recommendation_threshold* parameter is an optional parameter which
dictates the minimum confidence needed for a model's recommendations to
be used in providing recommendations to users. For example, if set to
0.1, recommendations from the model will only be provided if the maximum
class probability in the *top_n* predictions exceeds that threshold. To
take a model offline, simply call the same function as above
with *deploy=False* instead.

### Delete Model

To delete a model from the system (both from Elasticsearch and Amazon S3),
use the `delete_model` function.
When deleting a model with generator functions, those functions are
checked to see if they are used by other models. Only generator functions
unique to the model are deleted when deleting a model from the system.

> Delete Model

```python
# NOTE: if this is your first time loading the config yaml, set preload pickles to False in the config file
from Henosis.utils import Connect
s = Connect().config(config_yaml_path='config.yaml')
```

```python
Models().SKModel().delete_model('40fe247e909845be897a7c0c395a52c3', s)
```

# Deployment

## Server

The Server object is used by Henosis to reference the specifications in
**config.yaml** and to deploy a Flask application that runs the
Henosis API.

The Server object has the following attributes:

- _base_url_: The base URL for your API.
- _es_connection_requestlog_: Information about the connection to the request log elasticsearch index, as specified in the config file.
- _es_connection_models_: Information about the connection to the models elasticsearch index, as specified in the config file.
- _port_: The port on which the server (application) is running (e.g. 5005).
- _s3_connection_: The connection information for connecting to the specified Amazon S3 bucket.
- _sys_config_: A collection of other attributes related to system configuration based on options specified in the config file.

### Config

The *config* function accepts the path of the **config.yaml** file
and ingests it into Henosis for running the API and storing and deploying
models.

> Load Config

```python
from Henosis.server import Connect, Server
c = Connect().config(config_yaml_path='config.yaml')
s = Server(c).config()
```

### Run

The *run* function uses the previously-loaded configuration to start
a Flask application which runs the Henosis API, which is used for querying
for recommendations.

> Run

```python
s.run(port=5005)
```

Port 5005 is the default port.

### Custom API Endpoints and Routes

Henosis supports the specification of custom API routes and templates that
can be used to retrieve data useful in making recommendations and in testing.
Custom routes are *Flask-RESTful* resources, while custom templates are *Flask* templates.

By passing a *Flask* template, Henosis can be used as a stand-alone system for not
only providing form recommendations but also for providing the user interface for
the form itself. Additionally, by passing custom *Flask-RESTful* resources, models can
use data not available in the form itself for making recommendations in the passed template.
For example, user demographics and past behavioral information are often useful when
making recommendations, but this information may not be available in the form itself.
By passing the custom template with a *Flask-RESTful* resource, that resource can
be used to query for user demographics and pass this information to the recommendations
route in Henosis for obtaining recommendations in any tagged features (functions).

```python
from flask_restful import Resource

class _LDAPDirectory(Resource):

    @staticmethod
    def get():
        return {'userName': 'demostration_user'}


custom_endpoints = [
    {'class': _LDAPDirectory, 'endpoint': '/ldap'}
]


custom_routes = [
    {
        'route': '/',
        'template': 'index.html',
        'function_name': 'index',
        'template_directory': '<path_to_templates>', # must be same for all templates
        'static_directory': '<path_to_static_files>' # must be same for all static files
    }
]

c = Connect().config(config_yaml_path='config.yaml')
s = Server(c).config()
s.run(routes=custom_routes, api_resources=custom_endpoints)
```

# API

Querying Henosis is easy! The Henosis REST API allows software engineers
and others to query a deployed Henosis instance, providing recommendations
based on available data. The Henosis API also allows users to query for
important model information and request logs for debugging or user
analysis.

## Authentication

Henosis supports simple authentication. API authentication is set
by the Henosis administrator, and is specified in the system **config.yaml**. If
using authentication, please pass your authentication credentials as part of
each individual request.

> Some examples of passing auth with your request. In Python:

```python
import json
import requests

r = requests.get('https://<your_host>/api/<your_api_version>/models', auth=('username', 'pass'))
json.dumps(r.json(), indent=4)
```

> Using shell:

```shell
# With shell, you can just pass the correct header with each request
curl -XGET "https://<username>:<password>@<your_host>/api/<your_api_version>/models"
```

## GET Recommendations

This endpoint allows you pass fields that have not yet been recommended
and obtain recommendations for fields to use in your forms. Note that if
the available data is insufficient for making a recommendation using
the models deployed in the Henosis instance, recommendations may
not be returned from your request. Only once all data needed to make
a recommendation is available for a particular model is a recommendation
from that model provided.

There's just a few things to keep in mind:

- Make sure to pass fields that need recommendations using the missing
tag specified in **config.yaml**.
- Make sure to pass field names in the same format as they are specified
in the models, including capitalization (e.g. if the variable project name
was defined as 'projectName' when training, reference that variable as 'projectName'
in your request).

> In Python:

```python
import json
import requests

q = {
  'variableOne': '999999',
  'variableTwo': 'Spaceman',
  'variableThree': '999999',
}

r = requests.get('https://<your_host>/api/<your_api_version>/recommend' + '?formData=' + str(q), auth=('username', 'pass'))
json.dumps(r.json(), indent=4)
```

> Using shell:

```shell
curl -XGET "https://<username>:<password>@<your_host>/api/<your_api_version>/recommend" -d
  "formData={'variableOne': '999999', 'variableTwo': 'Spaceman', 'variableThree': '999999'}"
```

> You may pass something like this:

```python
{
  'variableOne': '999999',
  'variableTwo': 'Spaceman',
  'variableThree': '999999',
}
```

> And receive something like this in return:

```json

{
   "predictions": {
      "variableOne": "Proton Beam",
      "variableTwo": "Spaceman",
      "variableThree": "Liquid Nitrogen Resistant Gloves",
    },
   "modelsUsed": [
      "d440285ac22a4528b5df23ca72c86065",
      "658f5e1310cf42f9a62eed1028179bec"
    ]
   "description": "Model ids and form field predictions from used models."
}
```

### HTTP Request

`GET https://<your_host>/api/<your_api_version>/recommend`

<!-- ### URL Parameters -->

<!-- Parameter | Description -->
<!-- --------- | ----------- -->
<!-- ID | The ID of the kitten to retrieve -->

<!-- ### Query Parameters -->

<!-- Parameter | Default | Description -->
<!-- --------- | ------- | ----------- -->
<!-- include_cats | false | If set to true, the result will also include cats. -->
<!-- available | true | If set to false, the result will include kittens that have already been adopted. -->

<!-- <aside class="success"> -->
<!-- Remember — a happy kitten is an authenticated kitten! -->
<!-- </aside> -->

## GET Models

This endpoint retrieves information about models in the deployed Henosis instance. Depending on the query parameters, all
models may be returned or you may search for models that meet *specific* criteria, such as whether or not a model is
deployed, by the dependent variable, or by the sampling method. Note that ranges, such as models which have been called
more than 5 times, as an example, is currently not supported.


```python
import json
import requests

r = requests.get('https://<your_host>/api/<your_api_version>/models', auth=('username', 'pass'))
json.dumps(r.json(), indent=4)
```

```shell
# With shell, you can just pass the correct header with each request
curl -XGET "https://<username>:<password>@<your_host>/api/<your_api_version>/models"
```

> The above command returns JSON structured like this:

```json
{
    "description": "All models in the Elasticsearch index.",
    "models": [
        {
            "_id": "bb9e7a91c256412ca40f57e27e8e90b6",
            "_index": "models",
            "_score": 1.0,
            "_source": {
                "callCount": 30,
                "dependent": "variableTwo",
                "deployed": true,
                "encoderPath": "encoder_variableTwo_1.pickle",
                "encoderType": "CountVectorizer",
                "id": "bb9e7a91c256412ca40f57e27e8e90b6",
                "independent": [
                    {
                        "generator_path": "clean_text.pickle",
                        "inputs": [
                            "title"
                        ],
                        "name": "cleanText"
                    }
                ],
                "lastTestedDate": "2018-01-18T08:29:09",
                "lastTrainedDate": "2018-01-18T08:29:08",
                "modelPath": "model_variableTwo_1.pickle",
                "modelType": "OneVsRestClassifier(estimator=MultinomialNB(alpha=0.001, class_prior=None, fit_prior=True),\n          n_jobs=1)",
                "recommendationThreshold": 0.2,
                "testAccuracy": 0.9019607843137255,
                "testF1": 0.9008452056839288,
                "testPrecision": 0.9044056750340173,
                "testRecall": 0.9019607843137255,
                "testTime": 0.005343914031982422,
                "trainAccuracy": 0.9849376731301939,
                "trainDataBalance": "upsample",
                "trainF1": 0.9849744199555293,
                "trainPrecision": 0.9854426711336097,
                "trainRecall": 0.9849376731301939,
                "trainTime": 0.04866385459899902
            },
            "_type": "model"
        },
        {
            "_id": "2a53966c460340f89e72c4f65e238427",
            "_index": "models",
            "_score": 1.0,
            "_source": {
                "callCount": 29,
                "dependent": "variableOne",
                "deployed": true,
                "encoderPath": "encoder_variableOne_1.pickle",
                "encoderType": "CountVectorizer",
                "id": "2a53966c460340f89e72c4f65e238427",
                "independent": [
                    {
                        "generator_path": "clean_text.pickle",
                        "inputs": [
                            "title"
                        ],
                        "name": "cleanText"
                    }
                ],
                "lastTestedDate": "2018-01-18T08:28:28",
                "lastTrainedDate": "2018-01-18T08:28:28",
                "modelPath": "model_variableOne_1.pickle",
                "modelType": "OneVsRestClassifier(estimator=MultinomialNB(alpha=0.001, class_prior=None, fit_prior=True),\n          n_jobs=1)",
                "recommendationThreshold": 0.2,
                "testAccuracy": 0.9707006369426752,
                "testF1": 0.9686514941659194,
                "testPrecision": 0.9677349203607892,
                "testRecall": 0.9707006369426752,
                "testTime": 0.019140243530273438,
                "trainAccuracy": 0.9931685039959572,
                "trainDataBalance": "upsample",
                "trainF1": 0.9931892460624264,
                "trainPrecision": 0.993361976130266,
                "trainRecall": 0.9931685039959572,
                "trainTime": 0.3159041404724121
            },
            "_type": "model"
        }
    ]
}
```

Similarly, you can also make more complex queries, such as:

```shell
curl -XGET "https://<username>:<password>@<your_host>/api/<your_api_version>/models" -d "modelInfo={'deployed': 'true'}"
```

### HTTP Request

`GET https://<your_host>/api/<your_api_version>/models`

<!-- ### Query Parameters -->

<!-- Parameter | Default | Description -->
<!-- --------- | ------- | ----------- -->
<!-- include_cats | false | If set to true, the result will also include cats. -->
<!-- available | true | If set to false, the result will include kittens that have already been adopted. -->

<!-- <aside class="success"> -->
<!-- Remember — a happy kitten is an authenticated kitten! -->
<!-- </aside> -->

## GET Request Logs

You can also use Henosis to obtain information about requests from the
request log Elasticsearch index. This information may be useful for
debugging purposes or for assessing the behavior of your users (i.e.
those making requests to Henosis for recommendations).

### HTTP Request

`GET https://<your_host>/api/<your_api_version>/requestlogs`

```python
import json
import requests

r = requests.get('https://<your_host>/api/<your_api_version>/requestlogs', auth=('username', 'pass'))
json.dumps(r.json(), indent=4)
```

```shell
# With shell, you can just pass the correct header with each request
curl -XGET "https://<username>:<password>@<your_host>/api/<your_api_version>/requestlogs"
```

> The above command returns JSON structured like this:

```json
{
    "description": "All request logs in the Elasticsearch index.",
    "requests": [
        {
            "_index": "requestlog",
            "_type": "requestlog",
            "_id": "ztyScWEBxWqGuBjf84UG",
            "_score": 1,
            "_source": {
                "sessionId": "a1c5b9a7f01d4f30a6cb13b2390fc2ac",
                "sessionExpireDate": "2018-02-07T10:51:12",
                "timeIn": "2018-02-21T05:42:50",
                "timeOut": "2018-02-21T05:42:52",
                "timeElapsed": 2.2941131591796875,
                "responseStatusCode": 200,
                "responseDescription": "200 OK",
                "recommendations": {
                    "projectName": [
                        "projectOne",
                        "projectTwo",
                        "projectThree"
                    ],
                    "suspectedProblemArea": [
                        "hardware",
                        "software",
                        "procedural"
                    ]
                },
                "missingFields": [
                    "projectName",
                    "specificEnvironment",
                    "problemFailureNotedDuring",
                    "problemType",
                    "suspectedProblemArea"
                ],
                "modelsUsed": [
                    "d440285ac22a4528b5df23ca72c86065",
                    "658f5e1310cf42f9a62eed1028179bec"
                ]
            }
        }
    ]
}
```

Similarly, you can also make more complex queries, such as:

```shell
curl -XGET "https://<username>:<password>@<your_host>/api/<your_api_version>/requestlogs" -d "requestInfo={'responseStatusCode': 200}"
```

# FAQ

### Where was Henosis created?

Henosis was created at the [NASA Jet Propulsion Laboratory](https://jpl.nasa.gov/) (JPL) in
Pasadena, California in to support the Office of Safety and Mission Success (5X).

### When was Henosis created?

Concept formulation and development of Henosis began in October 2017.

### What does Henosis use for its back-end server?

Henosis used Flask to process and serve requests.

### Does Henosis support general Numpy arrays without field labels?

At this time Henosis **does not** support Numpy arrays. Pandas DataFrames or Series
must be used to work with Henosis.

### Do I really need an AWS S3 bucket and Elasticsearch to get started?

Yes. Henosis is set up to work directly with AWS S3 and Elasticsearch.

### How can I create the Elasticsearch mappings needed for model storage and request logging?

While Henosis does not automatically configure Elasticsearch indices, instructions
for creating indices and specifying mappings is included in this documentation.
Helpful scripts for specifying an index and mapping for each are also
included in the **scripts** directory on Github.

### How does Henosis aggregate predictions from different models for a single field?

Henosis averages predicted probabilities from each model for each class of a dependent
variable when ranking and providing recommendations to users. This capability can
be further extended to employ *bagging* (also called bootstrap aggregation) if
using training the same model across separate, randomly sampled datasets.
*predict_probas* should be set to **True** in **config.yaml** for bagging,
otherwise a simple majority vote is taken between each model (in the case when a
majority vote is used, *top_n* also defaults to **one**).

### How can I see information about my models or requests on a dashboard?

Since Henosis uses Elasticsearch for storing information about models and
requests made to a running Henosis instance, you can use
[Kibana](https://www.elastic.co/products/kibana) to
view high-level information about the models or requests being made to a deployed
Henosis instance. This is accomplished by specifying an index pattern and then
creating visualizations and dashboards.

### What is the development approach used for Henosis?

Henosis is developed using user-driven design (UDD), with development focused
on functionality but also on the ease in which Henosis can be integrated
and used in a data scientist's or software developer's workflow.

### How can I get involved in development?

Henosis is an open source project! As such, feel free to fork the Github
repository and make pull requests for new features, open issues, or other changes.

### Why in the world is it called Henosis, and what does that mean?

Have you ever seen *My Big Fat Greek Wedding?* Well, in writing out the
answer to this question, I ([Valentino](https://www.valentino.io))
realized I may be just like the dad that can tie
any word to its ancient Greek root. You see, I'm Greek in heritage and was
born on the island of Cyprus. Naturally I started looking for ancient Greek
words that could characterize this work. Turns out, the word henosis is an ancient
Greek word for *unity*. "You see? There you go!".

![Greeks](https://i.imgur.com/hrbpxGd.gif)

Real reasons? It's a distinctive name and easy to search for online. Plus, the
meaning is fitting as Henosis attempts to bring disparate things together like glue.
