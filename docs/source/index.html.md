---
title: API Reference

language_tabs: # must be one of https://git.io/vQNgJ
  - shell
  - python

search: true
---

# About

Henosis, a cloud-native, Python-based form recommender framework that brings
together model training and testing, storage and deployment, and querying under a
single framework (Henosis is a classical Greek word for “unity”). Developed using
user-driven design (UDD), Henosis serves as a bridge between two roles in an
organization. Henosis provides data scientists with a straight-forward and
generalizable environment in which to train, test, store, and deploy categorical
machine learning models for making form field recommendations using scikit-learn,
while also providing developers with a REST API that can be easily queried for
recommendations and integrated across different enterprise applications. This
user-driven design and simplification of the integration process allows for
easier deployment of recommender system capability across different enterprise
applications.

Henosis was developed at the [NASA Jet Propulsion Laboratory](https://jpl.nasa.gov/) (JPL)
using user-driven development (UDD) with generous support from the Office of Safety and
Mission Success (5X).

## How Does Henosis Work?

Henosis works by acting as a bridge between end users and data scientists
which train recommendation (predictive) models. There are several classes
that facilitate the interaction between data, scikit-learn models, and a
REST API that provides recommendations and other information.

<br/>
<br/>

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
    - pymssql
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
for form fields. What we needed from a data science and IT operations perspective
was a straightforward process to deploy a simple recommendation system for use in
enterprise applications containing categorical form inputs (like dropdown menus).

While the initial effort focused on one internal use case, Henosis was developed
as a generalized, open-source framework and is freely available for use in
other applications.

# Configuration

Henosis is largely configured using a locally stored **config.yaml** file. This configuration
file contains parameters associated with Henosis API configuration, AWS S3 bucket
specification, Elasticsearch configuration, and Henosis model deployment settings,
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
  auth: !!python/tuple ['<username>', '<password>']
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
  auth: !!python/tuple ['<username>', '<password>']

# Models
models:
  preload_pickles: True
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
    *Must be the same as the port used starting a Henosis instance. Default parameters recommended.*
    - index: the name of your API following the host. *Default parameters recommended.*
    - version: the version of your API. *Default parameters recommended.*
    - missing: this is the key that tells Henosis whether or not a form field needs a recommendation.
    This value is passed to Henosis in requests for recommendations.
    - session_expiration: defines how long a session lasts when a user starts a new session
    (series of requests), in minutes.
    - auth: simple authentication parameters for the Henosis API. *Use of simple authentication
    for the Henosis API is recommended.*
- models
    - preload_pickles: tells Henosis to store models in S3 in application memory versus
    pulling from S3 for each request. Defaults to True, highly recommended.
    - predict_probabilities: if True, uses bagging to average model probabilities for each class.
    Else, provides the majority vote.
    - top_n: the number of recommendations to provide for each form field (defaults to 1 if
    predict_probabilities is set to False).


## Configuring Elasticsearch

Henosis utilizes Elasticsearch to identify which models are available for use,
when models have the appropriate data for making recommendations, and to
keep track of models as they are used to make recommendations for users.
Additionally, a separate Elasticsearch index is used to keep track of requests
made to the Henosis API.

These indices **must** first be created and mapped before storing and deploying
models in Henosis and prior to running a Henosis instance.

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
        "type": "text"
      },
      "independent": {
        "type": "nested",
        "properties": {
          "name": {
            "type": "text"
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
        "type": "text"
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
      "threshold": {
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
      }
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
        "type": "text"
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
        "type": "text"
      },
      "recommendations": {
        "type": "nested",
        "properties": {
          "fieldName": {
            "type": "text"
          }
        }
      },
      "modelsQueried": {
        "type": "text"
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

Henosis is developed using user-driven-development (UDD), and is intended to
work easily in a data scientist or statistician's workflow.
Two classes, *Data* and *Models*, are used to work with your data and
scikit-learn models that form the basis of the form recommendation system.

## Data class

The Data class facilitates the loading, saving, and splitting of
data into training and testing splits.

### Load Data

Henosis supports loading data from local storage, specifically CSV files, and
can be called after loading the *Data* class as an object.

> Load Data

```python
d = henosis.Data()
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
must be specified.

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

## Models class

The Henosis Models class is used jointly with categorical scikit-learn
models to train, test, store, and deploy models for use with form fields.
While a specified **config.yaml** file is not needed to get started with
the train and test methods, storing and deploying models to a running
Henosis instance requires configuration.

**Note:** Henosis only supports categorical scikit-learn models, and
does not support regression models or collaborative filtering at this time.
Additionally, it is necessary to ensure that variable names are *identical*
between models (training data) and the form data sent in each request. This
is how Henosis identifies when the data necessary to make a recommendation
is available for each model.

### Defining Models

Defining a model in Henosis is as easy as passing a scikit-learn model
into the Henosis models SKModel class. Simply pass any categorical
scikit-learn model into the SKModel class.

> Define Model

```python
m = henosis.Models().SKModel(MultinomialNB(alpha=1e-10))
```

### Train

Once training and testing splits have been created using the Data class,
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
# NOTE: if this is your first time loading the config yaml, set preload pickles to False
s = henosis.Server().config(yaml_path='config.yaml') # may need to be absolute path
```

```python
m.store(
    server_config=s,
    model_path='model_variableOne_1.pickle', # make sure not to overwrite an old file
    encoder_path='encoder_variableOne_1.pickle', # make sure not to overwrite an old file
    encoder=count_vect, # only needed for CountVectorizer and TfidfVectorizer data
    override=True # to overwrite old AWS S3 files
)
```

*encoder_path* and *encoder* are optional and only needed if variable
encoders are used as part of the training process, such as when
using CountVectorizer or TfidfVectorizer when training models.

### Deploy

When deploying models, reference the same *s* object created
as part of the model storage process.

> Deploy Model

```python
m.deploy(
    server_config=s,
    deploy=True
)
```

To take a model offline, simply do the same with *deploy=False*.

# Deployment

## Server class

The Server class is used by Henosis to reference the specifications in
**config.yaml** and to deploy a Flask application that runs the
Henosis API.

### Config

The *config* function accepts the path of the **config.yaml** file
and ingests it into Henosis for running the API and storing and deploying
models.

> Load Config

```python
s = henosis.Server().config(yaml_path='config.yaml') # may need to be absolute path
```

### Run

The *run* function uses the previously-loaded configuration to start
a Flask application which runs the Henosis API, which is used for querying
for recommendations.

> Run

```python
s.run(port=5005)
```

Port 5005 is the default port, but this port **must match** that specified
in your API host (in config.yaml) for Henosis to run properly.

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
be used to query for user demographics and pass this information to the **recommendations**
route in Henosis for obtaining recommendations.

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

s = henosis.Server().config(yaml_path='config.yaml') # may need absolute path
s.run(routes=custom_routes, api_resources=custom_endpoints)
```

# Henosis API

Querying Henosis is easy! The Henosis REST API allows software engineers
and others to query a deployed Henosis instance, providing recommendations
based on available data. The Henosis API also allows users to query for
important model information and request logs for debugging or user
analysis.

## Authentication

Henosis currently supports simple authentication. API authentication is set
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
provided.

There's just a few things to keep in mind:

- Make sure to pass fields that need recommendations using the missing
tag specified in **config.yaml**.
- Make sure to pass field names in the same format as they are specified
in the models, including capitalization.

> In Python:

```python
import json
import requests

q = {
  'variableOne': '999999',
  'variableTwo': 'Spaceman',
  'variableThree': '999999',
}

r = requests.get('https://<your_host>/api/<your_api_version>/recommend', json=q, auth=('username', 'pass'))
json.dumps(r.json(), indent=4)
```

> Using shell:

```shell
curl -XGET "https://<username>:<password>@<your_host>/api/<your_api_version>/recommend" -d
  "formData={'variableOne': 999999, 'variableTwo': 'Spaceman', 'variableThree': 999999}"
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
  "variableOne": "Proton Beam",
  "variableTwo": "Spaceman",
  "variableThree": "Liquid Nitrogen Resistant Gloves",
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
                "f1_threshold": 0.85,
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
                "f1_threshold": 0.85,
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
                    "specificEnvironment": [
                        "specificEnvironmentOne",
                        "specificEnvironmentTwo",
                        "specificEnvironmentThree"
                    ],
                    "problemFailureNotedDuring": [
                        "inspection",
                        "testing",
                        "flight"
                    ],
                    "problemType": [
                        "engineering",
                        "testing",
                        "design"
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
                "modelsQueried": [
                    "be4d26d624a6408abd105b015e0526b3",
                    "b1889ca0646b46fabbfac28127a97d8a",
                    "d440285ac22a4528b5df23ca72c86065",
                    "658f5e1310cf42f9a62eed1028179bec",
                    "7e7298f649ae4540aa2c1fba2e014a81",
                    "897ffab10a26446288560260027ddfff"
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

Henosis uses an ensemble method called *bagging* to aggregate predictions when more than one
model provides a recommendation for a particular field, provided each model is trained
separately on unique training data and *predict_probas* is set to **True** in **config.yaml**.
Else, a simple majority vote is taken between each model. In the case when a
majority vote is used, *top_n* defaults to **one**.

### How can I see information about my models or requests on a dashboard?

Since Henosis uses Elasticsearch for storing information about models and
requests made to a running Henosis instance, you can use
[Kibana](https://www.elastic.co/products/kibana) to
view high-level information about the models or requests being made to a deployed
Henosis instance. This is accomplished by specifying an index pattern and then
creating visualizations and dashboards.

### How can I get involved in development?

Henosis is an open source project! As such, feel free to fork the Github
repository and make pull requests for new features, open issues, or other changes.
