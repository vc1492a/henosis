# Henosis

Henosis is a cloud-native, lightweight Python-based recommender framework
that facilitates providing recommendations to users of applications, like this:

<br/>

![PRS Demo Gif](https://i.imgur.com/5mzhJzm.gif)

<br/>

Henosis is in **active development** and is released as alpha software.
Henosis is being developed at the [NASA Jet Propulsion Laboratory](https://jpl.nasa.gov/) (JPL)
using user-driven development (UDD) with generous support from the Office of Safety and
Mission Success (5X).

[![PyPi](https://img.shields.io/badge/pypi-0.0.7-green.svg)](https://pypi.python.org/pypi/Henosis/0.0.7)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


Henosis brings together model training and testing,
storage and deployment, and querying under a single framework. Henosis provides Data Scientists with a straight-forward and
generalizable environment in which to train, test, store, and deploy categorical machine
learning models for making form field recommendations, while also providing software engineers
and web developers with a REST API that can be easily queried for recommendations
and integrated across different enterprise applications.

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
models to a running Henosis instance.

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
making recommendations is as easy as placing the following in a Python file.

```python
from Henosis.server import Server

# run the server
s = Server().config(config_yaml_path='config.yaml')
s.run()
```

Once a Henosis instance is running, developers can query for recommendations,
model information, and API request information from the available Henosis API (REST)
endpoints.

### The Internal Structure

Henosis works by acting as a bridge between end users and data scientists
which train recommendation (predictive) models. There are several classes
that facilitate the interaction between data, scikit-learn models, and a
REST API that provides recommendations and other information.

![Henosis Flowchart](https://i.imgur.com/EXUh0cx.png)

Henosis classes (bold borders) interface with the data scientist or statistician,
developers querying for recommendations, and trained models. When queried for
recommendations, Henosis references model information in Elasticsearch,
loads appropriate models from AWS S3, and serves recommendations by using
data available in the query (REST API request).

## Prerequisites

- Python 3.6+ (untested on lower versions) with the following packages:
    - boto3
    - dill
    - gevent
    - Flask
    - Flask-CORS
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

## Getting Started

Simply do a pip install and import the Henosis library as follows:

```bash
pip install Henosis
```

```python
from Henosis import henosis
```

If you'd like, you can also fork the repository and pull Henosis to a local directory.

## Documentation

The latest Henosis documentation is available [here](https://www.henosis.io/)
and covers how to use Henosis for modeling and providing recommendations
within your form-containing applications.

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

## License

This project is licensed under the Apache 2.0 license and released by
the California Institute of Technology.

## Acknowledgements

- [NASA Jet Propulsion Laboratory](https://jpl.nasa.gov/)
    - Ian Colwell
    - Leslie Callum
    - Harold Schone
    - [Kyle Hundman](https://github.com/khundman)
    - [Paul Ramirez](https://github.com/darth-pr)

