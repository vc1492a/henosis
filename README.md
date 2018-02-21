# Henosis

Henosis is a cloud-native, lightweight Python-based recommender framework that brings together model training and testing,
storage and deployment, and querying under a single framework. Henosis provides Data Scientists with a straight-forward and
generalizable environment in which to train, test, store, and deploy categorical machine
learning models for making form field recommendations, while also providing software engineers
and web developers with a REST API that can be easily queried for recommendations
and integrated across different enterprise applications.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Henosis is in **active development** and is released as alpha software.

## Prerequisites

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

## How Does Henosis Work?

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

## Getting Started

A *pip install* is in the works, but for now simply fork the
repository and pull Henosis to a local directory.

## Documentation

The latest Henosis documentation is available [here](https://henosis-docs.herokuapp.com/)
and covers how to use Henosis for modeling and providing recommendations
within your form-containing applications.

## Motivation

The [NASA Jet Propulsion Laboratory](https://jpl.nasa.gov/) (JPL) has many internal enterprise
systems that collect data from large, user-populated forms (sometimes
upwards of 70 fields). These forms contain valuable spacecraft diagnostic
information input into the system by operations engineers and other subject
matter experts, but are very time consuming. We wanted to provide
users interacting with large forms recommendations on categorical inputs
to reduce the total time required and improve the efficiency of employees
interacting with these types of systems. Initially an internal effort
that began with a single use case, we decided to continue development
of Henosis as open-source software to foster collaboration and
publicly release a capability that's potentially useful to others while
development continues.

## License

This project is licensed under the Apache 2.0 license and released by
the California Institute of Technology.

## Acknowledgements

- [NASA Jet Propulsion Laboratory](https://jpl.nasa.gov/)
    - Ian Colwell
    - [Kyle Hundman](https://github.com/khundman)
    - [Paul Ramirez](https://github.com/darth-pr)

## Far-out Ideas
- parent / sibling models
- support of numpy arrays (no pandas)
- keras support
- collaborative filtering
