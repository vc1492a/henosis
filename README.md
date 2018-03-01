# Henosis

Henosis is a cloud-native, lightweight Python-based recommender framework that brings together model training and testing,
storage and deployment, and querying under a single framework. Henosis provides Data Scientists with a straight-forward and
generalizable environment in which to train, test, store, and deploy categorical machine
learning models for making form field recommendations, while also providing software engineers
and web developers with a REST API that can be easily queried for recommendations
and integrated across different enterprise applications.

[![PyPi](https://img.shields.io/badge/pypi-0.0.5-green.svg)](https://pypi.python.org/pypi/Henosis/0.0.5)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Henosis is in **active development** and is released as alpha software.

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

## Prerequisites

- Python 3.6+ (untested on lower versions) with the following packages:
    - boto3
    - dill
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

The latest Henosis documentation is available [here](https://henosis-docs.herokuapp.com)
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
    - [Kyle Hundman](https://github.com/khundman)
    - [Paul Ramirez](https://github.com/darth-pr)

## Far-out Ideas
- parent / sibling models
- support of numpy arrays (no pandas)
- keras support
- collaborative filtering
