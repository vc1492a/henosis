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

- Python 3.6+ (untested on lower versions)
    - see requirements.txt for list of needed packages
- A working Amazon Web Services (AWS) S3 bucket, along with:
    - AWS key
    - AWS secret
- A running Elasticsearch server (we are using version 6.1.1). You'll need to create an index and
specify a mapping, so ensure you the means to do that.

## Getting Started

There are two *roles* associated with a Henosis instance; the data scientist
or modeler and the system administrator. These roles may be two
different individuals or the same individual depending on the context.

Simply pull or download this repository to get started.
Future releases will be available as a pip install.

### config.yaml

The specifications provided in *config.yaml* are used in both roles.
The data scientist or modeler uses the file to manage models, while the
system admin uses an identically configured file to deploy a Henosis instance.

Below is an example *config.yaml* file:

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
  host: 'http://localhost:5005' # leave the same, for the time being
  index: '/api'
  version: '/v0.1' # an example
  missing: '999999'
  session_expiration: 10 # in minutes
  auth: !!python/tuple ['<username>', '<password>'] # simple auth

# Models
models:
  preload_pickles: True
  predict_probabilities: True
  top_n: 3

```

Each top level of the file describes the specifications for each of those resources.
You must have your elasticsearch index and AWS S3 bucket set up prior to using your
configured *config.yaml* file. Helper functions for setting up indices and mappings
for elasticsearch are includes in the *scripts* directory. A few notes on some of
the specifications:

- elasticsearch
    - verify: tells Henosis whether or not to use SSL verification when making requests to
    the specified elasticsearch index.
- api
    - missing: this is the key that tells Henosis whether or not a form field needs a recommendation.
    This value is passed to Henosis in requests for recommendations.
    - session_expiration: defines how long a session lasts when a user starts a new session
    (series of requests).
    - auth: simple authentication parameters for the Henosis API.
- models
    - preload_pickles: tells Henosis to store models in S3 in application memory versus
    pulling from S3 for each request. Defaults to True, highly recommended.
    - predict_probabilities: if True, uses bagging to average model probabilities for each class.
    Else, provides the majority vote.
    - top_n: the number of recommendations to provide for each form field (defaults to 1 if
    predict_probabilities is set to False).

Details on how to use the *config.yaml* file in each role are detailed below.

### Using Henosis for Modeling

Henosis is developed using user-driven-development (UDD), and is intended to
work easily in a data scientist or modeler's workflow. Here's an example of a
simple modeling workflow:

#### Imports

```python
# import first
import sys
sys.path.insert(0, '../../src') # source where app.py of henosis lies
# import Henosis
import app as henosis
# other imports for this particular workflow
import itertools
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_validate
from tqdm import tqdm
```

#### Read in Data

```python
d = henosis.Data()
d.load(csv_path='data.csv')
print(d.all.columns.values)
```

#### Data Prep

```python
# clean the text for use in modeling... there can be some exploration here.
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
    # remove stopwords
    # filtered_words = [word for word in s.split(' ') if word not in stopwords.words('english')]
    # s = " ".join(t for t in filtered_words)

    if len(s) == 0:
        s = '<empty>'

    return s
```

```python
# restrict our stuff
y_var = 'y_var'

X_vars = [
    'x1_var',
]

all_vars = X_vars + [y_var]
to_model = d.all[all_vars]

# remove any empty obs for modeling
df = to_model[pd.notnull(to_model[y_var])]
y = df[y_var]

unique, counts = np.unique(df[y_var], return_counts=True)
print(np.asarray((unique, counts)).T)
```

```python
df['cleanText'] = df['title'].apply(lambda x: clean_text(x))
df[['title', 'cleanText']].head(10)
```

```python
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df['cleanText'])
X_train_counts.shape
```

```python
d.test_train_split(
    X_train_counts,
    df[y_var],
    share_train=0.8,
    X_label=['cleanText'], # needed for anything that uses the countvectorizer or tf or tf/idf vectorizers
    balance='upsample'
)
```

#### Define Model

```python
m = henosis.Models().SKModel(MultinomialNB(alpha=1e-10))
```

#### Train / Test

Simple example, practitioners can do cross-validation outside of Henosis.

```python
m.train(d)
# show the training results
print(json.dumps(m.train_results, indent=4))
```

```python
m.test(d)
# show the rest results
print(json.dumps(m.test_results, indent=4))
```

#### Tag Generated Features

Necessary if generating features from inputs in form data for models.

```python
m.tag_generator(clean_text, 'cleanText', ['title'])
print(m.independent)
```

**Note:** Generated features currently limited to data available from form data sent in requests.

#### Store Model

This call stores the pickled model in AWS S3 and indexes the model information
in elasticsearch.

```python
# NOTE: if this is your first time loading the config yaml, set preload pickles to false
s = henosis.Server().config(yaml_path='config.yaml') # may need to be absolute path
m.store(
    server_config=s,
    model_path='model_variableOne_1.pickle', # make sure not to overwrite an old file
    encoder_path='encoder_variableOne_1.pickle', # make sure not to overwrite an old file
    encoder=count_vect,
    override=True # to overwrite old AWS S3 files
)
```

#### Deploy Model

This call updates the entry in the elasticsearch index to indicate the model is deployed.

```python
m.deploy(
    server_config=s,
    deploy=True
)
```

**Note:** Henosis currently only supports categorical scikit-learn models, and
does not support regression models or collaborative filtering at this time. It is also
necessary to ensure that variable names are identical between models and the form data
sent in each request.

### Deploying a Henosis Instance

#### Imports

```python
import app as henosis # you may need to specify a path using sys first
```

#### Start an Instance

```python
s = henosis.Server().config(yaml_path='config.yaml') # may need to be absolute path
s.run()
```

While a Henosis can be run on an nginx server or Heroku, we recommend using
Docker and a container manager such as Kubernetes or
Docker Swarm for larger applications.

### Session and Request Tracking

Henosis records session and request information in the *request_log*
index specified in *config.yaml*. An API resource for these logs is
coming soon, but you could set up a Kibana dashboard, for instance.

## Henosis API

Latest documentation is in *docs* directory, and is a work in progress.
To run the docs locally, see: https://github.com/lord/slate. These docs will be
published online soon.

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
