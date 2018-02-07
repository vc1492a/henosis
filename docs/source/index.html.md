---
title: API Reference

language_tabs: # must be one of https://git.io/vQNgJ
  - shell
  <!-- - ruby -->
  - python
  <!-- - javascript -->

<!-- toc_footers: -->
  <!-- - <a href='#'>Sign Up for a Developer Key</a> -->
  <!-- - <a href='https://github.com/lord/slate'>Documentation Powered by Slate</a> -->

<!-- includes: -->
  <!-- - errors -->

search: true
---

# About

Henosis is a cloud-native, lightweight Python-based recommender framework that brings together model training and testing,
storage and deployment, and querying under a single framework. Henosis provides Data Scientists with a straight-forward and
generalizable environment in which to train, test, store, and deploy categorical machine
learning models for making form field recommendations, while also providing software engineers
and web developers with a REST API that can be easily queried for recommendations
and integrated across different enterprise applications.

## Requirements

- Python 3.6+ (untested on lower versions)
    - see requirements.txt for list of needed packages
- A working Amazon Web Services (AWS) S3 bucket, along with:
    - AWS key
    - AWS secret
- A running Elasticsearch server (we are using version 6.1.1). You'll need to create an index and
specify a mapping, so ensure you the means to do that.

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

# Modeling

Henosis is developed using user-driven-development (UDD), and is intended to
work easily in a data scientist or modeler's workflow. Here's an example of a
simple modeling workflow:

## Example Workflow

> Imports

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

> Read in Data

```python
d = henosis.Data()
d.load(csv_path='data.csv')
print(d.all.columns.values)
```

> Data Prep

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

> Define Model

```python
m = henosis.Models().SKModel(MultinomialNB(alpha=1e-10))
```

> Tag Generated Features

Necessary if generating features from inputs in form data for models.

```python
m.tag_generator(clean_text, 'cleanText', ['title'])
print(m.independent)
```

> Store Model: This call stores the pickled model in AWS S3 and indexes the model information
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

> Deploy Model: This call updates the entry in the elasticsearch index to indicate the model is deployed.

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

# Deployment

While a Henosis can be run on an nginx server or Heroku, we recommend using
Docker and some sort of container manager such as Kubernetes or
Docker Swarm for larger applications.

```python
import app as henosis # you may need to specify a path using sys first
```

```python
s = henosis.Server().config(yaml_path='config.yaml') # may need to be absolute path
s.run()
```

# Session and Request Tracking

Henosis records session and request information in the *request_log*
index specified in *config.yaml*. An API resource for these logs is
coming soon, but you could set up a Kibana dashboard, for instance.

# Henosis API

Querying Henosis is easy! The Henosis REST API allows software engineers and others to query a deployed Henosis instance,
providing recommendations based on available data and model information important to maintaining a recommender system.

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
the models deployed in the Henosis instance that recommendations may
not be returned from your request.
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
            "_index": "recsys",
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
            "_index": "recsys",
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

## GET Requests

Coming soon, will fetch session and request logs from elasticsearch.

# FAQ

### How does Henosis aggregate predictions from different models for a single field?

Henosis uses an ensemble method called *bagging* to aggregate predictions when more than one
model provides a recommendation for a particular field, provided each model is trained
separately on unique training data and *predict_probas* is set to **True** in **config.yaml**.
Else, a simple majority vote is taken between each model.
