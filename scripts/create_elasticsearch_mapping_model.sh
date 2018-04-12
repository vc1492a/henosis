# add a mapping to the model index
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
      }
    }
  }
}'