# add a mapping to the model index
curl -XPUT '<host>/requestlog/_mapping/requestlog' -H 'Content-Type: application/json' -d '{
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