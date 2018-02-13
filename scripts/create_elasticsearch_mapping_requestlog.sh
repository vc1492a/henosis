# add a mapping to the model index
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