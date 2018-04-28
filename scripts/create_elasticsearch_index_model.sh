# create the model index
curl -XPUT '<host>/model' -H 'Content-Type: application/json' -d '{
    "settings" : {
        "index" : {
            "number_of_shards" : 3,
            "number_of_replicas" : 2
        }
    }
}'
