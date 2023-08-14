"""
See this link on how to setup the Elastic Search connection between Python and Elastic Search
Server

Link : https://www.elastic.co/guide/en/elasticsearch/client/python-api/master/connecting.html

"""

from elasticsearch import Elasticsearch
import logging
import requests

CERT_FINGERPRINT = "7e73d3cf8918662a27be6ac5f493bf55bd8af2a95338b9b8c49384650c59db08"
#CERT_FINGERPRINT = "d05aaa8eba62fbb871cd966a29d0a9ba3336e29fbb6463deab015c1d985a246e"

ELASTIC_PASSWORD = "Eldernangkai92"

es = Elasticsearch(
    "https://localhost:9200",
    ssl_assert_fingerprint=CERT_FINGERPRINT,
    basic_auth=("elastic", ELASTIC_PASSWORD)
)

if es.ping():
    print("Connected to server")
else:
    print("Failed to connect")

# Index name and mapping configuration - index mapping is obtained from elastic_vector mapping 
index_name = "elastic_wiki"
index_mapping = {
  "mappings": {
    "properties": {
      "metadata": {
        "properties": {
          "page": {
            "type": "long"
          },
          "source": {
            "type": "text",
            "fields": {
              "keyword": {
                "type": "keyword",
                "ignore_above": 256
              }
            }
          }
        }
      },
      "text": {
        "type": "text"
      },
      "vector": {
        "type": "dense_vector",
        "dims": 384
      }
    }
  }
}

# Create the index with the specified mapping
response = es.indices.create(index=index_name, body=index_mapping)

# Check if the index was created successfully
if response["acknowledged"]:
    print(f"Index '{index_name}' created successfully with vector field 'vector_field'.")
else:
    print("Failed to create the index.")

# Close the Elasticsearch connection
es.close()


