from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import initialize as p
import json

key = p.initialize()

class elastic:
    def __init__(self):
        self.cert = key.cert
        self.pwd  = key.pwd
        
    def connect(self):
        self.es = Elasticsearch(
            "https://localhost:9200",
            ssl_assert_fingerprint=self.cert,
            basic_auth=("elastic", self.pwd)  
        )
        if self.es.ping(): print("Connected to server")  
        else: print("Failed to connect")
        
    def search(self,data):
        operator = []
        operator.append(data)
        
        #print(data)
        for x in operator:
            y = x.split()
            
            if 'and' in y:      
                y.remove('and')
                
                separator = ' '
                data_and = separator.join(y)
                print(data)
            
                return self.search_and(data_and)
            
            else:
                return self.search_or(data)
                            
    def search_or(self,data):
        # See Link https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/examples.html
        # See link https://dylancastillo.co/elasticsearch-python/
        print("OR")

        print(data)
        resp = self.es.search(
        index = "hotline_wikidb",
        body={     
            "query": {
            "multi_match": {
            "query": data,
            "fields": ["Answer","Question","Topic"],
            "operator": "or",
            "type": "cross_fields"
            },
        },    
        })

        return (resp)
                
        #jsonString = json.dumps(resp['hits']['hits'])
        ##jsonString = json.dumps(resp['hits'])

        #jsonFile = open(r"D:\Flask\Flask_FAQ\v1_1\static\json\data.json","w")
        #jsonFile.write(jsonString)
        #jsonFile.close()

        
    def search_and(self,data):
        print("AND")
        # See Link https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/examples.html
        # See link https://dylancastillo.co/elasticsearch-python/

        print(data)
        resp = self.es.search(
        index = "hotline_wikidb",
        body={     
            "query": {
            "multi_match": {
            "query": data,
            "fields": ["Answer","Question","Topic"],
            "operator": "and",
            "type": "cross_fields"
            },
        },    
        })

        # jsonString = json.dumps(resp['hits']['hits'])
        # jsonFile = open(r"D:\Flask\Flask_FAQ\v1_1\static\json\data.json","w")
        # jsonFile.write(jsonString)
        # jsonFile.close()

        return (resp)
        
    def update(self,tic, pro, cus, sub, top, des, sup):
        # See Link https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/examples.html
        doc = {
            'Ticket' : tic,
            'Process' : pro,
            'Customer' : cus,
            'Subject' : sub,
            'Topic' : top,
            'Description' : des,
            'Support' : sup,
        }
        resp = self.es.update(index="hotline_wikidb",id=127, doc=doc) 
        
    def add(self,tic, pro, cus, sub, top, des, sup):
        # See Link https://www.elastic.co/guide/en/cloud/current/ec-getting-started-python.html
        self.es.index(
            index = "hotline_wikidb",
            document = {
            'Ticket' : tic,
            'Process' : pro,
            'Customer' : cus,
            'Subject' : sub,
            'Topic' : top,
            'Description' : des,
            'Support' : sup,
        }    
        )
            
    def all(self):
        resp = self.es.search(
            index = "hotline_wikidb",
            body={
                "query": {
                    "match_all": {}
                }
            }
        )
               
        value = resp['hits']['total']['value']
        
        return value
    
    def refresh(self):
        self.es.indices.refresh(index="hotline_wikidb")
        
if __name__ == "__main__":
    init = elastic()
    init.connect()
    
    #init.search('"XP018" "ESD"')
    
    #init.search_and("XR013")
    
    #init.search("XR013")
    #init.update("00","xt018","lukas","esd","esd clamp","guideline","refer to guideline")
    
    #init.all()
    
    #init.add("00","xt018","james","esd","esd clamp","guideline","refer to guideline")

        
        
        