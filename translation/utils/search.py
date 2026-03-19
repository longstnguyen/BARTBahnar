import json
import pandas as pd
from urllib3 import PoolManager
from urllib.parse import quote
from config import DICTIONARY_PATH


class SolrClient:
    """
    Client for interacting with a Solr instance.
    """
    def __init__(self, solr_url):
        self.solr_url = solr_url.rstrip('/')
        self.http = PoolManager()

    def delete_all_documents(self):
        """
        Delete all documents from the Solr core.
        """
        delete_query = '<delete><query>*:*</query></delete>'
        headers = {"Content-Type": "text/xml"}
        response = self.http.request('POST', f'{self.solr_url}/update?commit=true', body=delete_query, headers=headers)

        if response.status == 200:
            print("All data on Solr has been deleted.")
        else:
            print(f"Error deleting data: {response.status}, {response.data.decode('utf-8')}")

    def upload_documents(self, data):
        """
        Upload documents to Solr.
        """
        headers = {'Content-Type': 'application/json'}
        response = self.http.request('POST', f'{self.solr_url}/update?commit=true', body=json.dumps(data).encode('utf-8'), headers=headers)

        if response.status == 200:
            print("Data successfully uploaded to Solr.")
        else:
            print(f"Error uploading data to Solr: {response.data.decode('utf-8')}")

    def search_bahnar_words(self, words):
        """
        Search a list of Bahnar words in Solr and return Bahnar-Vietnamese translation pairs.
        """
        or_query = " OR ".join([f'bahnar:"{quote(word)}"' for word in words])
        search_url = f'{self.solr_url}/select?indent=true&q.op=OR&q=({or_query})&rows=1000&fl=bahnar,vietnamese&wt=json'

        response = self.http.request('GET', search_url)

        try:
            data = json.loads(response.data.decode('utf-8'))
        except json.JSONDecodeError:
            # print("❌ Failed to decode Solr response.")
            return []

        if 'response' not in data:
            # print(f"❌ Solr error: {data.get('error', 'No error info')}")
            return []

        results = {}
        for doc in data['response']['docs']:
            bahnar_word = doc.get('bahnar', [''])[0]
            vietnamese_word = doc.get('vietnamese', [''])[0]

            if bahnar_word and vietnamese_word:
                if bahnar_word not in results:
                    results[bahnar_word] = []
                results[bahnar_word].append(vietnamese_word)

        final_results = [{"bahnar": k, "vietnamese": list(set(v))} for k, v in results.items()]
        return final_results


class DictionaryReader:
    """
    Reads the local bilingual dictionary CSV.
    """
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def read(self):
        """
        Load the dictionary and return a DataFrame with Bahnaric and Vietnamese columns.
        """
        df = pd.read_csv(self.csv_path)
        return df[['Bahnaric', 'Vietnamese']]


class SearchTranslator:
    """
    Indexes the bilingual dictionary into Solr on startup and searches Bahnar words.
    If Solr is not running, the translator operates without dictionary lookup.
    """
    def __init__(self, solr_url):
        self.solr_client = SolrClient(solr_url)
        self.available = False

        try:
            # Load dictionary from local CSV
            df = DictionaryReader(DICTIONARY_PATH).read()

            # Clear old index and re-upload
            self.solr_client.delete_all_documents()
            documents = [{"bahnar": row["Bahnaric"], "vietnamese": row["Vietnamese"]} for _, row in df.iterrows()]
            self.solr_client.upload_documents(documents)
            self.available = True
        except Exception as e:
            print(f"Solr not available ({e.__class__.__name__}): dictionary lookup disabled.")

    def search(self, words):
        """
        Search a list of Bahnar words in Solr. Returns empty list if Solr is unavailable.
        """
        if not self.available:
            return []
        return self.solr_client.search_bahnar_words(words)

