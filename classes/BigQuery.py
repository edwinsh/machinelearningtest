from google.cloud import bigquery
import os
from classes.FileRead import FileReader


class BigQueryClient(object):

	bq_setup_array = {
		'projectId':'api-project-561775070285',
		'keyFilePath':'Shop_Heroes-ded407771624.json',	
		'csvquote_string':'"'
	}

	def __init__(self):
		filedir = FileReader.getMainFileDir()
		print("PrintFile: ",filedir)
		self.keyfile = filedir + "/" + BigQueryClient.bq_setup_array['keyFilePath']
		#self.client = bigquery.Client.from_service_account_json('Shop_Heroes-ded407771624.json')
		self.client = bigquery.Client.from_service_account_json(self.keyfile)

	def query(self):
		query = (
		    "SELECT id FROM `Analytics.EventsTable2` "
		    "LIMIT 100"
		)
		
		query_job = self.client.query(
		    query,
		    # Location must match that of the dataset(s) referenced in the query.
		    location="US",
		)  # API request - starts the query

		for row in query_job:  # API request - fetches results
		    # Row values can be accessed by field name or index
		    #assert row[0] == row.name == row["name"]
		    print(row['id'])
