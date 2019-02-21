
from pymongo import MongoClient
from classes.FileRead import FileReader

class MongoServer(object):
	configsettings = {
		1:{
		},
		2:{

		}
	}

	def __init__(self,server=1):

		self.user = MongoServer.configsettings[server]['user'];
		self.password = MongoServer.configsettings[server]['password'];
		self.host = MongoServer.configsettings[server]['host'];
		self.port = MongoServer.configsettings[server]['port'];
		self.database = MongoServer.configsettings[server]['database'];
		self.collection = MongoServer.configsettings[server]['collection'];

		#print(f"User is {self.user}, Passwd is {self.password}, host is {self.host}, port is {self.port}")
		loginstring = f"mongodb://{self.user}:{self.password}@{self.host}:{self.port}";
		self.client = MongoClient(loginstring)
		self.db = self.client[self.database]
		self.col = self.db[self.collection]


	def queryScan5(self):
		counter = 0
		for x in self.col.find({},{"_id":1}):
			counter+=1
			id = x["_id"]
			print(id)
			if (counter==5):
				break

	@classmethod 
	def testscan(cls):
		md = MongoServer(1)
		md.queryScan5()

