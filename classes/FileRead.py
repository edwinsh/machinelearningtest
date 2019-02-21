
import os

class FileReader(object):

	filedir =  os.path.dirname(os.path.realpath(__file__))+'/../'
	list1 = ['a','b','c']
	list2 = [1,2,3]
	hash1 = {'a':1,'b':2,"c":3}

	def __init__(self):
		self.filename = 'filename2'
		print("\n\n***FilenameInstIs***: " + self.filename)
		print("\n\n***FilenameClassIs***: " + FileReader.filedir)
		#print("\n\n***FilenameClassIs***: " + cls.filename)

	#decorator
	@classmethod
	def getFileInfo(cls):
		#print("\n\n***ClassMethod getFileName: " + FileReader.filename)
		print("\n\n***ClassMethod getFileInfo: " + cls.filedir)
		current_file1 = __file__
		current_file2 = os.path.realpath(__file__)
		current_dir1 = os.path.dirname(__file__)
		current_dir2 = os.path.dirname(os.path.realpath(__file__))
		print("CurrentFile1 is: ", current_file1)
		print("CurrentFile2 is: ", current_file2)
		print("CurrentDirectory is: ", current_dir1)
		print("CurrentDirectory is: ", current_dir2)

	#decorator
	@classmethod
	def getMainFileDir(cls):
		return cls.filedir

	@classmethod
	def writeFile(cls,newfile):
		filedir = cls.getMainFileDir()
		file = filedir+'/'+newfile
		print("\nFileDir ", filedir)
		print("\nFile ",file)
		fhandle = open(file,'w')
		writeData = {
			1:"This is the first line!",
			2:"This is the second line of my python filewriting",
			3:"This is the third line of my python filewriting",
			4:"This is the fourth line of my python filewriting"
		}
		for index,dataline in writeData.items():
			print(f"Line:{index}, ",dataline)
			fhandle.write(f"{dataline}\n")

		readoption = 2
		cls.readFile(file,readoption)

	@classmethod
	def readFile(cls,newfile,readoption=2):
		if (readoption == 1):
			#readback what was written
			fhandle = open(file,'r')
			counter = 0;
			while True:
				counter+=1
				line = fhandle.readline()
				if (not line):
					print("line is 'not' ")
					break 	
				elif (line.strip == ''):
					print("line is empty string")
					break 
				else:
					print(f"Line is: {line}", end="")
		if (readoption == 2):
			fhandle = open(newfile,'r')
			listOfLines = fhandle.readlines()
			for line in listOfLines:
				print(f"Line is: {line}", end="")
				print(f"Split line into array: ",line.split(" "), "\n")

	

###PRACTICE FUNCTIONS BELOW###

	@classmethod
	def scanDirectory(cls, filename):
		directory = cls.filedir
		filelist = os.listdir(directory)
		for name in filelist:
			print(name)
			if (name == filename):
				txt = open(filename)
				print(f"\n\nHere's your file {filename}:")
				print(txt.read())

	@classmethod
	def iterateLists(cls):
		for alpha,num in zip(cls.list1,cls.list2):
			print(f"Alpha: {alpha}, Num: {num}")

	@classmethod
	def iterateHashes(cls):

		tempHash = {}
		for alpha, num in cls.hash1.items():
			print(f"Alpha: {alpha}, Num: {num}")
			tempHash[alpha]=num*2
		print(f"NewHash is {tempHash}")
		
		counter = 0
		for alpha, num in cls.hash1.items():
			counter += 1
			print(f"Hash Item# {counter}", cls.hash1[alpha])

	@classmethod
	def doubleHashValues(cls):
		print("\ndoubleHashValues function")
		counter = 0
		for alpha, num in cls.hash1.items():
			counter += 1
			#cls.hash1[alpha] = cls.hash1[alpha] * 2
			cls.hash1[alpha] *= 2
			newstringindex = alpha + str(counter)
			print(f"new index name: ",newstringindex)
			#cls.hash1[newstringindex] = cls.hash1[alpha];
			print(f"Hash Item# {counter}, index: {alpha}, ", cls.hash1[alpha])




