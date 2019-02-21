
from sys import argv
from classes.FileRead import FileReader

fr = FileReader()
FileReader.getFileInfo()
FileReader.scanDirectory('testfile')
FileReader.iterateLists()
FileReader.iterateHashes()
FileReader.doubleHashValues()
FileReader.writeFile('testwritefile')

#print("FileNameInstance is: ",fr.filename)
#print("FileNameClass is: ",FileReader.filename)
#print("Name is " + __name__)
#print("\n\n")


"""
script, filename = argv
txt = open(filename)

print(f"Here's your file {filename}:")
print(txt.read())

print("Type the filename again:")
file_again = input("> ")
txt_again = open(file_again)

print(txt_again.read())
"""
