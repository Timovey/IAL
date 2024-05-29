import owlready2
from owlready2 import *

onto = get_ontology("C:/Users/Timovey/Study/IAL/ontology1.owl").load()
owlready2.JAVA_EXE = "C:/Program Files/Java/jdk-22/bin/java.exe"
sync_reasoner()

# print(list(onto.classes()))
# print(list(onto.individuals()))
# print(list(onto.search(type=onto['Theme'])))
# print(list(onto.search(type=onto['Review'])))

onto.save(file="ontology2.owl", format="rdfxml")