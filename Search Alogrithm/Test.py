import pickle
import pprint
obj = pickle.load(open("romania_references.pickle", "rb"))
with open("out.txt","a") as f:
    pprint.pprint(obj,stream=f)