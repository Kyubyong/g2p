import codecs
import sklearn
from nltk.corpus import cmudict

# nltk.download('cmudict')
cmu = cmudict.dict()
data = []
for k, v in cmu.items():
    item = []
    for vv in v:
        if len(k)!=len(vv):
            print(k, vv)
        else:
            for kk, vvv in zip(k, vv):
                item.append((kk, vvv))
    data.append(item)
print(data[0])

