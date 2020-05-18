alpha=0.1
beta=0.1
file=open("data/MED.REL","r")
gt=file.readlines()
dic={}
for i in gt:
    query=int(i.split(" ")[0])
    doc=int(i.split(" ")[2])
    if query not in dic.keys():
        dic[query]=[doc]
    else:
        dic[query].append(doc)
  
print (dic)
  
