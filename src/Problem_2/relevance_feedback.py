import numpy as np
from scipy import spatial
alpha=1
beta=1
file=open("data/MED.QRY","r")
print (file.readlines())
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



  
def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    
  
    simt=sim.T
   
    c=0
    for i in range (0,30):
        quer=i+1
        query_res=simt[i]
     
        top_query=np.argsort(-query_res)[:n]
    
        for j in top_query:
            if (j+1) in dic[quer]:
                c=c+1
                simt[i][j]=simt[i][j]*(1+alpha)
            else:
              
                simt[i][j]=simt[i][j]*(1-beta)
  
    sim=simt.T
    
     

        



   
 
    
 


    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    rf_sim = sim # change
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)

    """
    arr= ((vec_queries).toarray())
    w=10
    dic2={}
    
    for i in range (30):
       
        for j in range (30):
       
                val=1-spatial.distance.cosine(arr[i],arr[j])
                
                if i not in dic2.keys():
                    dic2[i]=[val]
                else:
                     dic2[i].append(val)
     
        l=np.argsort(dic2[i])
        dic2[i]=l[:w]
    simt=sim.T
   
    c=0
    for i in range (0,30):
        quer=i+1
        query_res=simt[i]
     
        top_query=np.argsort(-query_res)
        lst=[]
    
        for j in top_query:
            if (j+1) in dic[quer]:
                c=c+1
                simt[i][j]=simt[i][j]*(1+alpha)
                for j in range (len(dic2[i])):
                    q=dic2[i][j]
                    simt[q][j]=simt[q][j]*(1+alpha)
            
            else:
              
                simt[i][j]=simt[i][j]*(1-beta)
                for j in range (len(dic2[i])):
                    q=dic2[i][j]
                    simt[q][j]=simt[q][j]*(1-beta)

            


        
  
    sim=simt.T
    
    
        
       

    

    

    


    rf_sim = sim  # change
    return rf_sim