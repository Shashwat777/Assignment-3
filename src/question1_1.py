import numpy as np
import tensorflow as tf
import re
from nltk.corpus import abc
import sys
from collections import OrderedDict
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import pandas as pd
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt



# This function cleans the corpus

def corpus_cleaning(corpus):
    cor=[]
    for word in corpus:
        word=word.lower()
        if(len(word)>1 and  word.isalpha()):
            cor.append(word.lower())
    return (cor)
# The below function assigns id to each word , and maps is to word and vece versa

def mapping(vocab):
    word_to_id=dict((word.lower(),id) for id,word in enumerate(vocab))
    id_to_word=dict((id,word.lower()) for id,word in enumerate(vocab))
    return ({"wti":word_to_id,"itw":id_to_word})
# function makes pairs of neighbouring words within the window size
def make_pairs(vocab,window):
    ln=len(vocab)
    pair=[]
    for i in range(ln):
        relatives= list(range(max(0, i - window), i)) +  list(range(i + 1, min(ln, i + window + 1)))
        for relative in relatives :
           pair.append([vocab[i],vocab[relative]])
  
    return (pair)
        

# The following function creates the training set using one-hot-encoding
def create_traindata(ln,words,window,word_to_id):
        pairs=make_pairs(words,window)
      
        
        X=[]
        Y=[]
  
        for pair in pairs:
            x=[0]*ln
            y=[0]*ln
            # one hot encoding
            x[word_to_id[pair[0].lower()]]=1
            y[word_to_id[pair[1].lower()]]=1
            X.append(x)
            Y.append(y)
        return (X,Y)
def plot(word2int,vectors,epoch,int2word):
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    vectors = model.fit_transform(vectors)
    # normalizer = preprocessing.Normalizer()
    # vectors =  normalizer.fit_transform(vectors, 'l2')

    # fig, ax = plt.subplots()
    # for word in word2int.keys():
    #  ax.annotate(word, (vectors[word2int[word]][0],vectors[word2int[word]][1] ))

    # plt.savefig('plots/epoch'+str(i))
    x = []
    y = []
    df=pd.DataFrame({'x': np.random.normal(10, 1.2, 20000), 'y': np.random.normal(10, 1.2, 20000), 'group': np.repeat('A',20000) })
    for value in vectors:
     x.append(value[0])
     y.append(value[1])
    
    plt.figure(figsize=(50, 50)) 
    for i in range(len(x)):
     plt.scatter(x[i],y[i])
     plt.annotate(int2word[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
# plt.plot( 'x', 'y', data=df, linestyle='', marker='o', markersize=.713)

    plt.savefig('plots/epoch'+str(epoch))

    


    



# Inputs
learning_rate=0.1
embdg_size=10
epochs=50
wordlen=10000

corpus=(abc.words()[:wordlen])

window=4
# main
corpus=corpus_cleaning(corpus)
vocab=set(corpus)

vocab_size=len(vocab)
mapped=mapping(vocab)
word_to_id=mapped["wti"]
id_to_word=mapped["itw"]
x_train,y_train=create_traindata(vocab_size,corpus,window,word_to_id)
x_train=np.asarray(x_train)
y_train=np.asarray(y_train)
print (vocab_size)
print (x_train.shape)
# layer1
x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))
W1 = tf.Variable(tf.random_normal([vocab_size, embdg_size]))
b1 = tf.Variable(tf.random_normal([embdg_size])) #bias
hidden_representation = tf.add(tf.matmul(x,W1), b1)
W2 = tf.Variable(tf.random_normal([embdg_size, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) #make sure you do this!
# define the loss function:
cost=-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1])
cross_entropy_loss = tf.reduce_mean(cost)
# define the training step:
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss)
# train for n_iter iterations
val=epochs/10

for i in range(epochs+1):
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    print('epoch  '+str(i)+'     loss : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))
    vectors = sess.run(W1 + b1)
 
    
    if (i%10==0):
     print ('plotting'+str(i))

     plot(word_to_id,vectors,i,id_to_word)
    if(i==epochs-1):
        file=open("result2.txt","w")
        for j in range (len(vectors)):
            file.write(str(id_to_word[j])+":"+str(vectors[j]))
            file.write("\n")
            file.write("\n")
            file.write("\n")


        
