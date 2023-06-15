

import os
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# import torch.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import re
import numpy as np
import time
from collections import defaultdict
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# %matplotlib inline
import itertools
import csv
import pandas as pd
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import spacy
from sklearn.manifold import TSNE

from gensim.models.phrases import Phrases, Phraser
import logging
import gensim
from gensim.models import Word2Vec
import multiprocessing
# from gensim.models.wrappers import FastText
from gensim.models import fasttext as ftext
from gensim.test.utils import datapath
from torch.utils.data import Dataset, DataLoader


from torch.autograd import Variable
from torch.nn.parameter import Parameter
from sklearn.cluster import DBSCAN
# import fasttext.util
from sklearn.metrics import confusion_matrix
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

random.seed(50)

class MyData(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y
        # self.length = [ np.sum(1 - np.equal(x, 0)) for x in X]
        self.length = [ len(x) for x in X]
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        return x, y, x_len
    
    def __len__(self):
        return len(self.data)




class ConstructVocab():
    def __init__(self, sentences):
        self.sentences = sentences
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()
        
    def create_index(self):

        print(type(self.sentences))
        self.vocab.update(self.sentences)
            
        # sort the vocab
        self.vocab = sorted(self.vocab)
#         print(self.vocab)
        

        # add a padding token with index 0
        self.word2idx['<pad>'] = 0
        
        # word to index mapping
        for index, word in enumerate(self.vocab):
            if word == self.vocab[-1]:
                self.word2idx[word] = 0
            else:
                self.word2idx[word] = index + 1 # +1 because of pad token
        
        # index to word mapping
        for word, index in self.word2idx.items():
            self.idx2word[index] = word
            


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from torch.nn.parameter import Parameter


class LSTM_Attn_Emo(torch.nn.Module):
    def __init__(self,  vocab_size, embedding_matrix, embedding_dim, batch_size, output_size, hidden_size, n_layers, bidirectional,
                  dropout, NUM_FILTERS =10, window_sizes=(1,2,3,5)):
      super(LSTM_Attn_Emo, self).__init__()
      
      """
      Arguments
      ---------
      batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
      output_size : 2 = (pos, neg)
      hidden_sie : Size of the hidden_state of the LSTM
      vocab_size : Size of the vocabulary containing unique words
      embedding_length : Embeddding dimension of GloVe word embeddings
      weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
      
      --------
      return : logits, sentence embedding
      """
      
      
      self.batch_size = batch_size
      self.output_size = output_size
      self.hidden_size = hidden_size
      self.n_layers = n_layers
      

      # Pretrained
      self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix)) 
      # self.embedding = nn.Embedding(vocab_size, embedding_dim) 


      self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers = n_layers, bidirectional = True,  batch_first = True)

      self.attn = Attention(self.hidden_size*2)
      self.dropout = nn.Dropout(dropout)
      self.softmax = nn.Softmax(dim = 1)
      self.label = nn.Linear(hidden_size*2, output_size)


    def forward(self, input_sentences, batch_size=None):

      """ 
      Parameters
      ----------
      input_sentence: input_sentence of shape = (batch_size, num_sequences)
      batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
      
      Returns
      -------
      Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
      final_output.shape = (batch_size, output_size)
      
      """
      
      input = self.embedding(input_sentences)

      # print("input.shape: ",input.shape)

      #NEW LINES FOR SENTENCE PLOTTING
      # sentence_embedding = torch.mean(input, dim = 1)
      # print("input.shape: ",input.shape)
      # print("After mean: ", torch.mean(input, dim = 1).shape)
   
      if batch_size is None:
        h_0 = Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size).cuda())
      else:
        h_0 = Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size).cuda())
        
      output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0.detach(), c_0.detach())) # final_hidden_state.size() = (1, batch_size, hidden_size) 
      # output = output.permute(1, 2, 0) # output.size() = (batch_size, num_seq, hidden_size)
      
      # output = self.relu(output)

      
      # output = output[:, -1 , :]
      # attn_output = output
      # print("output.shape: ",output.shape)
      attn_output = self.attn(output)


      # # FC
      # print("attn_output.shape: ",attn_output.shape)

      attn_output = self.dropout(attn_output)
      
      logits = self.label(attn_output)
      logits = self.softmax(logits)
      # print("logits.shape",logits.shape)
      # logits = logits[-1,:,:]
      # print(logits)
      
      return logits, input, attn_output


class Attention(nn.Module):
    def __init__(self, dimension):
        super(Attention, self).__init__()
        self.wt = Parameter(torch.Tensor(dimension, dimension))
        self.u = nn.Linear(dimension, dimension)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, h):
        # h : Batch * timestep * dimension

        x = self.u(h)
        # u(h) : Batch * timestep * att_dim
        
        # softmax(x) : Batch * timestep * att_dim
        x = self.softmax(x)

        # Batch matrix multiplication
        output = x * h

        output = torch.sum(output, dim=1)

        return output
            
            
def max_length(tensor):
    return 50




def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len: padded[:] = x[:max_len]
    else: padded[:len(x)] = x
    return padded


def pytorch_embedding(df):
  vocabs = set()
  corpus = df.Data.to_list()
  for sentences in corpus:
    vocabs.update(sentences.split())

  print(vocabs)
  dict_vocab = {}
  for index, word in enumerate(vocabs):
    # print(index, word)
    dict_vocab[word] = index

  return dict_vocab

      






def converting_to_w2v(data, inputs, emotion, testy):
    
    
    ##Taking control of the unknown word
    
    input_tensor1 = []

    for es in data["Data"].values.tolist():
      
      temp = []
      for s in es.split():
        try:
          temp.append(inputs.word2idx[s])
        except:
          temp.append(0)
        temp.append(0)
      # if(testy == True):
      #   print("es: ", es)
      #   print("temp: ", temp)
      input_tensor1.append(temp)
    
    # print("input_tensor1 ", input_tensor1)
    max_length_inp = 50
    # print("max_length_inp: ",max_length_inp)

    
    
    input_tensor = [pad_sequences(x, max_length_inp) for x in input_tensor1]


    # for i in range(100):
    #   # print("Before padding: ", input_tensor1[i])
    #   print("After padding: ", input_tensor[i])

    
    ### convert targets to one-hot encoding vectors
    emotions = list(set(data[emotion].unique()))
    num_emotions = len(emotions)
  
    
    
    target_tensor = data[emotion].to_numpy()

    return input_tensor, target_tensor, num_emotions, input_tensor1, max_length_inp


def dataset_preprocessing( emotion, batchSize ):
    
    # fastText_wv = Word2Vec.load("/content/drive/My Drive/Research_Shanto/pretrained/test_w2v.model") 
    # fastText_wv = Word2Vec.load("/content/drive/MyDrive/Research_Shanto/pretrained/200D_model_skipgram/200D_model_skipgram/word2vec_model_200.model") #Pipilika
    # fastText_wv = Word2Vec.load("/content/drive/MyDrive/Research_Shanto/pretrained/600D_model_skipgram/600D_model_skipgram/word2vec_model_600.model")
    # ft = fasttext.load_model('/content/drive/MyDrive/Research_Shanto/pretrained/cc.bn.300.bin/cc.bn.300.bin')
    # fastText_wv = ft
    # fastText_wv = gensim.models.fasttext.load_facebook_vectors(datapath("/content/drive/MyDrive/Research_Shanto/pretrained/bengali_fasttext_wiki_bnlp.bin"))
    # File = '/content/drive/MyDrive/Research_Shanto/pretrained/cc.bn.300.bin/cc.bn.300.vec'
    # f = open(File,'r')
    # lst = []
    # wordss = []
    # cnt = 0
    # for line in f:
    #     cnt += 1
    #     if(cnt == 1):
    #       continue
    #     splitLines = line.split()
    #     word = splitLines[0]
    #     wordEmbedding = [float(value) for value in splitLines[1:]]
    #     # gloveModel[word] = wordEmbedding
    #     wordss.append(word)
    #     lst.append(wordEmbedding)

    # weights = torch.FloatTensor(lst)
    # print(weights)

    # weights = torch.FloatTensor(fastText_wv.wv.vectors)

    path_parent = os.path.dirname( os.getcwd() )

    os.chdir( path_parent )

    fastText_wv = gensim.models.KeyedVectors.load_word2vec_format( os.getcwd() + "/embeddings/cc.bn.300.vec" )

    weights = torch.from_numpy(fastText_wv.wv.vectors)
    # weights = 0
    # print(weights)

    # inputs = ConstructVocab(fastText_wv.wv.vocab)

    df_train_ = pd.read_csv( os.getcwd() + "/EmoNoBa Dataset/Final_Train.csv" )
    df_val_ =  pd.read_csv( os.getcwd() + "/EmoNoBa Dataset/Final_Val.csv" )
    df_test_ = pd.read_csv( os.getcwd() + "/EmoNoBa Dataset/Final_Test.csv" )

    dictions = pytorch_embedding(df_train_)
    inputs = ConstructVocab(dictions)

    df_train = df_train_[ ["Data", emotion]]

    df_val = df_val_[ ["Data", emotion]]

    df_test = df_test_[ ["Data", emotion]]

    # lst = []
    # for i in range(len(df_train)):
    #   lst.extend(df_train['Data'][i].split())

    # lst = list(set(lst))
    # File = '/content/drive/MyDrive/Research_Shanto/pretrained/cc.bn.300.bin/cc.bn.300.vec'
    # f = open(File,'r')
    # lst__ = []
    # wordss = []
    # cnt = 0
    # for line in f:
    #     cnt += 1
    #     if(cnt == 1):
    #       continue
    #     splitLines = line.split()
    #     word = splitLines[0]
    #     if(word in lst):
    #         wordEmbedding = [float(value) for value in splitLines[1:]]
    #         # gloveModel[word] = wordEmbedding
    #         wordss.append(word)
    #         lst__.append(wordEmbedding)

    # weights = torch.FloatTensor(lst__)
    # print(weights)

    # inputs = ConstructVocab(wordss)
    input_tensor_train, target_tensor_train, num_emotions, iktu, max_length = converting_to_w2v(df_train,inputs, emotion, testy = False)
    input_tensor_val, target_tensor_val, num_emotions, iktu, max_length = converting_to_w2v(df_val, inputs, emotion, testy= False)
    input_tensor_test, target_tensor_test, num_emotions, iktu, max_length = converting_to_w2v(df_test, inputs, emotion, testy = True)



    ###Split data
    
    # # Creating training and validation sets using an 80-20 split
    # input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.3)

    # # Split the validataion further to obtain a holdout dataset (for testing) -- split 50:50
    # input_tensor_val, input_tensor_test, target_tensor_val, target_tensor_test = train_test_split(input_tensor_val, target_tensor_val, test_size=0.5)

    # Show length
    print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val), len(input_tensor_test), len(target_tensor_test))
    
    
    
    ##Data Loader
    
    TRAIN_BUFFER_SIZE = len(input_tensor_train)
    VAL_BUFFER_SIZE = len(input_tensor_val)
    TEST_BUFFER_SIZE = len(input_tensor_test)
    BATCH_SIZE = batchSize
    TRAIN_N_BATCH = TRAIN_BUFFER_SIZE // BATCH_SIZE
    VAL_N_BATCH = VAL_BUFFER_SIZE // BATCH_SIZE
    TEST_N_BATCH = TEST_BUFFER_SIZE // BATCH_SIZE

    
    vocab_inp_size = len(inputs.word2idx)
    target_size = num_emotions
#     print(num_emotions)
    
    train_dataset = MyData(input_tensor_train, target_tensor_train)
    val_dataset = MyData(input_tensor_val, target_tensor_val)
    test_dataset = MyData(input_tensor_test, target_tensor_test)
    

    # sentences = dataset_to_list(test_dataset)
    # emne = []
    # for i in range(sentences):
    #   final = []
    #   final.append(sentences[i])
    #   final.append()

    # ccnt = 0
    # for sentences in test_dataset.data:
    #   st = ''
    #   for words in sentences:
    #     if(words == 0):
    #       st += ' '
    #     else:
    #       st += inputs.idx2word[words]
        
      
    #   print("st: ", st)
    #   # test_dataset.length
    #   print("Length: ",test_dataset.length[ccnt])
    #   print()
    #   ccnt += 1

    

    train_dataset = DataLoader(train_dataset, batch_size = BATCH_SIZE, 
                         drop_last=True,
                         shuffle=False)
    val_dataset = DataLoader(val_dataset, batch_size = BATCH_SIZE, 
                         drop_last=True,
                         shuffle=False)
    test_dataset = DataLoader(test_dataset, batch_size = BATCH_SIZE, 
                         drop_last=True,
                         shuffle=False)
    # sentences = []
    # for (batch, (inp, targ, lens)) in enumerate(test_dataset):
    #             # print(batch)
    #             print("Aghe inp: ", inp)
    #             # print("Bahire lens.numpy(): ",lens.numpy())
    #             for i in range(batch):
    #               try:
    #                 st = ''
                    
    #                 for j in range(lens[i]):
    #                   # print("Bhitore: ", lens.numpy()[i])
    #                   print("Bhitore: ",inp.numpy()[i][j])

    #                   if(inp[i][j] == 0):
    #                     st += ' '
    #                   else:
    #                     st += inputs.idx2word[inp.numpy()[i][j]]
    #                 print("Bhitore st: ", st)
    #                 sentences.append(st)
    #               except:
    #                 print()
                    

    # print(sentences)



    return train_dataset, val_dataset, test_dataset, input_tensor_train, input_tensor_val, input_tensor_test, weights, vocab_inp_size, inputs, max_length
    


def create_model(weights, vocab_size, embeddingDim, dropOut, numLayers, lstmUnits, batchSize):
  """
  Initializes the model
  Moves to GPU if found any

  Return: the model
  """
  print("vocab_size: ",vocab_size)
  embedding_dim = embeddingDim
  dropout = dropOut
  bidirectional = True
  n_layers = numLayers
  units = lstmUnits
  BATCH_SIZE = batchSize
  target_size = 2

  model = LSTM_Attn_Emo(vocab_size, weights, embedding_dim, BATCH_SIZE, target_size, units, n_layers, bidirectional, dropout)

  if torch.cuda.is_available():
    model = model.cuda()

    return model, embedding_dim, units

def accuracy(acc_targ, acc_pred):
    acc_tot = 0
    arr = confusion_matrix(acc_targ, acc_pred)
    for k in range(len(acc_targ)):
        if(acc_targ[k] == acc_pred[k]):
            acc_tot += 1
    
    # fp = arr[0][1] + arr[0][2] + arr[1][0] + arr[1][2] + arr[2][1] + arr[2][2]
    # fn = arr[1][0] + arr[2][0] + arr[0][1] + arr[2][1] + arr[0][2] + arr[1][2]
    # tp = arr[0][0] + arr[1][1] + arr[2][2]
    # print("tp: ", tp)
    # print("fp: ",fp)
    # print("fn: ", fn)
    # precision = tp/(tp+fp)
    # recall = tp/(tp+fn)
    # print("precision: ", round(tp/(tp+fp)*100,2) )
    # print("recall: ", round(tp/(tp+fn)*100,2) )
    # print("f1: ", round( ((2*precision*recall )/(precision+recall) )*100,2) )
    numenator = arr[1][1]
    precisionDeno = arr[1][1] + arr[1][0]
    recallDeno = arr[1][1] + arr[0][1]

            
    return numenator, precisionDeno, recallDeno, acc_tot/len(acc_targ)


def dataset_to_list(test_dataset):

  sentences = []    
  tars = []

  for (batch, (inp, targ, lens)) in enumerate(test_dataset):
          # print(batch)
          # print("Aghe inp: ", inp)
          # print("Bahire lens.numpy(): ",lens.numpy())
          
          tars.extend(targ.tolist())
          
          for i in range(inp.size(0)):
            # try:
            
            st = ''

            
            for j in range(inp.size(1)):
              
              # print("purata: ", purata)
              # print("sentence: ", len(sentences))
              # print("lens[i]: ", lens.numpy()[i])
              # print("j: ", j)
              # print("Bhitore: ", lens.numpy()[i])
              # print("Bhitore: ",inp.numpy()[i][j])

              if(inp.numpy()[i][j] == 0):
                st += ' '
              else:
                st += inputs.idx2word[inp.numpy()[i][j]]
                
                # print("Bhitore st: ", st)
            fin_st = ''
            tired = False
            for char in st:
              if(char == ' ' and tired == False):
                fin_st += ' '
                tired = True
              elif(char != ' '):
                fin_st += char
                tired = False               

            
            sentences.append(fin_st)
          # except:
            
  return sentences, tars


def train(model, iterator, optimizer, tot_len, max_length, embedding_dim):
    
    epoch_loss = 0
    
    acc_targ = []
    acc_pred = []
    
    model.train()
    word_embedding = torch.empty(((0,max_length, embedding_dim)))
    
    cnt = 0
    
    if(torch.cuda.is_available()):
      word_embedding = word_embedding.cuda()
    
    for (batch, (inp, targ, lens)) in enumerate(iterator):
        
        cnt += 1
        
        if(torch.cuda.is_available()):
            inp = inp.cuda()
            targ = targ.cuda()
        
        optimizer.zero_grad()
        
        
        predictions, input, sentence_embedding = model(inp)

        word_embedding = torch.cat((word_embedding,input),dim = 0)
        

        criterion = nn.CrossEntropyLoss()
        

        loss = criterion(predictions, targ)

        # print("predictions: ",predictions)
        # print("target: ", targ)
        
        

        
        loss.backward()
        
        
        optimizer.step()


        
        epoch_loss += loss.data.item()
        
        acc_targ.extend(targ.tolist())

        acc_pred.extend(torch.max(predictions, 1)[1].tolist())
    

    return epoch_loss / len(iterator), acc_targ, acc_pred, word_embedding



def evaluate(model, iterator, tot_len, embedding_dim, inputs, units):
    
    epoch_loss = 0
    acc_targ = []
    acc_pred = []
    index_tensors = []
    
    model.eval()
    
    for_projection = torch.empty((0,units*2))
    # word_embedding = torch.empty((0,embedding_dim))
    if(torch.cuda.is_available()):
      for_projection = for_projection.cuda()

    sentences = []

    with torch.no_grad():
      for (batch, (inp, targ, lens)) in enumerate(iterator):

            if(torch.cuda.is_available()):
                inp = inp.cuda()
                targ = targ.cuda()

            predictions, input, sentence_embedding = model(inp)
            # word_embedding = torch.cat((word_embedding,input),dim = 0)
                 
            for_projection = torch.cat((for_projection,sentence_embedding),dim = 0)
            # print("for_projection: ",for_projection.shape)

            criterion = nn.CrossEntropyLoss()
            
            loss = criterion(predictions, targ)           

            for i in range(inp.size(0)):
            # try:
            
              st = ''

            
              for j in range(inp.size(1)):
                
                # print("purata: ", purata)
                # print("sentence: ", len(sentences))
                # print("lens[i]: ", lens.numpy()[i])
                # print("j: ", j)
                # print("Bhitore: ", lens.numpy()[i])
                # print("Bhitore: ",inp.numpy()[i][j])

                if(inp.cpu().numpy()[i][j] == 0):
                  st += ' '
                else:
                  st += inputs.idx2word[inp.cpu().numpy()[i][j]]
                  
                  # print("Bhitore st: ", st)
              fin_st = ''
              tired = False
              for char in st:
                if(char == ' ' and tired == False):
                  fin_st += ' '
                  tired = True
                elif(char != ' '):
                  fin_st += char
                  tired = False               

              # print("fin_st: ",fin_st)
              sentences.append(fin_st)

            acc_targ.extend(targ.tolist())
            # print(targ.tolist())
            # predictions = torch.nn.Softmax(predictions)
            # print("predictions: ", predictions)
            acc_pred.extend(torch.max(predictions, 1)[1].tolist())
            index_tensors.extend(torch.max(predictions, 1)[0].tolist())
            # print(torch.max(predictions, 1)[0])
            

            epoch_loss += loss.data.item()

            
      


    
    return epoch_loss / len(iterator), acc_targ, acc_pred, for_projection, index_tensors, sentences

def training_loop(train_dataset, val_dataset, input_tensor_train, input_tensor_val,test_dataset, input_tensor_test, weights, vocab_size, inputs, max_len, embeddingDim, dropOut, numLayers, lstmUnits, batchSize, learningRate, maxEpoch):
    """
    :return:
    """
    model, embedding_dim, units = create_model(weights, vocab_size, embeddingDim, dropOut, numLayers, lstmUnits, batchSize)

    optimizer = optim.Adam( model.parameters(), lr=learningRate)

    max_epochs = maxEpoch

    best_valid_acc = 0


    for epoch in range(max_epochs):

        print('[Epoch %d]' % (epoch + 1))

        train_loss, cnt_targ, cnt_pred, input= train(model, train_dataset, optimizer, len(input_tensor_train), max_len, embedding_dim)
        
        numenatorTrain, precisionDenoTrain, recallDenoTrain, train_acc = accuracy(cnt_targ, cnt_pred)

        val_loss, cnt_targ, cnt_pred, _, index_tensors, _ = evaluate(model, val_dataset, len(input_tensor_val), embedding_dim, inputs, units)
        
        numenatorVal, precisionDenoVal, recallDenoVal, val_acc = accuracy(cnt_targ, cnt_pred)





        # print('Training Loss %.5f, Validation Loss %.5f' % (train_loss, val_loss))
        # print('Training Accuracy %.5f, Validation Accuracy %.5f' % (train_acc, val_acc) )

        # history['val_loss'].append(val_loss)
        # history['train_loss'].append(train_loss)



        # writer.add_scalars('Loss', {'Train_Loss':train_loss,
        #                         'Val_loss':val_loss}, epoch)
        
        # writer.add_scalars('Accuracy', {'Train_Acc':train_acc,
        #                 'Val_Acc':val_acc}, epoch)
        

        
        # history['val_acc'].append(val_acc)
        # history['train_acc'].append(train_acc)

        if val_acc > best_valid_acc:
          best_valid_acc = val_acc
          torch.save(model.state_dict(), 'tut6-model.pt')



        print()



    model.load_state_dict(torch.load('./tut6-model.pt'))

    
    
    loss, cnt_targ, cnt_pred, sentence_embedding, index_tensors, sentences  = evaluate(model, test_dataset, len(input_tensor_test), embedding_dim, inputs, units)

    # print("Test Data: ", cnt_targ)

    print("sentence_embedding: ",sentence_embedding.shape)

    # print(cnt_pred)
    # print(len(cnt_pred))

    sentence_for_embedding, _ = dataset_to_list(train_dataset)

    # for_tensorboard(input,sentence_for_embedding)

    sentences_, tar = dataset_to_list(test_dataset)
  
    


    # print("purata: ",purata)
    # print("purata_upre: ", purata_upre)
    # print(sentences)
    # label_tensor = torch.Tensor(cnt_pred)
    # if(torch.cuda.is_available()):
    #   label_tensor = label_tensor.cuda()
    # print("cnt_pred: ",cnt_pred)
    # modi_sent = []
    # for i in range(len(cnt_pred)):
    #   st = ''
    #   st += str(cnt_pred[i]) + ' ' + str(sentences[i])
    #   modi_sent.append(st)
    # converting2tsv(sentence_embedding, metadata=modi_sent, tag = "test_809_skip_both_finetuned", w2v = False)


    # final_txt = []
    # print("len(sentences): ",len(sentences))
    # for i in range(len(cnt_targ)):
    #   temp = []
    #   temp.append(cnt_targ[i])
    #   temp.append(tar[i])
    #   temp.append(sentences[i])
    #   temp.append(cnt_pred[i])
    #   temp.append(index_tensors[i])
      
    #   final_txt.append(temp)

    



    # converting2tsv(sentence_embedding, metadata=sentences, tag = "test_480_skip_sentence_finetuned")

    # writer.add_embedding(sentence_embedding, metadata=cnt_pred)

    # deil = TSNE(n_components=2).fit_transform(corpus_embeddings_test)


    # writer.add_scalars('Loss', {'Train_Loss':train_loss,
    #                         'Val_loss':val_loss}, epoch)
    
    # writer.add_scalars('Accuracy', {'Train_Acc':train_acc,
    #                 'Val_Acc':val_acc}, epoch)    

    print("TEST RESULTSS")
    numenator, precisionDeno, recallDeno, accTest = accuracy(cnt_targ, cnt_pred)
    # print(cnt_pred)
    
    # print("Test Accuracy: ", result)
    
    # loss, cnt_targ, cnt_pred, _, index_tensors, _ = evaluate(model, val_dataset, len(input_tensor_val), embedding_dim, inputs, units)
    val_loss, val_cnt_targ, val_cnt_pred, val_sentence_embedding, val_index_tensors, val_sentences  = evaluate(model, val_dataset, len(input_tensor_val), embedding_dim, inputs, units)
    loss, cnt_targ, cnt_pred, sentence_embedding, index_tensors, sentences  = evaluate(model, test_dataset, len(input_tensor_test), embedding_dim, inputs, units)
    # print("VALIDATION RESULTSS")
    # numenatorFinal, precisionDenoFinal, recallDenoFinal, accFinal = accuracy(cnt_targ, cnt_pred)
    # # print(cnt_pred)
    
    # print("Val Accuracy: ", result)
    
    # loss, cnt_targ, cnt_pred, _, index_tensors, _ = evaluate(model, train_dataset, len(input_tensor_train),embedding_dim, inputs, units)
    
    # result = accuracy(cnt_targ, cnt_pred)
    # # print(cnt_pred)
    
    # print("Train Result: ", result)

    # return numenatorFinal, precisionDenoFinal, recallDenoFinal,

    print(val_cnt_targ, val_cnt_pred, cnt_targ, cnt_pred)
    return val_cnt_targ, val_cnt_pred, cnt_targ, cnt_pred

    


    
    

    

def test(test_dataset, input_tensor_test, weights):
    model = create_model(weights)

    model.load_state_dict(torch.load('tut6-model.pt'))

    loss, result = evaluate(model, test_dataset, len(input_tensor_test))
    print("Test Accuracy: ", result)

def plot_loss(history):
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.title('Loss history')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1.5]);
    plt.savefig('Loss')

def plot_acc(history):
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')
    plt.title('Acc history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('Accuracy')
    

if __name__ == '__main__':

    finalList = []

    try:
      df = pd.read_csv('/content/drive/MyDrive/Research_Shanto/Datasets/Bangla Emotion Dataset/Final/modelData/result_admin_BiLSTM_Attention_random.csv') # designate a path where you would want to save and update your accuracy log (try manipulating with the `temp` variable by appending to this dataframe)
    except:
      df = pd.DataFrame(finalList, columns = ['Batch_Size', 'Embedding dim', 'Drop-out', 'num_layers', 'LSTM units', 'lr', 'epochs','Macro-Averaged Precision Val', 'Micro-Averaged Precision Val', 'Macro-Averaged Recall Val', 'Micro-Averaged Recall Val', 'Macro-Averaged F1 Val', 'Micro-Averaged F1 Val', 'Accuracy Val', 'Macro-Averaged Precision Test', 'Micro-Averaged Precision Test', 'Macro-Averaged Recall Test', 'Micro-Averaged Recall Test', 'Macro-Averaged F1 Test', 'Micro-Averaged F1 Test', 'Accuracy Test'])
    temp = []
    batchSize = 64
    embeddingDim = 300 
    dropOut = 0.30
    numLayers = 2
    lstmUnits = 64
    learningRate = 0.0001
    maxEpoch = 40
    temp.append( batchSize )
    temp.append( embeddingDim )
    temp.append( dropOut )
    temp.append( numLayers )
    temp.append( lstmUnits )
    temp.append( learningRate )
    temp.append( maxEpoch )

    f1_dic = dict()

    scores = []

    emotions = [ "Love", "Joy", "Surprise", "Anger", "Sadness", "Fear"]
    writer = SummaryWriter()
  
    
    for emotion in emotions:
      train_dataset, val_dataset, test_dataset, input_tensor_train, input_tensor_val, input_tensor_test, weights, vocabs, inputs, max_length = dataset_preprocessing( emotion, batchSize)
      val_cnt_targ, val_cnt_pred, test_cnt_targ, test_cnt_pred = training_loop(train_dataset, val_dataset, input_tensor_train, input_tensor_val, test_dataset, input_tensor_test, weights, vocabs, inputs, max_length, embeddingDim, dropOut, numLayers, lstmUnits, batchSize, learningRate, maxEpoch)

      f1 = f1_score( test_cnt_targ, test_cnt_pred )

      scores.append( round(f1 * 100, 2) )

      f1_dic[emotion] = round(f1 * 100, 2)
        
    print(f1_dic)
    print( scores )
    



