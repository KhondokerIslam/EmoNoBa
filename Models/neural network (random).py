import os
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
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
import itertools
import csv
import pandas as pd
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import spacy
from sklearn.manifold import TSNE

import multiprocessing
from torch.utils.data import Dataset, DataLoader


from torch.autograd import Variable
from torch.nn.parameter import Parameter
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from torch.nn.parameter import Parameter

random.seed(50)

class MyData(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y
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
            


class LSTM_Attn_Sentiment(torch.nn.Module):
    def __init__(self,  vocab_size, embedding_dim, batch_size, output_size, hidden_size, n_layers, bidirectional,
                  dropout, NUM_FILTERS =10, window_sizes=(1,2,3,5)):
      super(LSTM_Attn_Sentiment, self).__init__()
      
      self.batch_size = batch_size
      self.output_size = output_size
      self.hidden_size = hidden_size
      self.n_layers = n_layers
      
      self.embedding = nn.Embedding(vocab_size, embedding_dim) 


      self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers = n_layers, bidirectional = True,  batch_first = True)

      self.attn = Attention(self.hidden_size*2)
      self.dropout = nn.Dropout(dropout)
      self.softmax = nn.Softmax(dim = 1)
      self.label = nn.Linear(hidden_size*2, output_size)


    def forward(self, input_sentences, batch_size=None):
      
      input = self.embedding(input_sentences)
   
      if batch_size is None:
        h_0 = Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size).cuda())
      else:
        h_0 = Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size).cuda())
        c_0 = Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.hidden_size).cuda())
        
      output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0.detach(), c_0.detach())) # final_hidden_state.size() = (1, batch_size, hidden_size) 
      
      attn_output = self.attn(output)

      attn_output = self.dropout(attn_output)
      
      logits = self.label(attn_output)
      logits = self.softmax(logits)
      
      return logits, input, attn_output


class Attention(nn.Module):
    def __init__(self, dimension):
        super(Attention, self).__init__()
        self.wt = Parameter(torch.Tensor(dimension, dimension))
        self.u = nn.Linear(dimension, dimension)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)

    def forward(self, h):

        x = self.u(h)

        x = self.softmax(x)

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

    path_parent = os.path.dirname( os.getcwd() )
    os.chdir( path_parent )
    df_train_ = pd.read_csv( os.getcwd() + "/EmoNoBa Dataset/Final_Train.csv" )
    df_val_ =  pd.read_csv( os.getcwd() + "/EmoNoBa Dataset/Final_Val.csv" )
    df_test_ = pd.read_csv( os.getcwd() + "/EmoNoBa Dataset/Final_Test.csv" )

    dictions = pytorch_embedding(df_train_)
    inputs = ConstructVocab(dictions)

    df_train = df_train_[ ["Data", emotion]]

    df_val = df_val_[ ["Data", emotion]]

    df_test = df_test_[ ["Data", emotion]]

    input_tensor_train, target_tensor_train, num_emotions, iktu, max_length = converting_to_w2v(df_train,inputs, emotion, testy = False)
    input_tensor_val, target_tensor_val, num_emotions, iktu, max_length = converting_to_w2v(df_val, inputs, emotion, testy= False)
    input_tensor_test, target_tensor_test, num_emotions, iktu, max_length = converting_to_w2v(df_test, inputs, emotion, testy = True)


    print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val), len(input_tensor_test), len(target_tensor_test))
    
    
    TRAIN_BUFFER_SIZE = len(input_tensor_train)
    VAL_BUFFER_SIZE = len(input_tensor_val)
    TEST_BUFFER_SIZE = len(input_tensor_test)
    BATCH_SIZE = batchSize
    TRAIN_N_BATCH = TRAIN_BUFFER_SIZE // BATCH_SIZE
    VAL_N_BATCH = VAL_BUFFER_SIZE // BATCH_SIZE
    TEST_N_BATCH = TEST_BUFFER_SIZE // BATCH_SIZE

    
    vocab_inp_size = len(inputs.word2idx)
    target_size = num_emotions

    train_dataset = MyData(input_tensor_train, target_tensor_train)
    val_dataset = MyData(input_tensor_val, target_tensor_val)
    test_dataset = MyData(input_tensor_test, target_tensor_test)
    

    train_dataset = DataLoader(train_dataset, batch_size = BATCH_SIZE, 
                         drop_last=True,
                         shuffle=False)
    val_dataset = DataLoader(val_dataset, batch_size = BATCH_SIZE, 
                         drop_last=True,
                         shuffle=False)
    test_dataset = DataLoader(test_dataset, batch_size = BATCH_SIZE, 
                         drop_last=True,
                         shuffle=False)



    return train_dataset, val_dataset, test_dataset, input_tensor_train, input_tensor_val, input_tensor_test, vocab_inp_size, inputs, max_length
    


def create_model(vocab_size, embeddingDim, dropOut, numLayers, lstmUnits, batchSize):
    
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

    model = LSTM_Attn_Sentiment(vocab_size, embedding_dim, BATCH_SIZE, target_size, units, n_layers, bidirectional, dropout)

    if torch.cuda.is_available():
        model = model.cuda()

    return model, embedding_dim, units

def accuracy(acc_targ, acc_pred):
    acc_tot = 0
    arr = confusion_matrix(acc_targ, acc_pred)
    for k in range(len(acc_targ)):
        if(acc_targ[k] == acc_pred[k]):
            acc_tot += 1

    numenator = arr[1][1]
    precisionDeno = arr[1][1] + arr[1][0]
    recallDeno = arr[1][1] + arr[0][1]

            
    return numenator, precisionDeno, recallDeno, acc_tot/len(acc_targ)




def for_tensorboard(input, sentence):
  dictions = {}
  print("input.shape: ", input.shape)
  print("len(sentence): ",len(sentence))
  for i in range(input.size(0)):
    for j in range(input.size(1)):
      dictions[sentence[i][j]] = input.cpu().detach().numpy()[i][j][:]


def dataset_to_list(test_dataset):

  sentences = []    
  tars = []

  for (batch, (inp, targ, lens)) in enumerate(test_dataset):
          
          tars.extend(targ.tolist())
          
          for i in range(inp.size(0)):
            
            st = ''

            
            for j in range(inp.size(1)):

              if(inp.numpy()[i][j] == 0):
                st += ' '
              else:
                st += inputs.idx2word[inp.numpy()[i][j]]

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
    if(torch.cuda.is_available()):
      for_projection = for_projection.cuda()

    sentences = []

    with torch.no_grad():
      for (batch, (inp, targ, lens)) in enumerate(iterator):

            if(torch.cuda.is_available()):
                inp = inp.cuda()
                targ = targ.cuda()

            predictions, input, sentence_embedding = model(inp)
                 
            for_projection = torch.cat((for_projection,sentence_embedding),dim = 0)

            criterion = nn.CrossEntropyLoss()
            
            loss = criterion(predictions, targ)           

            for i in range(inp.size(0)):
            # try:
            
              st = ''

            
              for j in range(inp.size(1)):

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
        
            acc_pred.extend(torch.max(predictions, 1)[1].tolist())
            index_tensors.extend(torch.max(predictions, 1)[0].tolist())

            epoch_loss += loss.data.item()     

    return epoch_loss / len(iterator), acc_targ, acc_pred, for_projection, index_tensors, sentences


def training_loop(train_dataset, val_dataset, input_tensor_train, input_tensor_val,test_dataset, input_tensor_test, vocab_size, inputs, max_len, embeddingDim, dropOut, numLayers, lstmUnits, batchSize, learningRate, maxEpoch):
    """
    :return:
    """
    model, embedding_dim, units = create_model( vocab_size, embeddingDim, dropOut, numLayers, lstmUnits, batchSize)

    optimizer = optim.Adam( model.parameters(), lr=learningRate)

    max_epochs = maxEpoch

    best_valid_acc = 0


    for epoch in range(max_epochs):

        print('[Epoch %d]' % (epoch + 1))

        train_loss, cnt_targ, cnt_pred, input= train(model, train_dataset, optimizer, len(input_tensor_train), max_len, embedding_dim)
        
        numenatorTrain, precisionDenoTrain, recallDenoTrain, train_acc = accuracy(cnt_targ, cnt_pred)

        val_loss, cnt_targ, cnt_pred, _, index_tensors, _ = evaluate(model, val_dataset, len(input_tensor_val), embedding_dim, inputs, units)
        
        numenatorVal, precisionDenoVal, recallDenoVal, val_acc = accuracy(cnt_targ, cnt_pred)


        if val_acc > best_valid_acc:
          best_valid_acc = val_acc
          torch.save(model.state_dict(), 'tut6-model.pt')



        print()



    model.load_state_dict(torch.load('./tut6-model.pt'))
    
    
    loss, cnt_targ, cnt_pred, sentence_embedding, index_tensors, sentences  = evaluate(model, test_dataset, len(input_tensor_test), embedding_dim, inputs, units)

    sentence_for_embedding, _ = dataset_to_list(train_dataset)

    sentences_, tar = dataset_to_list(test_dataset)

    numenator, precisionDeno, recallDeno, accTest = accuracy(cnt_targ, cnt_pred)
    
    val_loss, val_cnt_targ, val_cnt_pred, val_sentence_embedding, val_index_tensors, val_sentences  = evaluate(model, val_dataset, len(input_tensor_val), embedding_dim, inputs, units)
    loss, cnt_targ, cnt_pred, sentence_embedding, index_tensors, sentences  = evaluate(model, test_dataset, len(input_tensor_test), embedding_dim, inputs, units)
    

    print(val_cnt_targ, val_cnt_pred, cnt_targ, cnt_pred)
    return val_cnt_targ, val_cnt_pred, cnt_targ, cnt_pred



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
      df = pd.read_csv('/content/drive/MyDrive/Research_Shanto/Datasets/Bangla Emotion Dataset/Final/modelData/result_admin_BiLSTM_Attention_random.csv')
    except:
      df = pd.DataFrame(finalList, columns = ['Batch_Size', 'Embedding dim', 'Drop-out', 'num_layers', 'LSTM units', 'lr', 'epochs','Macro-Averaged Precision Val', 'Micro-Averaged Precision Val', 'Macro-Averaged Recall Val', 'Micro-Averaged Recall Val', 'Macro-Averaged F1 Val', 'Micro-Averaged F1 Val', 'Accuracy Val', 'Macro-Averaged Precision Test', 'Micro-Averaged Precision Test', 'Macro-Averaged Recall Test', 'Micro-Averaged Recall Test', 'Macro-Averaged F1 Test', 'Micro-Averaged F1 Test', 'Accuracy Test'])
    temp = []
    batchSize = 64
    embeddingDim = 300 
    dropOut = 0.30
    numLayers = 2
    lstmUnits = 128
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
      train_dataset, val_dataset, test_dataset, input_tensor_train, input_tensor_val, input_tensor_test, vocabs, inputs, max_length = dataset_preprocessing( emotion, batchSize)
      val_cnt_targ, val_cnt_pred, test_cnt_targ, test_cnt_pred = training_loop(train_dataset, val_dataset, input_tensor_train, input_tensor_val, test_dataset, input_tensor_test, vocabs, inputs, max_length, embeddingDim, dropOut, numLayers, lstmUnits, batchSize, learningRate, maxEpoch)

      f1 = f1_score( test_cnt_targ, test_cnt_pred )

      scores.append( round(f1 * 100, 2) )

      f1_dic[emotion] = round(f1 * 100, 2)
        
    print(f1_dic)
    print( scores )
    print( "Macro Average F1-score: ", float('{0:.2f}'.format(sum(scores)/len(scores))) )
    
