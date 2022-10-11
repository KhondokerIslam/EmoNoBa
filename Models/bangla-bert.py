import os
import logging
import transformers
from transformers import BertModel, BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# from smooth_gradient import SmoothGradient
# from integrated_gradient import IntegratedGradient

from IPython.display import display, HTML

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-13T17:08:18.735489Z","iopub.execute_input":"2021-08-13T17:08:18.735788Z","iopub.status.idle":"2021-08-13T17:08:18.744036Z","shell.execute_reply.started":"2021-08-13T17:08:18.735741Z","shell.execute_reply":"2021-08-13T17:08:18.743213Z"}}

# sample_txt = 'আজকে আপনার সাফল্য কামনা করি'
# tokens = tokenizer.tokenize(sample_txt)
# # print(tokenizer.prepare_for_tokenization(sample_txt,is_pretokenized= False))
# tok_to_int = tokenizer.convert_tokens_to_ids(tokens)
# print('Sentence: ', sample_txt)
# print('Tokens: ', tokens)
# print('To Int: ', tok_to_int)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-13T17:08:18.752667Z","iopub.execute_input":"2021-08-13T17:08:18.753282Z","iopub.status.idle":"2021-08-13T17:08:18.763653Z","shell.execute_reply.started":"2021-08-13T17:08:18.752964Z","shell.execute_reply":"2021-08-13T17:08:18.76276Z"}}
class ShantoDataset(Dataset):
  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews =  reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, item):
    reviews = str(self.reviews[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
        reviews,
        add_special_tokens = True,
        max_length = self.max_len,
        return_token_type_ids=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    return {
      'review_text': reviews,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-13T17:08:18.76508Z","iopub.execute_input":"2021-08-13T17:08:18.765715Z","iopub.status.idle":"2021-08-13T17:08:18.774973Z","shell.execute_reply.started":"2021-08-13T17:08:18.765385Z","shell.execute_reply":"2021-08-13T17:08:18.774178Z"}}
def create_data_loader(df, emotion, tokenizer, max_len, batch_size):
  ds = ShantoDataset(
    reviews=df.Data.to_numpy(),
    targets=df[emotion].to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size
  )



# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-17T16:40:36.994879Z","iopub.execute_input":"2021-08-17T16:40:36.995163Z","iopub.status.idle":"2021-08-17T16:40:37.072414Z","shell.execute_reply.started":"2021-08-17T16:40:36.995115Z","shell.execute_reply":"2021-08-17T16:40:37.070850Z"}}
# bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)


class EmoClassifier(nn.Module):
  def __init__(self, PRE_TRAINED_MODEL_NAME, dropOut, n_classes):
    super(EmoClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=dropOut)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.softmax = nn.LogSoftmax(dim = 1)
#     self.softmax = F.log_softmax(dim = 1)
    
  def forward(self, input_ids, attention_mask):
    last_hidden, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
#     print(pooled_output.shape)
    output = self.drop(pooled_output)
    output = self.out(output)
    output = self.softmax(output)
#     output = nn.LogSoftmax(output)
#     p = torch.relu(last_hidden)
#     for k in input_ids:
#         print(tokenizer.convert_ids_to_tokens(k))
#     print("p.shape",p.shape)
#     print(p)
    
#     return  emne, self.out(output)
    return output

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-08-13T17:08:38.407731Z","iopub.execute_input":"2021-08-13T17:08:38.40804Z","iopub.status.idle":"2021-08-13T17:08:38.416891Z","shell.execute_reply.started":"2021-08-13T17:08:38.407981Z","shell.execute_reply":"2021-08-13T17:08:38.416203Z"}}

def instantiateDataset( batchSize, maxLen, emotion, tokenizer ):

    path_parent = os.path.dirname( os.getcwd() )
    os.chdir( path_parent )
    df_train = pd.read_csv( os.getcwd() + "/EmoNoBa Dataset/Final_Train.csv" )
    df_val =  pd.read_csv( os.getcwd() + "/EmoNoBa Dataset/Final_Val.csv" )
    df_test = pd.read_csv( os.getcwd() + "/EmoNoBa Dataset/Final_Test.csv" )

    BATCH_SIZE = batchSize

    MAX_LEN = maxLen

    train_data_loader = create_data_loader(df_train, emotion, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, emotion, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, emotion, tokenizer, MAX_LEN, BATCH_SIZE)
    
    return train_data_loader, val_data_loader, test_data_loader, df_train, df_val, df_test

    
def instantiateModel(PRE_TRAINED_MODEL_NAME, dropOut):
    model = EmoClassifier(PRE_TRAINED_MODEL_NAME, dropOut, 3)

    model = model.to(device)    

    for name, param in model.named_parameters():                
        if name.startswith('bert'):
            param.requires_grad = False
            
    return model



def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  n_examples,
  thresh
):
    model.train()
    losses = []
    targs = []
    preds = []
    correct_predictions = 0
#   check = False
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
          input_ids = input_ids,
          attention_mask=attention_mask
    )
#     print("outputs: ", outputs)
    # print(outputs)
#     _, preds = torch.max(F.softmax(outputs, dim = 1), dim=1)
#     correct_predictions += torch.sum(preds == targets)
    
    # correct_predictions += jaccard_index(targets, outputs, thresh)
    # correct_predictions += accuracy_thresh(outputs, targets)
    # print("correct_predictions: ",correct_predictions)
        optimizer.zero_grad()
#     print(outputs)
        loss = loss_fn(outputs, targets)
    
    # print(preds)
    
        losses.append(loss.item())
        loss.backward()
    
#     nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
#     scheduler.step()
   
#     outputs = F.softmax(outputs, dim=1)
#     outputs = outputs.long()
#     targets = targets.long()
        preds = torch.max(outputs, 1)[1]
#     cnt = 0
#     for val in preds.tolist():
#         if(val == 0):
#            cnt += 1
    
#     print("Number of Negative: ",cnt)
#     outputs = outputs > thresh
#     outputs = outputs.long()
#     targets = targets.long()
        correct_predictions += torch.sum(preds == targets)
#         print( "n_examples: ", n_examples )
#         print("correct_predictions: ", correct_predictions)
#         print( "preds: ", len(preds) )
#         print("targets: ", len(targets) )
#     print("outputs",outputs)
#     print("targets",targets)
#     targs.extend(targets.detach().cpu().numpy())
#     preds.extend(outputs.detach().cpu().numpy())
#     print(preds)
  
  # correct_predictions = jaccard_index(targs, preds,thresh)
    return correct_predictions.double()/n_examples, np.mean(losses)



def eval_model(model, data_loader, loss_fn, device, n_examples,thresh):
  model.eval()
  losses = []
  targs = []
  preds = []
  correct_predictions = 0
  check = False
  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      outputs = model(
        input_ids = input_ids,
        attention_mask=attention_mask
      )
      loss = loss_fn(outputs, targets)
      losses.append(loss.item())
#       outputs = F.softmax(outputs, dim=1)
#       outputs = outputs.long()
#       targets = targets.long()
      preds = torch.max(outputs, 1)[1]
      # outputs = outputs > thresh
      # outputs = outputs.long()
      # targets = targets.long()
      # targs.extend(targets.detach().cpu().numpy())
      # preds.extend(outputs.detach().cpu().numpy())

#       _, preds = torch.max(F.softmax(outputs, dim = 1), dim=1)
      
      correct_predictions += torch.sum(preds == targets)

      # correct_predictions += jaccard_index(targets, outputs,thresh)
      # correct_predictions += accuracy_thresh(outputs, targets)
      
  
  # correct_predictions = jaccard_index(targs, preds,thresh)
     
  return correct_predictions.double()/n_examples, np.mean(losses)


def get_training( model, df_train, df_val, train_data_loader, val_data_loader,  threshHold, epochRuns, learningRate ):
    history = defaultdict(list)
    best_accuracy = 0
    thresh = threshHold
    EPOCHS = epochRuns
    optimizer = AdamW(model.parameters(), lr=learningRate)
    total_steps = len(train_data_loader) * EPOCHS
    loss_fn = nn.NLLLoss().to(device)
    for epoch in range(EPOCHS):
      print(f'Epoch {epoch + 1}/{EPOCHS}')
      print('-' * 10)
      train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
    #     scheduler,
        len(df_train),
        thresh
      )
      print(f'Train loss {train_loss} accuracy {train_acc}')
      val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val),
        thresh
      )
      print(f'Val   loss {val_loss} accuracy {val_acc}')
      print()
      history['train_acc'].append(train_acc)
      history['train_loss'].append(train_loss)
      history['val_acc'].append(val_acc)
      history['val_loss'].append(val_loss)
      if val_acc > best_accuracy:
    #     torch.save(model.state_dict(), 'best_model_state.bin')
        torch.save(model.state_dict(), 'tut6-model.pt')
        best_accuracy = val_acc
    

def get_predictions(model, data_loader):
  model.eval()
  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = []
  with torch.no_grad():
    for d in data_loader:
      texts = d["review_text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
#       preds = torch.max(outputs, 1)[1]
#       _, outputs = torch.max(F.softmax(outputs, dim = 1), dim=1)
#       outputs = F.softmax(outputs, dim=1)
#     outputs = outputs.long()
      preds = torch.max(outputs, 1)[1].detach().cpu().numpy()
#       outputs = outputs.sigmoid()
#       outputs = outputs > thresh
#       outputs = outputs.detach().cpu().numpy()
      targets = targets.long()
      targets = targets.detach().cpu().numpy()
      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(outputs)
      real_values.extend(targets)
#   predictions = torch.stack(predictions).cpu()
#   prediction_probs = torch.stack(prediction_probs).cpu()
#   real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values

def RecallPrecisionFScore(y_test, pred):
    arr = confusion_matrix(y_test, pred)
    numenator = arr[1][1]
    precisionDeno = arr[1][1] + arr[1][0]
    recallDeno = arr[1][1] + arr[0][1]
    
    return numenator, precisionDeno, recallDeno,




def getResult( test_data_loader ):
    
    model.load_state_dict(torch.load('./tut6-model.pt'))
    
    y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
      model,
      test_data_loader
    )
    print(y_pred)
    print()
    print(y_test)
    class_names = df_test.columns.to_list()
    
    return y_pred, y_test
    
    

if __name__ == '__main__':
    
    finalList = []
    
    scores = []
    
    batchSize = 128
    maxLen = 50
    dropOut = 0.30
    threshHold = 0.2
    epochRuns = 80
    learningRate = 2e-2
    
    f1_dic = dict()
    
    temp = []
    
    temp.append( batchSize )
    temp.append( maxLen )
    temp.append( dropOut )
    temp.append( threshHold )
    temp.append( learningRate )
    temp.append( epochRuns )
    
    logging.basicConfig(level=logging.ERROR)
    PRE_TRAINED_MODEL_NAME = 'sagorsarker/bangla-bert-base'
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    
    emotions = [ "Love", "Joy", "Surprise", "Anger", "Sadness", "Fear"]
    
    
    for emotion in emotions:
        
        print( emotion )
    
        train_data_loader, val_data_loader, test_data_loader, df_train, df_val, df_test = instantiateDataset( batchSize, maxLen, emotion, tokenizer )
        model = instantiateModel( PRE_TRAINED_MODEL_NAME, dropOut )
        get_training( model, df_train, df_val, train_data_loader, val_data_loader,  threshHold, epochRuns, learningRate )
        pred_test, targ_test = getResult(test_data_loader)
        
        f1 = f1_score( targ_test, pred_test )

        scores.append( round(f1 * 100, 2) )

        f1_dic[emotion] = round(f1 * 100, 2)

        
        
    print( f1_dic )
    # print( scores )
    print( "Macro Average F1-score: ", float('{0:.2f}'.format(sum(scores)/len(scores))) )
    
