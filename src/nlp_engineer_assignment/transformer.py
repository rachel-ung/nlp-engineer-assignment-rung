import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchtext.vocab import vocab
from torchtext.transforms import VocabTransform
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
from collections import OrderedDict
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from .utils import count_letters, score
import json

vocabs = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
max_length = 20
model_dim = 512
dropout = 0.1
n_layers = 2

class MyDataset(Dataset):
    def __init__(self, data, vocab, func): #, transform=None, target_transform=None):
        self.data = data
        self.vocab = vocab
        self.function = func

        self.src = self.tokenize_src(self.data, self.vocab)
        self.tar = self.tokenize_tar(self.data, self.function)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tar[idx]

    @staticmethod
    def tokenize_src(data, vocabulary):
        vocab_obj = vocab(OrderedDict([(token, 1) for token in vocabulary]))
        vocab_transform = VocabTransform(vocab_obj)
        output = vocab_transform(pd.DataFrame(data)[0].apply(lambda x: [*x]).tolist())
        return torch.Tensor(output).to(torch.int64)

    @staticmethod
    def tokenize_tar(data, func):
        # prediction mode
        if func is None:
            return
        
        output = []
        for i in range(len(data)):
            output.append(func(data[i]).tolist())
        return torch.Tensor(output).to(torch.int64)
    



# Embedding
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model=model_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model=model_dim, max_len=max_length):
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class MyEmbedding(nn.Module):
    """
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos (BERT)
    """

    def __init__(self, vocab_size, d_model, dropout=dropout):
        super().__init__()
        self.token = Embedding(vocab_size=vocab_size, d_model=d_model) # 27x512
        self.position = PositionalEmbedding(d_model=d_model) # 27x512
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)



# Attention
class SelfAttention(nn.Module):
    def __init__(self, d_model, dropout=dropout):
        super().__init__()
        self.d_model = d_model

        self.query = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key = nn.Linear(self.d_model, self.d_model, bias=False)
        self.value = nn.Linear(self.d_model, self.d_model, bias=False)\

        self.softmax = nn.Softmax(dim=-1)
        self.fc_out = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None):
        x = x.to(dtype=torch.float32)
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.d_model ** 0.5)
        if mask is not None: 
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = self.dropout(self.softmax(scores))

        weighted = torch.bmm(attention, values)
        return self.fc_out(weighted)





# Residual Layer
class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
      return x + self.dropout(y) 



# Feed Forward Layer
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    



class TransformerBlock(nn.Module):
    def __init__(self, d_model, feed_forward_hidden, dropout):
        """
        hidden: hidden size of transformer
        feed_forward_hidden: feed_forward_hidden, usually 4*hidden_ssize
        dropout: dropout rate
        """
        super().__init__()
        self.attention = SelfAttention(d_model=d_model)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_residual = ResidualConnection(dropout=dropout)
        self.output_residual = ResidualConnection(dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        out = self.input_residual(x, self.attention.forward(x, mask))
        out = self.output_residual(out, self.feed_forward(out))
        return self.dropout(x)
    



class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=model_dim, nlayers=n_layers, dropout=dropout):
        super().__init__()
        self.d_model = d_model
        self.nlayers = nlayers
        self.ff_hidden = d_model * 4  # from paper

        # construct encoder model
        self.embedding = MyEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.d_model, self.ff_hidden, dropout) for _ in range(nlayers)])

    def forward(self, x):
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x)
        return x


class Transformer(nn.Module):
    """
    You should implement the Transformer model from scratch here. You can
    use elementary PyTorch layers such as: nn.Linear, nn.Embedding, nn.ReLU, nn.Softmax, etc.
    DO NOT use pre-implemented layers for the architecture, positional encoding or self-attention,
    such as nn.TransformerEncoderLayer, nn.TransformerEncoder, nn.MultiheadAttention, etc.
    """
    def __init__(self, vocab_size, d_model=512, nlayers=2):
        super().__init__()
        self.d_model = d_model
        self.nlayers = nlayers

        # construct encoder-only transformer model
        self.encoder = Encoder(vocab_size=vocab_size)
        self.fc_hidden = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, 3)
        

    def forward(self, x):
        out = self.encoder(x)
        out = self.fc_hidden(out)
        out = self.fc_out(out) # output logits for cross-entropy
        return out




# training
def train_epoch(model, data_loader, optimizer, loss_fn, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    running_accuracy = 0.

    for i, data in enumerate(data_loader):
        # 20 character segment, true frequencies
        source, target = data[0], data[1]

        # zero gradient, produce outputs
        optimizer.zero_grad()
        outputs = model(source)

        # accuracy
        predictions = outputs.max(2).indices
        accuracy = score(predictions.numpy(), target.numpy())

        # loss, gradient, learning rate
        loss = loss_fn(outputs.permute(0, 2, 1), target)
        loss.backward()
        optimizer.step()

        # log progress per batch
        running_loss += loss.item()
        running_accuracy += accuracy
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            last_acc = running_accuracy / 100 # accuracy per batch
            print('  batch {} loss: {} accuracy: {}'.format(i + 1, last_loss, last_acc))
            tb_x = epoch_index * len(data_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
            running_accuracy = 0.

    return last_loss

    


def train_classifier(train_inputs, vocabulary, transform_fn=count_letters, EPOCHS=100, batch_size=2, learning_rate=0.001, weight_decay=0.1):
    # Implement the training loop for the Transformer model.
    model = Transformer(vocab_size=len(vocabulary))
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, eps=1e-9
    )
    # scheduler1 = ExponentialLR(optimizer, gamma=0.9)
    scheduler2 = MultiStepLR(optimizer, milestones=[EPOCHS*.30,EPOCHS*.80], gamma=0.1)
    train_split = len(train_inputs)-1000
    train_dataloader = torch.utils.data.DataLoader(MyDataset(train_inputs[:train_split], vocabulary, transform_fn), batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(MyDataset(train_inputs[train_split+1:], vocabulary, transform_fn), batch_size=batch_size)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/trainer_{}'.format(timestamp))

    epoch_number = 0
    best_vloss = 1_000_000.

    # model training
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train()
        avg_loss = train_epoch(model, train_dataloader, optimizer, loss_fn, epoch_number, writer)
        running_vloss = 0.0

        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(val_dataloader):
                vinputs, vlabels = vdata[0], vdata[1]
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs.permute(0, 2, 1), vlabels)
                running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'best_model_{}_{}'.format(timestamp, epoch_number)
                best_model = (model.state_dict(), model_path)


            epoch_number += 1
            # scheduler1.step()
            scheduler2.step()

    # save and return best model
    model = Transformer(vocab_size=len(vocabulary))
    torch.save(best_model[0], best_model[1])
    model.load_state_dict(torch.load(best_model[1]))
    return model




def test_classifier(model, test_inputs, vocabulary, transform_fn=count_letters, batch_size=1):
    # model = Transformer(vocab_size=len(vocabulary))
    loss_fn = torch.nn.CrossEntropyLoss() 
    test_dataloader = torch.utils.data.DataLoader(MyDataset(test_inputs, vocabulary, transform_fn), batch_size=batch_size)
    model.eval()

    running_loss = 0
    predictions = []
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            source, targets = data[0], data[1]
            outputs = model(source)

            # compute loss
            loss = loss_fn(outputs.permute(0, 2, 1), targets)
            running_loss += loss

            # predictions
            pred = F.softmax(outputs, dim=-1)
            pred = pred.max(dim=2).indices.tolist()
            predictions += pred

    predictions = np.stack([np.array(i) for i in predictions])
    return predictions


def predict_freq(PATH, input, vocab=vocabs):
    # load trained model and data
    model = Transformer(vocab_size=len(vocab))
    model.load_state_dict(torch.load(PATH))
    data = MyDataset([input], vocab, func=None)

    # prepare prediction
    output = model(data.src)
    output = F.softmax(output, dim=-1)
    pred = output.max(dim=2).indices.tolist()

    # change from list -> str format
    out = ""
    for i in pred[0]:
        out+=str(i)
    
    return out

    

    # raise NotImplementedError(
    #     "You should implement `train_classifier` in transformer.py"
    # )
