import torch
import torch.nn as nn
import torch.optim as optim
from RNN_dataloader import *

from torchtext.vocab import GloVe, vocab
myvec = GloVe(name = '6B', dim = '100')




class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_size = 100, p=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding.from_pretrained(text.vocab.vectors)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=p)
        
    def forward(self, x, hidden):
        embedding = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedding, hidden)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, embedding_size = 100, p=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding.from_pretrained(text.vocab.vectors)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size) 

    def forward(self, x, hidden):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedding, hidden)
        predictions = self.fc(output)
        predictions = predictions.squeeze(0)
        return predictions, output, hidden

class seq2seq(nn.Module):
  def __init__(self,encoder, decoder):
    super(seq2seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, input, target, teacher_force_ratio = 0.5):
    # print("seq2seq fwd")
    batch_size = input.shape[1]
    target_len = target.shape[0]
    target_vocab_size = len(text.vocab)
    
    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
    # print("Before encoder")
    hidden = self.encoder(input)
    x = target[0]

    for t in range(1,target_len):
      # print(text.vocab.itos[target[t]])
      # print("x shape :", x.shape)
      output, hidden = self.decoder(x,hidden)
      # print("Output Shape", output.shape)
      # print("After decoder ",t,target[t],text.vocab.itos[target[t]])
      outputs[t] = output
      best_guess = output.argmax(2)[0]
      # print("IF Teacher force enabled ",target[t])
      # print("IF Teacher force disabled ",best_guess)
      x = target[t] if random.random() < teacher_force_ratio else best_guess
    
    return outputs