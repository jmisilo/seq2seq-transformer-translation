import itertools
import string
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence      

class Vocabulary:
    def __init__(self, data):
        
        self.vocab = {
            '<unk>': 0,
            '<pad>': 1,
            '<sos>': 2,
            '<eos>': 3
        }
        
        self.build_vocab(data, min_freq=2)
        
    def __getitem__(self, index):
        assert type(index) in [str, int], 'Index type must be string or int'
        
        if isinstance(index, str):
            try:
                return self.vocab[index]
            
            except KeyError:
                return self.vocab['<unk>']
        
        elif isinstance(index, int):
            try:
                return list(self.vocab.keys())[list(self.vocab.values()).index(index)]
            except (KeyError,ValueError):
                return self[0]
    
    def __len__(self):
        return len(self.vocab)
    
    def append_word(self, word):
        if not word in self.vocab and word.isalpha():
            self.vocab[word] = len(self)
    
    def build_vocab(self, data, min_freq=2):
        freq = {}
        for word in data:
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1
            
        freq = pd.Series(freq)
        freq = freq[freq >= min_freq]
        data = list(freq.index)
        
        bag_of_words = sorted(list(set(data)))

        for word in bag_of_words:
            self.append_word(word)
  

class PolEngDS(Dataset):
    def __init__(self, pl_path, en_path, limit=None):
        
        assert (limit >= 0 and type(limit) == int) or limit == None, 'Limit has to be integer, >= 0.' 
        
        self.data = {
            'polish': self._load_data(pl_path),
            'english': self._load_data(en_path)
        }
        
        if limit:
            self.data['polish'] = self.data['polish'][:limit]
            self.data['english'] = self.data['english'][:limit]
            
        self.preprocessing()
        
        self.vocab_pl = Vocabulary(self._flat_list(self.data['polish']))
        self.vocab_en = Vocabulary(self._flat_list(self.data['english']))
        
    def __getitem__(self, index):
        pl, en = [text.split() for text in self.data.iloc[index].values]
        
        pl = torch.IntTensor([self.vocab_pl['<sos>'], *[self.vocab_pl[word] for word in pl], self.vocab_pl['<eos>']])
        en = torch.IntTensor([self.vocab_en['<sos>'], *[self.vocab_en[word] for word in en], self.vocab_en['<eos>']])
        return pl, en 
    
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def _load_data(path):
        with open(path, 'r', encoding='UTF-8') as f:
            data = f.read()
        data = data.split('\n')[:-1]
        
        return data
    
    def preprocessing(self):
        preprocessed_data = {
            'polish': [],
            'english': []
        }
        
        for pl, en in zip(*self.data.values()):
            preprocessed_data['polish'].append(self._text_prep(pl))
            preprocessed_data['english'].append(self._text_prep(en))
        
        self.data = pd.DataFrame(preprocessed_data)
   
    @staticmethod
    def _text_prep(text):
        #remove punctuations
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.strip().lower()
        text.split('/n')
        
        return text
    
    @staticmethod
    def _flat_list(data):
        data = [text.split() for text in data]
        return list(itertools.chain.from_iterable(data))


def pad_seq(batch, padding_pl=1, padding_en=1):
    pl, en = [], []

    for i, (pl_text, en_text) in enumerate(batch):
        pl.append(pl_text)
        en.append(en_text)

    pl = pad_sequence(pl, batch_first=True, padding_value=padding_pl)
    en = pad_sequence(en, batch_first=True, padding_value=padding_en)

    return pl, en


def get_loader(data, batch_size=32):
    return DataLoader(
    data, 
    batch_size=batch_size, 
    collate_fn=pad_seq,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)