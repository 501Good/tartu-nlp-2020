import torch
from torch.utils.data import Dataset
import numpy as np

from collections import Counter

NEGATIVE_TABLE_SIZE = 1e8

class Word2VecDataReader:
    def __init__(self, input_filename, min_count):
        self.id2unit = {}
        self.unit2id = {}
        self.token_count = 0
        self.word_frequency = Counter()
        self.lines = []

        self._negatives = []
        self._discrads = []

        self.input_filename = input_filename
        self._init_vocab(min_count)
        self._init_negatives()
        self._init_discards()
    

    def _init_vocab(self, min_count):
        """Reads all the words from the data."""

        word_frequency = Counter()
        for line in open(self.input_filename, encoding='utf-8', errors='ignore'):
            words = line.split()
            if len(words) > 1:
                self.lines.append(words)
                for word in words:
                    if word:
                        self.token_count += 1
                        word_frequency[word] += 1

        word_frequency = {k: v for k, v in word_frequency.items() if v > min_count}
        for idx, (word, count) in enumerate(word_frequency.items()):
            self.unit2id[word] = idx
            self.id2unit[idx] = word
            self.word_frequency[idx] = count
        
        print("Total words:", len(self.unit2id))

    
    def _init_negatives(self):
        """Initializes the table of negatives."""

        # Frequency for each word to the power of 0.75
        # Hint: use our self.word_frequency dict
        word_frequency = ...

        # Sum of all word frequencies
        total_word_frequency = ...

        # Probabilites for the negative sampling table
        word_probabilities = word_frequency / total_word_frequency

        # Initialize the negatives table
        counts = np.round(word_probabilities * NEGATIVE_TABLE_SIZE)
        for idx, count in enumerate(counts):
            self._negatives += [idx] * int(count)
        self._negatives = np.array(self._negatives)
        np.random.shuffle(self._negatives)

    
    def _init_discards(self):
        """Initializes the table of discards for subsampling."""

        t = 1e-5
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self._discrads = np.sqrt(t / f) + (t / f)


    def get_negatives(self, target, size):
        """Chooses the negative examples for the target word."""

        negatives = []
        while len(negatives) < size:
            random_idx = np.random.randint(0, NEGATIVE_TABLE_SIZE)
            negative = self._negatives[random_idx]
            if negative not in negatives and negative != target:
                negatives.append(negative)
        return np.array(negatives)


class Word2VecDataset(Dataset):
    def __init__(self, data, window_size, negative_size):
        self.data = data
        self.window_size = window_size
        self.negative_size = negative_size      

    
    def __len__(self):
        return len(self.data.lines)

    
    def __getitem__(self, idx):
        words = self.data.lines[idx]
        word_ids = [self.data.unit2id[word] for word in words if
                    word in self.data.unit2id and 
                    np.random.rand() < self.data._discrads[self.data.unit2id[word]]]
        
        return [(u, v, self.data.get_negatives(v, self.negative_size)) for i, u in enumerate(word_ids)
                for v in word_ids[max(i - self.window_size, 0):i + self.window_size] if u != v]

    
    @staticmethod
    def collate(batches):
        pos_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        pos_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(pos_u), torch.LongTensor(pos_v), torch.LongTensor(neg_v)
