import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Word2VecLoader:
    def __init__(self):
        self.embeddings = None
        self.norm_embeddings = None
        self.unit2id = {}
        self.id2unit = {}


    def load_vectors_from_file(self, vec_file, keep_original=False):
        with open(vec_file, encoding='utf-8', errors='ignore') as f:
            stats = f.readline().strip().split()
            num_emb, emb_dim = int(stats[0]), int(stats[1])
            weights = torch.zeros((num_emb, emb_dim), dtype=torch.float32)
            print(f'Loading {num_emb} embeddings with the dimension of {emb_dim}')
            idx = 0
            for line in tqdm(f, total=num_emb):
                line = line.strip().split()
                word = ' '.join(line[:-emb_dim])
                vec = torch.FloatTensor([float(x) for x in line[-emb_dim:]])
                self.unit2id[word] = idx
                self.id2unit[idx] = word
                weights[idx, :] = vec
                idx += 1
        
        if keep_original:
            self.embeddings = nn.Embedding.from_pretrained(weights, freeze=True)
        self.norm_embeddings = nn.Embedding.from_pretrained(F.normalize(weights), freeze=True)


    def get_vector(self, word):
        if self.embeddings:
            idx = torch.LongTensor([self.unit2id[word]])
            return self.embeddings(idx)
        else:
            raise ValueError('Load the vector file first!')


    def get_norm_vector(self, word):
        if self.norm_embeddings:
            idx = torch.LongTensor([self.unit2id[word]])
            return self.norm_embeddings(idx).squeeze()
        else:
            raise ValueError('Load the vector file first!')


    def get_similarity(self, word1, word2):
        """Computes the cosine similarity between two word vectors."""

        return torch.matmul(self.get_norm_vector(word1), self.get_norm_vector(word2))

    
    def get_most_similar(self, positive=None, negative=None, k=10):
        """Returns topk most similar words to the mean of positives and negatives."""
        
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        positive = [(word, 1.0) for word in positive]
        negative = [(word, -1.0) for word in negative]

        mean = []
        for word, weight in positive + negative:
            mean.append(weight * self.get_norm_vector(word))

        if not mean:
            raise ValueError("Cannot compute similarity with no input")

        mean = torch.stack(mean)
        mean = torch.mean(mean, dim=0, dtype=torch.float32)

        dists = torch.matmul(self.norm_embeddings.weight, mean)
        topk_dists, topk_idx = torch.topk(dists, k+1, largest=True, sorted=True)
        topk_words = [self.id2unit[i.item()] for i in topk_idx]
        return (topk_words[1:], topk_dists[1:])