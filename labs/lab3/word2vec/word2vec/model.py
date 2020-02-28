import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dim):
        super(SkipGramModel, self).__init__()
        self.emb_dim = emb_dim

        # Create two embedding layers with nn.Embedding
        # Hint: use emb_size and emb_dim as parameters
        self.u_emb = ...
        self.v_emb = ...

        # Initialize the weights for better training
        initrange = 1.0 / emb_dim
        init.uniform_(self.u_emb.weight.data, -initrange, initrange)
        init.constant_(self.v_emb.weight.data, 0)
    
    def forward(self, pos_u, pos_v, neg_v):
        emb_pos_u = self.u_emb(pos_u)
        emb_pos_v = self.v_emb(pos_v)
        emb_neg_v = self.v_emb(neg_v)

        # Complete the objective function for the skip-gram model

        # Multiply emb_pos_u and emb_pos_v and take sum for it
        # Hint: use torch.mul() and torch.sum()
        score = ...
        # Clamp the score for better training
        score = torch.clamp(score, max=10, min=-10)

        # Take the negative logsigmoid of the score
        # Hint: use F.logsigmoid()
        score = ...

        # Multiply emb_neg_v and emb_pos_u 
        # Hint: use torch.bmm this time as well as squeese() and unsqueese() to match the dimensions
        neg_score = ...

        # Sum the neg_score
        neg_score = ...

        # Clamp the neg_score for better training
        neg_score = torch.clamp(neg_score, max=10, min=-10)

        # Take the negative logsigmoid of the -neg_score
        neg_score = ...

        return torch.mean(score + neg_score)

    
    def save_embedding(self, id2word, file_name):
        embedding = self.u_emb.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dim))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))