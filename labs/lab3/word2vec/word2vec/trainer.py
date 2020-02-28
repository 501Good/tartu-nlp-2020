import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from word2vec.data import Word2VecDataReader, Word2VecDataset
from word2vec.model import SkipGramModel


class Word2VecTrainer:
    def __init__(self, input_file, output_file, emb_dim=100, batch_size=32, window_size=5,
                negative_size=5, iters=3, initial_lr=0.001, min_count=12):

        self.data = Word2VecDataReader(input_file, min_count)
        dataset = Word2VecDataset(self.data, window_size, negative_size)
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)

        self.output_file_name = output_file
        self.emb_size = len(self.data.unit2id)
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.iters = iters
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dim)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):

        for it in range(self.iters):

            print("\n\n\nIteration: " + str(it + 1))

            # Use SparseAdam optimiser
            optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)

            for sample_batched in tqdm(self.dataloader):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    # Clear the accumulated gradient for the optimiser
                    # Hint: Use .zero_grad() 
                    ...

                    # Perform a forward pass of our model
                    loss = ...

                    # Compute derivative of the loss
                    loss.backward()

                    # Update the optimizer
                    optimizer.step()

            self.skip_gram_model.save_embedding(self.data.id2unit, self.output_file_name)


if __name__ == '__main__':
    w2v = Word2VecTrainer(input_file="input.txt", output_file="out.vec")
    w2v.train()
