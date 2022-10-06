import random
import torch
import torchtext
import spacy
import numpy as np
from captum.concept._utils.data_iterator import (
    dataset_to_dataloader,
    CustomIterableDataset,
)
from sklearn.metrics import accuracy_score
from explanations.concept import CAR
import torch.nn as nn
import torch.nn.functional as F

"""
Code adapted from https://captum.ai/tutorials/TCAV_NLP 
"""


class CNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        n_filters,
        filter_sizes,
        output_dim,
        dropout,
        pad_idx,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=n_filters,
                    kernel_size=(fs, embedding_dim),
                )
                for fs in filter_sizes
            ]
        )

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]

        # text = text.permute(1, 0)

        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

    def representation(self, text):
        # text = [sent len, batch size]

        # text = text.permute(1, 0)

        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))
        return cat


def get_tensor_from_filename(filename):
    ds = torchtext.data.TabularDataset(
        path=filename,
        fields=[("text", torchtext.data.Field()), ("label", torchtext.data.Field())],
        format="csv",
    )
    const_len = 7
    for concept in ds:
        concept.text = concept.text[:const_len]
        concept.text += ["pad"] * max(0, const_len - len(concept.text))
        text_indices = torch.tensor(
            [TEXT.vocab.stoi[t] for t in concept.text], device=device
        )
        yield text_indices


def assemble_concept(concepts_path="data/tcav/text-sensitivity"):
    dataset = CustomIterableDataset(get_tensor_from_filename, concepts_path)
    concept_iter = dataset_to_dataloader(dataset, batch_size=1)
    return torch.cat([concept for concept in concept_iter])


if __name__ == "__main__":
    nlp = spacy.load("en")
    random.seed(123)
    np.random.seed(123)
    device = torch.device("cpu")

    # Load tokenizer
    TEXT = torchtext.data.Field(lower=True, tokenize="spacy")
    Label = torchtext.data.LabelField(dtype=torch.float)
    train, _ = torchtext.datasets.IMDB.splits(
        text_field=TEXT, label_field=Label, train="train", test="test", root="data/"
    )
    MAX_VOCAB_SIZE = 25_000
    loaded_vectors = torchtext.vocab.Vectors("data/imdb/glove.6B.50d.txt")
    TEXT.build_vocab(train, vectors=loaded_vectors, max_size=len(loaded_vectors.stoi))
    TEXT.vocab.set_vectors(
        stoi=loaded_vectors.stoi, vectors=loaded_vectors.vectors, dim=loaded_vectors.dim
    )
    print("Vocabulary Size: ", len(TEXT.vocab))

    # Load model
    model = torch.load("data/imdb/imdb-model-cnn-large.pt")
    model.eval()

    # Get concept sets
    negative_set = assemble_concept("data/imdb/neutral.csv")
    positive_set = assemble_concept("data/imdb/positive-adjectives.csv")
    positive_reps = model.representation(positive_set).detach()
    negative_reps = model.representation(negative_set)[: len(positive_reps)].detach()
    positive_reps_train, positive_reps_test = torch.split(
        positive_reps, split_size_or_sections=[90, 30]
    )
    negative_reps_train, negative_reps_test = torch.split(
        negative_reps, split_size_or_sections=[90, 30]
    )
    H_train = torch.cat([positive_reps_train, negative_reps_train]).numpy()
    H_test = torch.cat([positive_reps_test, negative_reps_test]).numpy()
    Y_train = np.concatenate([np.ones(len(H_train) // 2), np.zeros(len(H_train) // 2)])
    Y_test = np.concatenate([np.ones(len(H_test) // 2), np.zeros(len(H_test) // 2)])

    # Fit and test CAR classifier
    car = CAR(device)
    car.fit(H_train, Y_train)
    print(
        f"Accuracy of CAR classifier for the positive adjective concept: {accuracy_score(Y_test, car.predict(H_test)):.2g}"
    )
