#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.CHAR_EMBED_SIZE = 50
        self.dropout_rate = 0.3
        self.embed_size = embed_size
        pad_token_idx = vocab['<pad>']
        self.conv = CNN(self.CHAR_EMBED_SIZE,embed_size)
        self.highway = Highway(embed_size,embed_size)
        self.charEmbedding = nn.Embedding(num_embeddings=len(vocab.char2id), embedding_dim=self.CHAR_EMBED_SIZE, padding_idx=pad_token_idx)
        self.dropout = nn.Dropout(self.dropout_rate)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        """
        Input of charEmbedding is of shape(sentence_length, batch_size, max_word_length)
        Output of charEmbedding is of shape(sentence_length, batch_size, max_word_length, CHAR_EMBED_SIZE)
        X_reshaped.shape = (batch_size*sentence_length, max_word_length, embed_size)

        Input to CNN is of shape(batch_size, CHAR_EMBED_SIZE, max_word_length)
        """
        sent_len, batch_size, max_word_len = input.shape
        X_emb = self.charEmbedding(input).permute((0,1,3,2))
        X_emb = X_emb.contiguous().view(batch_size*sent_len,self.CHAR_EMBED_SIZE,max_word_len)
        X_conv_out = self.conv(X_emb)
        X_highway = self.highway(X_conv_out)
        X_out = self.dropout(X_highway).view(sent_len,batch_size,-1)
        return X_out

        ### END YOUR CODE
