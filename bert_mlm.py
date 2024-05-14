import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import BertConfig
from transformers import BertModel, BertForSequenceClassification
import numpy as np
from transformers import (
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
import torch
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
import collections
import logging
import os
import math
import unicodedata
import re
from copy import deepcopy
from transformers import BertForMaskedLM, Trainer, TrainingArguments
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
# from apex import amp
import logging
from bert_utils import DesignedTokenizer

import argparse

def get_args(): 
    parser = argparse.ArgumentParser(description='CE-Bert')
    parser.add_argument('--k', type=str, help='kmer')
    parser.add_argument('--embedding_size', type=int, default=256, help='embedding size')
    
    return parser.parse_args()

args = get_args()
vocab_filename = "vocab-{}.txt".format(args.k)
vocab_file = "./bert-config/vocab-{}.txt".format(args.k)


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

class DesignedTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a BertTokenizer.
    :class:`~transformers.BertTokenizer` runs end-to-end tokenization: punctuation splitting + wordpiece

    Args:
        vocab_file: Path to a one-wordpiece-per-line vocabulary file
        do_lower_case: Whether to lower case the input. Only has an effect when do_basic_tokenize=True
        do_basic_tokenize: Whether to do basic tokenization before wordpiece.
        max_len: An artificial maximum length to truncate tokenized sequences to; Effective maximum length is always the
            minimum of this value (if specified) and the underlying BERT model's sequence length.
        never_split: List of tokens which will never be split during tokenization. Only has an effect when
            do_basic_tokenize=True
    """

    vocab_files_names = {"vocab_file": vocab_filename}


    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        never_split=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        tokenize_chinese_chars=True,
        **kwargs
    ):
        """Constructs a BertTokenizer.

        Args:
            **vocab_file**: Path to a one-wordpiece-per-line vocabulary file
            **do_lower_case**: (`optional`) boolean (default True)
                Whether to lower case the input
                Only has an effect when do_basic_tokenize=True
            **do_basic_tokenize**: (`optional`) boolean (default True)
                Whether to do basic tokenization before wordpiece.
            **never_split**: (`optional`) list of string
                List of tokens which will never be split during tokenization.
                Only has an effect when do_basic_tokenize=True
            **tokenize_chinese_chars**: (`optional`) boolean (default True)
                Whether to tokenize Chinese characters.
                This should likely be deactivated for Japanese:
                see: https://github.com/huggingface/pytorch-pretrained-BERT/issues/328
        """
        self.vocab = self.load_vocab(vocab_file)    
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.k = 3
        self.reserved_tokens = ['P']
        
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        
        self.max_len = 512
        # self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        # self.max_len_sentences_pair = self.max_len - 3
        
        # self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        # self.max_len_sentences_pair = self.max_len - 3  # take into account special tokens

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file)
            )
        
        self.kmer = 3
        self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case, never_split=never_split, tokenize_chinese_chars=tokenize_chinese_chars
            )
        
    def tokenize(self, text):
        tokens = re.split(r"[<>]", text)
        new_tokens = []
        n = self.k
        for t in tokens:
            if not t:
                continue

            if f"<{t}>" in self.reserved_tokens:
                new_tokens.append(f"<{t}>")
            else:
                seq = t
                # split kmers
                chunks = [seq[i:i + n] for i in range(0, len(seq), n)]
                new_tokens += chunks
        return new_tokens
    
    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = {}
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                split_tokens.append(token)
        # print(split_tokens)
        return split_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    # def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    #     """
    #     Build model inputs from a sequence or a pair of sequence for sequence classification tasks
    #     by concatenating and adding special tokens.
    #     A BERT sequence has the following format:
    #         single sequence: [CLS] X [SEP]
    #         pair of sequences: [CLS] A [SEP] B [SEP]
    #     """
    #     cls = [self.cls_token_id]
    #     sep = [self.sep_token_id]

    #     if token_ids_1 is None:
    #         if len(token_ids_0) < 510:
    #             return cls + token_ids_0 + sep
    #         else:
    #             output = []
    #             num_pieces = int(len(token_ids_0)//510) + 1
    #             for i in range(num_pieces):
    #                 output.extend(cls + token_ids_0[510*i:min(len(token_ids_0), 510*(i+1))] + sep)
    #             return output

    #     return cls + token_ids_0 + sep + token_ids_1 + sep
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequences for sequence classification tasks
        by concatenating and adding special tokens. A BERT sequence has the following format:
            single sequence: [CLS] X [SEP]
            pair of sequences: [CLS] A [SEP] B [SEP]
        This method adjusts the input sequences to ensure they fit within the maximum length
        specified for the model, including the special tokens.

        Args:
            token_ids_0: List of token ids for the first sequence.
            token_ids_1: (Optional) List of token ids for the second sequence.

        Returns:
            A list of token ids representing the concatenated sequences with special tokens added.
        """
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        sequence_length = self.max_len - 2  # Accounting for [CLS] and [SEP]

        if token_ids_1 is None:
            # Single sequence case: Truncate if necessary and add special tokens.
            return cls + token_ids_0[:sequence_length] + sep
        else:
            # Pair of sequences: Split the max length more or less evenly between the two.
            # We reserve two spaces for [SEP] tokens, one after each sequence.
            half_max_len = sequence_length // 2  # Divide the available space in half for each sequence
            # Truncate both sequences to make sure combined lengths fit within the limit.
            # Adjust lengths to accommodate both sequences (try to use as much as possible from both)
            token_ids_0 = token_ids_0[:max(half_max_len, sequence_length - len(token_ids_1))]
            token_ids_1 = token_ids_1[:max(half_max_len, sequence_length - len(token_ids_0))]
            # Concatenate with special tokens
            return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        
        if len(token_ids_0) < 510:
            return [1] + ([0] * len(token_ids_0)) + [1]
        else:
            output = []
            num_pieces = int(len(token_ids_0)//510) + 1
            for i in range(num_pieces):
                output.extend([1] + ([0] * (min(len(token_ids_0), 510*(i+1))-510*i)) + [1])
            return output

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence

        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            if len(token_ids_0) < 510:
                return len(cls + token_ids_0 + sep) * [0]
            else:
                num_pieces = int(len(token_ids_0)//510) + 1
                return (len(cls + token_ids_0 + sep) + 2*(num_pieces-1)) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, vocab_path, filename_prefix=None):
        """Save the tokenizer vocabulary to a directory or file."""
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, vocab_filename)
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)
    
    def get_vocab(self):
        return self.vocab



class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=False, never_split=None, tokenize_chinese_chars=True):
        """ Constructs a BasicTokenizer.

        Args:
            **do_lower_case**: Whether to lower case the input.
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level (see :func:`PreTrainedTokenizer.tokenize`)
                List of token not to split.
            **tokenize_chinese_chars**: (`optional`) boolean (default True)
                Whether to tokenize Chinese characters.
                This should likely be deactivated for Japanese:
                see: https://github.com/huggingface/pytorch-pretrained-BERT/issues/328
        """
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = never_split
        self.tokenize_chinese_chars = tokenize_chinese_chars

    def tokenize(self, text, never_split=None):
        """ Basic Tokenization of a piece of text.
            Split on "white spaces" only, for sub-word tokenization, see WordPieceTokenizer.

        Args:
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level (see :func:`PreTrainedTokenizer.tokenize`)
                List of token not to split.
        """
        never_split = self.never_split + (never_split if never_split is not None else [])
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]



    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

MASK_LIST = {
    "3": [-1, 1],
    "4": [-1, 1, 2],
    "5": [-2, -1, 1, 2],
    "6": [-2, -1, 1, 2, 3]
}

def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    
    mask_list = MASK_LIST["{}".format(tokenizer.kmer)]

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args['mlm_probability'])
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    # change masked indices
    masks = deepcopy(masked_indices)
    for i, masked_index in enumerate(masks):
        end = torch.where(probability_matrix[i]!=0)[0].tolist()[-1]
        mask_centers = set(torch.where(masked_index==1)[0].tolist())
        new_centers = deepcopy(mask_centers)
        for center in mask_centers:
            for mask_number in mask_list:
                current_index = center + mask_number
                if current_index <= end and current_index >= 1:
                    new_centers.add(current_index)
        new_centers = list(new_centers)
        masked_indices[i][new_centers] = True
    

    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


class RNADataset(Dataset):
    def __init__(self, sequences, tokenizer, max_len, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.texts = sequences
        self.max_len = max_len
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize text and pad & truncate to max length
        encodings = self.tokenizer(text, max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
        
        inputs = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        # Prepare arguments structure, adapt this if your args are structured differently
        args = {
            'mlm_probability': self.mlm_probability
        }

        # Apply mask_tokens function to generate masked inputs and labels
        inputs, labels = mask_tokens(inputs, self.tokenizer, args)

        return inputs.squeeze(0), attention_mask.squeeze(0), labels.squeeze(0)

if __name__ == "__main__":

    # Assuming df is your pandas DataFrame containing the RNA sequences
    match_table = pd.read_csv("./datasets/ceRNA/circRNA_lncrna_miRNA_interaction.csv", index_col=0)
    tokenizer = DesignedTokenizer(vocab_file=vocab_file)
    seqes = list(match_table['circrna_or_lncrna_seq'].unique()) + list(match_table['mirna_seq'].unique())
    train_dataset = RNADataset(seqes, tokenizer, max_len=512)
    config = BertConfig(
    vocab_size=len(tokenizer),
    num_hidden_layers=4,        # Fewer layers
    hidden_size=args.embedding_size,            # Smaller hidden layer size
    num_attention_heads=4
    )
    model = BertForMaskedLM(
    config=config,
    ).to('cuda')
    BATCH_SIZE = 32
    MAX_LENGTH = 512
    EPOCHS = 100
    LR = 5e-5
    train_sampler = RandomSampler(train_dataset)
    
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=32
    )
    
    t_total = len(train_dataloader) // EPOCHS
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR, eps=1e-8, betas=(0.9, 0.999))
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=0, num_training_steps=t_total
    # )
    
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    
    for epoch in tqdm(range(EPOCHS), desc="Epoch", total=EPOCHS):
        model.train()
        for step, batch in tqdm(enumerate(train_dataloader), desc="Iteration", total=len(train_dataloader)):
            # Updated to accommodate attention masks
            input_ids, attention_masks, labels = batch
            # print(input_ids.shape, attention_masks.shape, labels.shape)
            inputs = {
                'input_ids': input_ids.to('cuda'),
                'attention_mask': attention_masks.to('cuda'),
                'labels': labels.to('cuda')
            }
            outputs = model(**inputs)
            # print(outputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            # scheduler.step()
            model.zero_grad()
            # print(f"Step: {global_step}, Loss: {loss.item()}")
            global_step += 1
            if global_step % 100 == 0:
                print(f"Step: {global_step}, Loss: {loss.item()}")
                
    # Save the model
    model.save_pretrained('./trained_model_bert_{}_{}'.format(args.k, args.embedding_size))
    tokenizer.save_pretrained('./trained_model_bert_{}_{}'.format(args.k, args.embedding_size))