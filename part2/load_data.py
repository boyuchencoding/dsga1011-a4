import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
TOKENIZER_NAME = 'google-t5/t5-small'
INPUT_PREFIX = 'translate English to SQL: '
# Decoder start token: an extra-id sentinel (never produced as a normal word piece)
BOS_TOKEN = '<extra_id_0>'


def sql_encode(sql):
    '''The T5 vocabulary has no "<" character (it becomes <unk>), so map comparison
    operators to words before tokenizing. Reversed exactly by sql_decode.'''
    sql = sql.replace('<=', ' lte ').replace('<', ' lt ')
    return re.sub(r'\s+', ' ', sql).strip()


def sql_decode(text):
    text = re.sub(r'\blte\b', '<=', text)
    text = re.sub(r'\blt\b', '<', text)
    return re.sub(r'\s+', ' ', text).strip()


def get_tokenizer():
    return T5TokenizerFast.from_pretrained(TOKENIZER_NAME)


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Data processing for the T5 model.
            * Encoder input: INPUT_PREFIX + natural language instruction (tokenized with the
              google-t5/t5-small tokenizer, which appends </s>).
            * Decoder input: <extra_id_0> followed by the SQL tokens (shifted right);
              decoder target: the SQL tokens followed by </s>.
            * On the test split there is no SQL, so only the encoder input is kept.
        '''
        self.split = split
        self.tokenizer = get_tokenizer()
        self.bos_id = self.tokenizer.convert_tokens_to_ids(BOS_TOKEN)
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_lines = load_lines(os.path.join(data_folder, f'{split}.nl'))
        enc_ids = [tokenizer(INPUT_PREFIX + nl)['input_ids'] for nl in nl_lines]

        if split == 'test':
            return [{'enc': torch.tensor(e)} for e in enc_ids]

        sql_lines = load_lines(os.path.join(data_folder, f'{split}.sql'))
        assert len(nl_lines) == len(sql_lines)
        data = []
        for e, sql in zip(enc_ids, sql_lines):
            tgt = tokenizer(sql_encode(sql))['input_ids']  # ends with </s>
            data.append({
                'enc': torch.tensor(e),
                'dec_in': torch.tensor([self.bos_id] + tgt[:-1]),
                'dec_tgt': torch.tensor(tgt),
            })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def normal_collate_fn(batch):
    '''
    Dynamic padding for training and for evaluation on the dev set.

    Returns
        * encoder_ids: BxT, encoder_mask: BxT, decoder_inputs: BxT', decoder_targets: BxT',
        * initial_decoder_inputs: Bx1 (the <extra_id_0> start token)
    '''
    encoder_ids = pad_sequence([b['enc'] for b in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_inputs = pad_sequence([b['dec_in'] for b in batch], batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence([b['dec_tgt'] for b in batch], batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = decoder_inputs[:, :1]
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    '''
    Dynamic padding for inference on the test set.
    Returns encoder_ids, encoder_mask, initial_decoder_inputs (Bx1 start tokens).
    '''
    encoder_ids = pad_sequence([b['enc'] for b in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    bos_id = get_tokenizer().convert_tokens_to_ids(BOS_TOKEN)
    initial_decoder_inputs = torch.full((len(batch), 1), bos_id, dtype=torch.long)
    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")

    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x
