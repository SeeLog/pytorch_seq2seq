import tqdm
import torch

__PAD__ = '<PAD>'
__PAD_ID__ = 0
__SOS__ = '<SOS>'
__SOS_ID__ = 1
__EOS__ = '<EOS>'
__EOS_ID__ = 2
__UNK__ = '<UNK>'
__UNK_ID__ = 3

MAX_LENGTH = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

__USE_TQDM__ = True

def my_tqdm(itr):
    if __USE_TQDM__:
        return tqdm.tqdm(itr)
    else:
        return itr


