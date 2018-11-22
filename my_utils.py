import tqdm

__SOS__ = 'SOS'
__EOS__ = 'EOS'

MAX_LENGTH = 30

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

__USE_TQDM__ = False

def my_tqdm(itr):
    if __USE_TQDM__:
        return tqdm.tqdm(itr)
    else:
        return itr


