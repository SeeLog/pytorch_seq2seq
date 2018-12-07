from seq2seq import *
from my_utils import __PAD__, __SOS__, __EOS__, __UNK__, MAX_LENGTH, __PAD_ID__, __SOS_ID__, __EOS_ID__
import torch
import random
import time
import math


from fileloader import loadPkls, loadEval
from Lang import LimitedLang

from tqdm import tqdm

import nltk.translate.bleu_score as bleu_score

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams['font.family'] = 'Ume Gothic O5'
plt.rcParams['font.size'] = 10

def compute_bleu(trues, preds):
    return np.mean([bleu_score.sentence_bleu(gt, p) for gt, p in zip(trues, preds)])

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] if word in lang.word2index else lang.word2index[__UNK__] for word in sentence.split(' ')]

def indexesFromTokens(lang: LimitedLang, tokens):
    return [lang.word2index[word] if word in lang.word2index else lang.word2index[__UNK__] for word in tokens]

def tensorFromSentence(lang: LimitedLang, sentence):
    length = len(sentence.split(' '))
    indexes = indexesFromSentence(lang, sentence)
    #indexes.append(lang.word2index[__EOS__])

    return indexes + [__EOS_ID__] + [__PAD_ID__] * (MAX_LENGTH - length - 1), length + 1

def tensorFromTokens(lang: LimitedLang, tokens):
    length = len(tokens)
    indexes = indexesFromTokens(lang, tokens)
    #indexes.append(lang.word2index[__EOS__])

    return indexes + [__EOS_ID__] + [__PAD_ID__] * (MAX_LENGTH - length - 1), length + 1

def tensorFromTokenPair(source_lang, target_lang, tokenpair):
    source_tensor = tensorFromTokens(source_lang, tokenpair[0])
    target_tensor = tensorFromTokens(target_lang, tokenpair[1])

    return (source_tensor, target_tensor)

def generate_batch(pairs, source_lang, target_lang, batch_size=200, shuffle=True):
    if shuffle:
        random.shuffle(pairs)

    for i in tqdm(range(len(pairs) // batch_size)):
        batch_pairs = pairs[batch_size * i: batch_size * (i+1)]

        source_batch = []
        target_batch = []
        source_lens = []
        target_lens = []

        for source_seq, target_seq in batch_pairs:
            #print(source_seq)
            last = target_seq
            source_seq, source_length = tensorFromTokens(source_lang, source_seq)
            target_seq, target_length = tensorFromTokens(target_lang, target_seq)
            source_batch.append(source_seq)
            target_batch.append(target_seq)
            source_lens.append(source_length)
            target_lens.append(target_length)

        source_batch = torch.tensor(source_batch, dtype=torch.long, device=device)
        target_batch = torch.tensor(target_batch, dtype=torch.long, device=device)
        source_lens = torch.tensor(source_lens)
        target_lens = torch.tensor(target_lens)

        #print(source_batch.shape)

        # sort
        source_lens, sorted_idxs = source_lens.sort(0, descending=True)

        source_batch = source_batch[sorted_idxs].transpose(0, 1)
        #print(source_batch.shape)
        #source_batch = source_batch[:source_lens.max().item()]

        target_batch = target_batch[sorted_idxs].transpose(0, 1)
        #target_batch = target_batch[:target_lens.max().item()]
        target_lens = target_lens[sorted_idxs]

        #print(source_batch.shape)

        yield source_batch, source_lens, target_batch, target_lens

def batch_train(source_batch, source_lens, target_batch, target_lens, encoder, decoder, optimizer, criterion, teacher_forcing_ratio=0.5):
    loss = 0
    optimizer.zero_grad()

    batch_size = source_batch.shape[1]
    target_length = target_lens.max().item()

    encoder_outputs, encoder_hidden = encoder(source_batch, source_lens)
    #print(batch_size)
    #print(encoder_outputs.shape)

    decoder_input = torch.tensor([[__SOS_ID__] * batch_size], device=device)
    decoder_inputs = torch.cat([decoder_input, target_batch], dim=0)
    decoder_hidden = (encoder_hidden[0].squeeze(0), encoder_hidden[1].squeeze(0))

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, attention = decoder(
                decoder_inputs[di], decoder_hidden, encoder_outputs
            )
            loss += criterion(decoder_output, decoder_inputs[di+1])
    else:
        decoder_input = decoder_inputs[0]
        for di in range(target_length):
            decoder_output, decoder_hidden, attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            loss += criterion(decoder_output, decoder_inputs[di+1])
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(1).detach()

    loss.backward()
    optimizer.step()

    return loss.item() / target_length


def batch_evaluation(source_batch, source_lens, target_batch, target_lens, encoder, decoder, criterion, target_lang: LimitedLang):
    with torch.no_grad():
        batch_size = source_batch.shape[1]
        target_length = target_lens.max().item()
        target_batch = target_batch[:target_length]

        loss = 0

        encoder_outputs, encoder_hidden = encoder(source_batch, source_lens)
        decoder_input = torch.tensor([__SOS_ID__] * batch_size, device=device)
        decoder_hidden = (encoder_hidden[0].squeeze(0), encoder_hidden[1].squeeze(0))
        decoded_outputs = torch.zeros(target_length, batch_size, target_lang.n_words, device=device)
        decoded_words = torch.zeros(batch_size, target_length, device=device)

        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoded_outputs[di] = decoder_output

            loss += criterion(decoder_output, target_batch[di])

            _, topi = decoder_output.topk(1)
            decoded_words[:, di] = topi[:, 0]

        bleu = 0

        for bi in range(batch_size):
            try:
                end_idx = decoded_words[bi, :].tolist().index(__EOS_ID__)
            except:
                end_idx = target_length
            score = compute_bleu(
                [[[target_lang.index2word[i] for i in target_batch[:, bi].tolist() if i > 2]]],
                [[target_lang.index2word[j] for j in decoded_words[bi, :].tolist()[:end_idx]]]
            )

            bleu += score

        return loss.item() / target_length, bleu / float(batch_size)

def inference(encoder, decoder, sentence, source_lang: LimitedLang, target_lang: LimitedLang, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_indxs, input_length = tensorFromTokens(source_lang, sentence)

        input_batch = torch.tensor([input_indxs], dtype=torch.long, device=device)  # (1, s)
        input_length = torch.tensor([input_length])  # (1)

        encoder_outputs, encoder_hidden = encoder(input_batch.transpose(0, 1), input_length)

        decoder_input = torch.tensor([__SOS_ID__], device=device)  # (1)

        decoder_hidden = (encoder_hidden[0].squeeze(0), encoder_hidden[1].squeeze(0))

        decoded_words = []
        attentions = []

        for di in range(max_length):
            decoder_output, decoder_hidden, attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)  # (1,odim), ((1,h),(1,h)), (l,1)
            attentions.append(attention)
            print("ATTN:", attention.shape)
            print(len(attentions))
            _, topi = decoder_output.topk(1)  # (1, 1)
            if topi.item() == __EOS_ID__:
                decoded_words.append(__EOS__)
                break
            else:
                decoded_words.append(target_lang.index2word[topi.item()])

            decoder_input = topi[0]

        attentions = torch.cat(attentions, dim=0)  # (l, n)
        print("ATTS:", attentions.shape)


        return decoded_words, attentions.squeeze(0).cpu().numpy()

def train(source_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimzer, criterion, max_length=MAX_LENGTH, teacher_forcing_ratio=0.5):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimzer.zero_grad()

    source_length = source_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    losssum = 0

    for ei in range(source_length):
        encoder_output, encoder_hidden = encoder(source_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[__SOS_ID__]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            loss = criterion(decoder_output, target_tensor[di])
            losssum += loss.item()
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss = criterion(decoder_output, target_tensor[di])
            losssum += loss.item()
            if decoder_input.item() == __EOS_ID__:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimzer.step()

    return losssum / target_length


def train_iters(encoder, decoder, train_pairs, test_pairs, source_lang, target_lang, epochs=20, batch_size=200, teacher_forcing=0.5, early_stopping=5):
    optimizer = optim.Adam([p for p in encoder.parameters()] + [p for p in decoder.parameters()])

    criterion = nn.NLLLoss(ignore_index=__PAD_ID__)

    validation_bleus = []

    for epoch in range(epochs):
        print("------------------- EPOCH", epoch, "-------------------")
        total_loss = 0
        for source_batch, source_lens, target_batch, target_lens in generate_batch(train_pairs, source_lang, target_lang, batch_size=batch_size):
            loss = batch_train(source_batch, source_lens, target_batch, target_lens, encoder, decoder,
                optimizer, criterion, teacher_forcing)

            total_loss += loss
            train_loss = total_loss / (len(train_pairs) / batch_size)

        total_bleu = 0

        for source_batch, source_lens, target_batch, target_lens in generate_batch(train_pairs, source_lang, target_lang, batch_size=batch_size, shuffle=False):
            loss, bleu = batch_evaluation(source_batch, source_lens, target_batch, target_lens, encoder, decoder, criterion, target_lang)
            total_bleu += bleu

        train_bleu = total_bleu / (len(train_pairs) / batch_size)

        total_loss = 0
        total_bleu = 0

        for source_batch, source_lens, target_batch, target_lens in generate_batch(test_pairs, source_lang, target_lang, batch_size=batch_size, shuffle=False):
            loss, bleu = batch_evaluation(source_batch, source_lens, target_batch, target_lens, encoder, decoder, criterion, target_lang)
            total_loss += loss
            total_bleu += bleu

        validation_loss = total_loss / (len(test_pairs) / batch_size)
        validation_bleu = total_bleu / (len(test_pairs) / batch_size)

        save_loss_graph({
            'loss': train_loss,
            'bleu': train_bleu,
            'val_loss': validation_loss,
            'val_bleu': validation_bleu
            })

        # Save models
        torch.save(encoder.state_dict(), './model/encoder_batch_epoch' + str(epoch) + '.model')
        torch.save(decoder.state_dict(), './model/decoder_batch_epoch' + str(epoch) + '.model')

        validation_bleus.append(validation_bleu)

        evaluate_randomly(test_pairs, encoder, decoder, source_lang, target_lang)

        if max(validation_bleus[-early_stopping:]) < max(validation_bleus):
            print()
            print("!!!!!    EARLY STOPPING    !!!!!")
            print()
            break

    return max(validation_bleus)

loss_graph_dict = {}
def save_loss_graph(dic):
    for key in dic:
        if key not in loss_graph_dict:
            loss_graph_dict[key] = [dic[key]]
        else:
            loss_graph_dict[key].append(dic[key])
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))

    axL.plot(range(1, len(loss_graph_dict["loss"]) + 1), loss_graph_dict["loss"], linewidth=2, label="training")
    axL.plot(range(1, len(loss_graph_dict["val_loss"]) + 1), loss_graph_dict["val_loss"], linewidth=2, label="validation")

    axL.set_title("loss")
    axL.set_xlabel("epoch")
    axL.set_ylabel("loss")
    axL.grid(True)
    axL.legend()

    axR.plot(range(1, len(loss_graph_dict["bleu"]) + 1), loss_graph_dict["bleu"], linewidth=2, label="training")
    axR.plot(range(1, len(loss_graph_dict["val_bleu"]) + 1), loss_graph_dict["val_bleu"], linewidth=2, label="validation")
    axR.set_title("BLEU")
    axR.set_xlabel("epoch")
    axR.set_ylabel("BLEU")
    axR.grid(True)
    axR.legend()

    plt.savefig("loss.png", bbox_inches="tight")



def show_attention(input_sentence, output_words, attentions, num):
    # Set up figure with colorbar
    input_words = input_sentence

    fig, ax = plt.subplots()
    print("attn", attentions[:, :len(output_words)])
    print(output_words)
    cax = ax.matshow(attentions[:, :len(output_words)], cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_words + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    #plt.show()
    plt.savefig("attention_" + str(num) + ".png", bbox_inches="tight")

def show_plot(points, savefilename):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(savefilename, bbox_inches="tight")


def evaluate_randomly(pairs, encoder, decoder, source_lang, target_lang, n=10):
    scores = []
    for i in range(n):
        pair = random.choice(pairs)

        output_words, attentions = inference(encoder, decoder, pair[0], source_lang, target_lang)
        output_sentence = ' '.join(output_words)
        print('>', " ".join(pair[0]))
        print('=', " ".join(pair[1]))
        print('<', output_sentence)
        score = compute_bleu([pair[1]], [output_words[:-1]])
        print('bleu:', score)
        print('')
        if output_words[0] != __EOS__:
            show_attention(pair[0], output_words, attentions, i)
        scores.append(score)
    return scores

'''
def trainIters(encoder, decoder, tokenPairs, n_iters, source_lang, target_lang, print_every=100000, plot_every=100000, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimzer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorFromTokenPair(source_lang, target_lang, random.choice(tokenPairs)) for i in range(n_iters)]

    criterion = nn.NLLLoss()

    for iter in tqdm(range(1, n_iters + 1)):
        training_pair = training_pairs[iter - 1]
        source_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(source_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimzer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    #showPlot(plot_losses)
'''

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    pltfig, ax = plt.subplots()

    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluate(encoder, decoder, tokens, source_lang, target_lang, max_length=MAX_LENGTH):
    with torch.no_grad():
        source_tensor = tensorFromTokens(source_lang, tokens)
        source_length = source_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(source_length):
            encoder_output, encoder_hidden = encoder(source_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[__SOS_ID__]], device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)

            if topi.item() == __EOS_ID__:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(target_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, token_pairs, source_lang, target_lang, n=10):
    for i in range(n):
        pair = random.choice(token_pairs)
        print('>', ' '.join(pair[0]))
        print('=', ' '.join(pair[1]))
        output_words, attentions = evaluate(encoder, decoder, pair[0], source_lang, target_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def main():
    embedding_size = 256
    hidden_size = 256
    source_lang, target_lang, token_pairs = loadPkls()

    eval_pairs = loadEval()

    encoder = EncoderRNN(source_lang.n_words, embedding_size, hidden_size).to(device)
    decoder = AttnDecoderRNN1(embedding_size, hidden_size, target_lang.n_words).to(device)
    train_iters(encoder, decoder, token_pairs, eval_pairs, source_lang, target_lang, epochs=20, batch_size=200, teacher_forcing=0.5, early_stopping=3)

    """
    encoder1 = EncoderRNN(source_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, target_lang.n_words, dropout_p=0.1).to(device)

    trainIters(encoder1, attn_decoder1, token_pairs, 2000000, source_lang, target_lang, print_every=1000)
    """






if __name__ == '__main__':
    main()