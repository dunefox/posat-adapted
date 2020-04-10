"""
Train a model on TACRED.
"""

import os
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim

# eigener Code
from torch.utils.tensorboard import SummaryWriter
from utils import torch_utils, scorer, constant, helper
writer = SummaryWriter()

from data.loader import DataLoader
from model.rnn import RelationModel as RelationModel
from model.rnn_matrix import RelationModel as RelationModel2
from model.rnn_matrix_concat import RelationModel as RelationModel3
from utils import scorer, constant, helper
from utils.vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--vocab_dir', type=str, default='dataset/vocab')
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
parser.add_argument('--dropout', type=float, default=0.5, help='Input and RNN dropout rate.')
parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N embeddings.')
parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
parser.add_argument('--no-lower', dest='lower', action='store_false')

# eigener Code
parser.add_argument('--fine_tune', type=bool, default=False, help="Fine-tune loaded model (last layer)")
parser.add_argument('--fine_tune_attn', type=bool, default=False, help="Fine-tune loaded model (attention layer)")
parser.add_argument('--fine_tune_combine', type=bool, default=False, help="Fine-tune loaded model (attention layer)")
parser.add_argument('--erw1', type=bool, default=False, help="Fine-tune loaded model for extension 1")
parser.add_argument('--erw2', type=bool, default=False, help="Fine-tune loaded model for extension 2")
parser.add_argument('--weight', type=float, default=1.0, help="Weight for matrix regularisation (both extensions)")
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--model_dir', type=str, help='Name of the model directory.', required=True)

parser.set_defaults(lower=False)

parser.add_argument('--attn', dest='attn', action='store_true', help='Use attention layer.')
parser.add_argument('--no-attn', dest='attn', action='store_false')
parser.set_defaults(attn=True)
parser.add_argument('--attn_dim', type=int, default=200, help='Attention size.')
parser.add_argument('--pe_dim', type=int, default=30, help='Position encoding dimension.')

parser.add_argument('--lr', type=float, default=1.0, help='Applies to SGD and Adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--optim', type=str, default='sgd', help='sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=5, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# make opt
opt = vars(args)
opt['num_class'] = len(constant.LABEL_TO_ID)

# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
opt['vocab_size'] = vocab.size
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt['emb_dim']

# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'] + '/train.json', opt['batch_size'], opt, vocab, evaluation=False)
dev_batch = DataLoader(opt['data_dir'] + '/dev.json', opt['batch_size'], opt, vocab, evaluation=True)

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
vocab.save(model_save_dir + '/vocab.pkl')
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_f1")

# print model info
helper.print_config(opt)

# eigener Code
# model
if opt['fine_tune']:
    # load opt
    print("Loading pre-trained model for finetuning last layer")
    model_file = opt['model_dir'] + '/' + opt['model']
    print("Loading model from {}".format(model_file))
    old_opt = torch_utils.load_config(model_file)
    model = RelationModel(old_opt)
    print("model:\n", model.model)

    #  model.model._modules['linear'] = nn.Linear(opt['hidden_dim'], opt['num_class']).cuda()

    print("#####################\nParams before swap")
    for name, param in model.model.named_parameters():
        param.requires_grad = False
        print(name, param.requires_grad)

    print("#####################\n")

    print("#####################\nParams after insert")
    for name, param in model.model.named_parameters():
        if name.startswith("linear"):
            param.requires_grad = True
        print(name, param.requires_grad)

    print("#####################\n")

    print("adapted model:\n", model.model)
elif opt['fine_tune_attn']:
    # load opt
    print("Loading pre-trained model for finetuning attention")
    model_file = opt['model_dir'] + '/' + opt['model']
    print("Loading model from {}".format(model_file))
    old_opt = torch_utils.load_config(model_file)
    model = RelationModel(old_opt)
    print("model:\n", model.model)

    model.load(model_file)
    for param in model.parameters:
        print("model param -> False? 1:\t", str(param))
        param.requires_grad = False
        print("model param -> False? 2:\t", str(param))

    for name, param in model.model.named_parameters():
        if name.startswith("attn_layer"):
            param.requires_grad = True

    print("adapted model:\n", model.model)

    # Vortrainiertes Modell M laden, neues Modell M' instantiieren
    # Gewichte austauschen
elif opt['fine_tune_combine']:
    print("Loading pre-trained model for comb. of erw2 and attn")
    model_file = opt['model_dir'] + '/' + opt['model']
    print("Loading model from {}".format(model_file))
    old_opt = torch_utils.load_config(model_file)
    old_model = RelationModel(old_opt)
    old_model.load(model_file)
    print("model:\n", old_model.model)
    print("Instantiating new model m'")
    model = RelationModel3(opt, emb_matrix=emb_matrix)
    print("New model:\n", model.model)
    print("\nSwapping weights\n")
    model.model.load_state_dict(old_model.model.state_dict(), strict=False)

    print("#####################\nParams before swap")
    for name, param in model.model.named_parameters():
        param.requires_grad = False
        print(name, param.requires_grad)

    print("#####################\n")

    print("#####################\nParams after swap")
    for name, param in model.model.named_parameters():
        if name.startswith("adapt"):
            param.requires_grad = True
        elif name.startswith("attn_layer"):
            param.requires_grad = True
        print(name, param.requires_grad)

    print("adapted model:\n", model.model)
elif opt['erw1']:
    print("Loading pre-trained model for extension 1")
    model_file = opt['model_dir'] + '/' + opt['model']
    print("Loading model from {}".format(model_file))
    old_opt = torch_utils.load_config(model_file)
    old_model = RelationModel(old_opt)
    print("old_model:\n", old_model.model)
    old_model.load(model_file)
    print("Instantiating new model m'")
    print(opt)
    model = RelationModel2(opt, emb_matrix=emb_matrix)

    print(f"Argument weight for model: {opt['weight']}")

    print("New model:\n", model.model)
    print("\nSwapping weights\n")
    model.model.load_state_dict(old_model.model.state_dict(), strict=False)

    print("#####################\nParams before swap")
    for name, param in model.model.named_parameters():
        param.requires_grad = False
        print(name, param.requires_grad)

    print("#####################\n")

    print("#####################\nParams after swap")
    for name, param in model.model.named_parameters():
        if name.startswith("adapt"):
            param.requires_grad = True
        print(name, param.requires_grad)

    print("\n##### weight debug ###\n")
    for name, param in model.model.named_parameters():
        if name.startswith("adapt"):
            print(name, param)

    print("Adapted model:\n", model.model)
elif opt['erw2']:
    print("Loading pre-trained model for extension 2")
    model_file = opt['model_dir'] + '/' + opt['model']
    print("Loading model from {}".format(model_file))
    old_opt = torch_utils.load_config(model_file)
    old_model = RelationModel(old_opt)
    old_model.load(model_file)
    print("model:\n", old_model.model)
    print("Instantiating new model m'")
    model = RelationModel3(opt, emb_matrix=emb_matrix)
    print("New model:\n", model.model)
    print("\nSwapping weights\n")
    model.model.load_state_dict(old_model.model.state_dict(), strict=False)

    print("#####################\nParams before swap")
    for name, param in model.model.named_parameters():
        param.requires_grad = False
        print(name, param.requires_grad)

    print("#####################\n")

    print("#####################\nParams after swap")
    for name, param in model.model.named_parameters():
        if name.startswith("adapt"):
            param.requires_grad = True
            
        print(name, param.requires_grad)

    # model.model.state_dict["adapt.weight"].copy_(
    #     torch.nn.Parameter(
    #         torch.cat(
    #             (model.model.state_dict["attn_layer.ulinear.weight"],
    #              torch.zeros(42, 42)),
    #             dim=1
    #         )
    #     )
    # )

    print("adapted model:\n", model.model)
else:
    print("Instantiating new model")
    model = RelationModel(opt, emb_matrix=emb_matrix)
    print("model:\n", model.model)

id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])
dev_f1_history = []
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']

# start training
for epoch in range(1, opt['num_epoch']+1):
    train_loss = 0
    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1
        loss = model.update(batch)
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))

    # eval on dev
    print("Evaluating on dev set...")
    predictions = []
    dev_loss = 0
    for i, batch in enumerate(dev_batch):
        preds, _, loss, _ = model.predict(batch)
        predictions += preds
        dev_loss += loss
    predictions = [id2label[p] for p in predictions]
    dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), predictions)

    train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
    dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']
    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,\
            train_loss, dev_loss, dev_f1))
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_f1))

    # eigener Code
    writer.add_scalar("F1/dev", dev_f1, epoch)
    writer.add_scalar("Loss/dev", dev_loss, epoch)
    writer.add_scalar("Loss/train", train_loss, epoch)

    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    model.save(model_file, epoch)
    if epoch == 1 or dev_f1 > max(dev_f1_history):
        copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)

    # eigener Code
    print("\n##### model saved ###")
    print("\n##### weight debug ###\n")
    for name, param in model.model.named_parameters():
        if name.startswith("adapt"):
            print(name, param)


    # lr schedule
    if len(dev_f1_history) > 10 and dev_f1 <= dev_f1_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad']:
        current_lr *= opt['lr_decay']
        model.update_lr(current_lr)

    dev_f1_history += [dev_f1]
    print("")

print("Training ended with {} epochs.".format(epoch))

