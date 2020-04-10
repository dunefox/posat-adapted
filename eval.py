"""
Run evaluation with saved models.
"""

import os

import random
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from data.loader import DataLoader
from model.rnn import RelationModel
from model.rnn_matrix import RelationModel as RelationModel2
from model.rnn_matrix_concat import RelationModel as RelationModel3
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
parser.add_argument('--out', type=str, default='', help="Save model predictions to this dir.")
# eigener Code
parser.add_argument('--model_desc', type=str, help="Save Model Debugging information to this file", required=True)
parser.add_argument('--eval_mode', type=str, help="Which model to evaluate: standard, erw1, erw2", required=True)

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)

# eigener Code
# print("Adapting opt loaded from trained model")
# opt["model_dir"] = args.model_dir

if args.eval_mode == "standard":
    model = RelationModel(opt)
elif args.eval_mode == "erw1":
    model = RelationModel2(opt)
elif args.eval_mode == "erw2":
    model = RelationModel3(opt)

model.load(model_file)

# eigener Code
#  print(f"Argument weight for model: {opt['weight']}")
#  print("####### weight debug eval #####\n")
#  for name, param in model.model.named_parameters():
#      if name.startswith("adapt"):
#          print(name, param)
#
# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

helper.print_config(opt)
id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])

predictions = []
all_probs = []
all_attn_weights = []
for i, b in enumerate(batch):
    preds, probs, _, attn_weights = model.predict(b)
    predictions += preds
    all_probs += probs
    attn_weights = [vec.tolist() for vec in attn_weights]
    all_attn_weights += attn_weights
predictions = [id2label[p] for p in predictions]
p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)

# eigener Code
# Vorhersage, Goldlabel, Tokens, Attention
lines = ["\t".join([pred, b['relation'], " ".join(b['token']),
                    " ".join([str(b["subj_start"]), str(b["subj_end"])]),
                    " ".join([str(b["obj_start"]), str(b["obj_end"])]),
                    " ".join([str(x) for x in all_probs[i]]),
                    " ".join([str(x) for x in all_attn_weights[i]])])
          for i, (pred, b) in enumerate(zip(predictions, batch.raw_data))]
with open("/big/f/fuchsp/posat-adapted/debuginfo/" + args.model_desc, "w") as out_:
    for line in lines:
        out_.write(line + "\n")

# save probability scores
if len(args.out) > 0:
    helper.ensure_dir(os.path.dirname(args.out))
    with open(args.out, 'wb') as outfile:
        pickle.dump(all_probs, outfile)
    print("Prediction scores saved to {}.".format(args.out))

print("Evaluation ended.")

