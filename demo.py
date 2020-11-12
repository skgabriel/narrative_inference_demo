import json
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from transformers import GPT2Tokenizer
from src.text_utils import TextEncoder, fix_malformed
from src.transformer_models import GPT2Model, GPT2LMHeadModel, GPT2MemModel
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_type',type=str,default='nomem')
    parser.add_argument('--model_dir',type=str,default='./')
    parser.add_argument('--source',type=str,default='example.jsonl')
    parser.add_argument('--save_filename',type=str,default='outputs.jsonl')
    parser.add_argument('--decoding',type=str,default='beam')
    parser.add_argument('--beam',type=int,default=10)
args = parser.parse_args()
print(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.model_type == 'mem':
   use_mem = True
else:
   use_mem = False

model_path = os.path.join(args.model_dir, args.model_type)
device = torch.device(device)
text_encoder = GPT2Tokenizer.from_pretrained('gpt2')
encoder = text_encoder.encoder
decoder = text_encoder.decoder

#sentence-level special tokens
encoder['<|sent0|>'] = len(encoder)
decoder[len(decoder)] = '<|sent0|>'

encoder['<|sent1|>'] = len(encoder)
decoder[len(decoder)] = '<|sent1|>'

encoder['<|sent2|>'] = len(encoder)
decoder[len(decoder)] = '<|sent2|>'

encoder['<|sent3|>'] = len(encoder)
decoder[len(decoder)] = '<|sent3|>'

encoder['<|sent4|>'] = len(encoder)
decoder[len(decoder)] = '<|sent4|>'

sent_ids = [encoder['<|sent0|>'],encoder['<|sent1|>'],encoder['<|sent2|>'],encoder['<|sent3|>'],encoder['<|sent4|>']]

#ATOMIC special tokens
encoder['<|xNeed|>'] = len(encoder)
decoder[len(decoder)] = '<|xNeed|>'

encoder['<|xIntent|>'] = len(encoder)
decoder[len(decoder)] = '<|xIntent|>'

encoder['<|xWant|>'] = len(encoder)
decoder[len(decoder)] = '<|xWant|>'

encoder['<|oEffect|>'] = len(encoder)
decoder[len(decoder)] = '<|oEffect|>'

encoder['<|xReact|>'] = len(encoder)
decoder[len(decoder)] = '<|xReact|>'

encoder['<|oWant|>'] = len(encoder)
decoder[len(decoder)] = '<|oWant|>'

encoder['<|oReact|>'] = len(encoder)
decoder[len(decoder)] = '<|oReact|>'

encoder['<|xEffect|>'] = len(encoder)
decoder[len(decoder)] = '<|xEffect|>'

encoder['<|xAttr|>'] = len(encoder)
decoder[len(decoder)] = '<|xAttr|>'

encoder['<|PAD|>'] = len(encoder)
decoder[len(decoder)] = '<|PAD|>'

text_encoder.encoder = encoder
text_encoder.decoder = decoder
n_vocab = len(text_encoder.encoder)

if args.model_type == 'mem':
    model = GPT2MemModel.from_pretrained('gpt2')
else:
    model = GPT2LMHeadModel.from_pretrained('gpt2')

model.resize_token_embeddings(n_vocab)
model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

n_gpu = 1
print("device", device, "n_gpu", n_gpu)
gen_len = 50

def topk(model, XMB,i, n=1,k=args.beam, mem=None,use_pointer=None,use_scores=None,size_mem=0):
    import copy
    gen = torch.Tensor([encoder['<|PAD|>']] * gen_len).long().to(device) #torch.zeros((gen_len)).long().to(device)
    prob = 0
    for step in range(gen_len):
        if encoder['<|endoftext|>'] in gen:
           break
        if step == 0:
           clear_mem = True
        else:
           clear_mem = False
        XMB[:,i+1:i+gen_len+1] = gen
        if mem == None:
           logits, _ =  model(XMB[:,:i+1+step])
        else:
           if step != 0:
              mem = None
           logits, _ =  model(XMB[:,:i+1+step],update_mem=mem,clear_mem=clear_mem,use_mem=use_mem,size_mem=size_mem)
        logits = torch.nn.functional.softmax(logits, dim=-1)
        logits = logits[:,i+step].squeeze(1)
        values, indices  = logits.sort(descending=True)
        next_indices = indices[:, :k].gather(-1, torch.multinomial(values[:, :k], 1))
        gen[step] = next_indices.view(-1).long() 
        prob += float(values[:,int(gen[step])])
    return [gen]

class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret

def beam_search(model, XMB, start_id, num_beams=1, max_length=gen_len, temperature=1, pad_token_id=encoder['<|PAD|>'], eos_token_ids=[encoder['<|endoftext|>']], length_penalty=1,mem=None,size_mem=0,use_mem=False):
    """ Generate sequences for each example with beam search.
    """

    # generated hypotheses
    vocab_size = len(encoder)
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=False) for _ in range(1)
    ]

    # scores for each sentence in the beam
    beam_scores = torch.zeros((1, num_beams), dtype=torch.float, device=XMB.device)
    # Greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

    # cache compute states
    past = None

    # done sentences
    done = [False for _ in range(1)]

    step = 0 
    XMB = XMB[:,:start_id+1]
    if mem != None:
       mem = mem.expand_as(torch.zeros(num_beams, mem.size(1),mem.size(2)))
    XMB = XMB.expand_as(torch.zeros(num_beams,XMB.size(1)))
    while step < max_length:
        if step == 0 and use_mem:
           clear_mem = True
        else:
           clear_mem = False

        if mem == None:
           if use_mem:
              outputs = model(XMB,size_mem=size_mem)  # (batch_size * num_beams, cur_len, vocab_size)
           else:
              outputs = model(XMB)
        else:
           if step != 0:
              mem = None
           outputs = model(XMB,update_mem=mem,clear_mem=clear_mem,use_pointer=False, use_scores=False,mem_k=1,use_mem=use_mem,size_mem=size_mem)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

        # do greedy beam search
        scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
        assert scores.size() == (num_beams, vocab_size)
        # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
        next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
        # re-organize to group the beam together (we are keeping top hypothesis accross beams)
        next_scores = next_scores.view(1, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
        next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
        assert next_scores.size() == next_tokens.size() == (1, 2 * num_beams)

        # next batch beam content
        # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
        next_batch_beam = []

        # for each sentence
        for batch_idx in range(1):

            # if we are done with this sentence
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item()
            )
            if done[batch_idx]:
                assert (
                    len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                    eos_token_ids is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next tokens for this sentence
            for idx, score in zip(next_tokens[batch_idx], next_scores[batch_idx]):

                # get beam and word IDs
                beam_id = idx // vocab_size
                token_id = idx % vocab_size

                effective_beam_id = batch_idx * num_beams + beam_id
                # add to generated hypotheses if end of sentence
                if eos_token_ids is not None and token_id.item() in eos_token_ids:
                    generated_hyps[batch_idx].add(
                        XMB[effective_beam_id].clone(), score.item(),
                    )
                else:
                    # add next predicted word if it is not eos_token
                    next_sent_beam.append((score, token_id, effective_beam_id))

                # the beam for next step is full
                if len(next_sent_beam) == num_beams:
                    break

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1)

        # sanity check / prepare next batch
        assert len(next_batch_beam) == 1 * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = XMB.new([x[1] for x in next_batch_beam])
        beam_idx = XMB.new([x[2] for x in next_batch_beam])

        # re-order batch
        XMB = XMB[beam_idx, :]
        XMB = torch.cat([XMB, beam_tokens.unsqueeze(1)], dim=-1)

        # re-order internal states
        if past:
            past = self._reorder_cache(past, beam_idx)

        # stop when we are done with each sentence
        if all(done):
            break

        # update current length
        step = step + 1

    # finalize all open beam hypotheses and end to generated hypotheses
    for batch_idx in range(1):
        if done[batch_idx]:
            continue

        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        if eos_token_ids is not None and all(
            (token_id % vocab_size).item() not in eos_token_ids for token_id in next_tokens[batch_idx]
        ):
            assert torch.all(
                next_scores[batch_idx, :num_beams] == beam_scores.view(1, num_beams)[batch_idx]
            ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                next_scores[:, :num_beams][batch_idx], beam_scores.view(1, num_beams)[batch_idx]
            )

        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = XMB[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)

    # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
    output_batch_size = 1 
    output_num_return_sequences_per_batch = 1

    # select the best hypotheses
    sent_lengths = XMB.new(output_batch_size)
    best = []

    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)

    # shorter batches are filled with pad_token
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`Pad_token_id` has to be defined"
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded = XMB.new(output_batch_size, sent_max_len).fill_(pad_token_id)

        # fill with hypothesis and eos_token_id if necessary
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_ids[0]
    else:
        # none of the hypotheses have an eos_token
        assert (len(hypo) == max_length for hypo in best)
        decoded = torch.stack(best).type(torch.long).to(next(model.parameters()).device)
    gens = [decoded[0][start_id+1:]] + [s[1][start_id+1:] for s in sorted_hyps]
    return gens
 
def get_token(next_idx):
    try:
       return text_encoder.decoder[next_idx]
    except:
       return next_idx

gen_file = open(os.path.join(args.model_type + '_' + args.decoding + '_' + args.save_filename),'w')

dims_ = ["<|xNeed|>","<|xIntent|>","<|xWant|>","<|oEffect|>","<|xReact|>","<|oWant|>","<|oReact|>","<|xEffect|>","<|xAttr|>"]
dims = [encoder[d] for d in dims_]

def clean_gen(gen):
    gen = [w for w in gen.tolist() if w != encoder['<|PAD|>']]
    gen = [get_token(idx) for idx in gen]
    if '<unk>' in gen:
       gen = [t for t in gen if t != '<unk>']
    gen = "".join([word.replace("Ġ", " ") for word in gen])
    gen = gen.replace("<|endoftext|>","")
    gen = fix_malformed(gen)
    return gen

def pad_rels(relation, pad_len=100):
    return relation[:100] + [encoder['<|PAD|>']] * (100-len(relation[:100]))

def handle_empty(list_of_rels):
    if len(list_of_rels) == 0:
       return [[] for i in range(max_mem_size)]
    if len(list_of_rels) < max_mem_size:
       list_of_rels.extend([[] for i in range(max_mem_size - len(list_of_rels))])
    return list_of_rels 

if use_mem:
   external_mem = {}

max_mem_size = 45
n_updates = 0 

teX = [json.loads(l) for l in open(args.source).readlines()]
for line in teX:
    id = line['id']
    print(id)
    if use_mem:
       if id not in external_mem.keys():
          external_mem[id] = []
          size_mem = 0
    original = line['full_context'][:5]
    save_output = {}
    save_output["storyid"] = id
    save_output["story"] = ' '.join(original)
    for i in range(len(original)):
        sent_id = "<|sent" + str(i) + "|>"
        with torch.no_grad():
             for d in range(len(dims)):
                 XMB = [encoder['<|PAD|>']] * 600
                 if ('xIntent' in dims_[d] or 'xNeed' in dims_[d] or 'xAttr' in dims_[d]):
                    context = text_encoder.convert_tokens_to_ids(text_encoder.tokenize(' '.join(original[:i+1])))
                 else:
                    context = text_encoder.convert_tokens_to_ids(text_encoder.tokenize(' '.join(original)))
                 context += [encoder[sent_id], dims[d]]
                 i_1 = len(context)-1
                 XMB[:len(context)] = context 
                 XMB = torch.tensor(XMB,dtype=torch.long).to(device)
                 XMB = XMB.unsqueeze(0)
                 if use_mem and size_mem != 0:
                    mem = external_mem[id]
                    mem = torch.LongTensor([pad_rels(r) for r in mem]).unsqueeze(0)
                    if args.decoding == 'topk':
                       gen = topk(model, XMB,i_1,mem=mem,size_mem=size_mem)
                    else:
                       gen = beam_search(model, XMB,i_1,mem=mem,num_beams=args.beam,size_mem=size_mem,use_mem=use_mem)
                 else:
                    if args.decoding == 'topk':
                       if use_mem:
                          gen = topk(model, XMB, i_1,size_mem=size_mem)
                       else:
                          gen = topk(model, XMB, i_1)
                    else:
                       if use_mem:
                          gen = beam_search(model, XMB, i_1,num_beams=args.beam,size_mem=size_mem,use_mem=use_mem)
                       else:
                          gen = beam_search(model, XMB, i_1,num_beams=args.beam)
                 gen = [clean_gen(g) for g in gen]
                 if use_mem:
                    mem_gen = gen[0]
                    external_mem[id].append(text_encoder.convert_tokens_to_ids(text_encoder.tokenize(mem_gen)))
                    size_mem += 1
                 if sent_id + '_' + "generated_relations" in save_output.keys(): 
                    save_output[sent_id + '_' + "generated_relations"].append(gen)
                    save_output[sent_id + '_' + "generated_dims"].append([decoder[dims[d]]] * len(gen))
                 else:
                    save_output[sent_id + '_' + "generated_relations"] = [gen]
                    save_output[sent_id + '_' + "generated_dims"] = [[decoder[dims[d]]] * len(gen)]
    gen_file.write(json.dumps(save_output) + '\n')
    n_updates += 1
