#!/usr/bin/env python

# Copyright (c) 2017 Elad Hoffer
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch

import seq2seq.utils as utils
import seq2seq.data.config as config
from seq2seq.data.dataset import RawTextDataset
from seq2seq.data.tokenizer import Tokenizer
from seq2seq.inference.beam_search import SequenceGenerator
from seq2seq.models.gnmt import GNMT

class Translator:
    def __init__(self,
                 batch_size,
                 device,
                 bmm=False,
                 fp16=False):
        torch.backends.cudnn.enabled = False
        self.batch_size = batch_size
        self.device = device
        self.bmm = bmm
        self.fp16 = fp16

        self.model, self.generator = self.get_gnmt_model()
        self.loader, self.bos = self.get_data_loader()
        self.it = iter(self.loader)

    def get_gnmt_model(self):
        model_config = {'hidden_size': 1024,
                        'vocab_size': 32317,
                        'num_layers': 4,
                        'dropout': 0.2,
                        'batch_first': False,
                        'share_embedding': True,
                        }
        if self.bmm:
            model_config['encoder_batch_size'] = self.batch_size
            model_config['decoder_batch_size'] = self.batch_size * 5 #beam_size
        model = GNMT(**model_config)
        if self.fp16:
            model.type(torch.HalfTensor)
        else:
            model.type(torch.FloatTensor)
        model.eval()
        model = model.to(self.device)
        generator = SequenceGenerator(model=model,
                                      beam_size=5,
                                      max_seq_len=50,
                                      len_norm_factor=0.6,
                                      len_norm_const=5.0,
                                      cov_penalty_factor=0.1)
        generator = generator.beam_search
        return model, generator

    def get_data_loader(self):
        dataset_dir = '/cmsdata/ssd0/cmslab/wmt16_de_en'
        vocab = os.path.join(dataset_dir, 'vocab.bpe.32000')
        bpe_codes = os.path.join(dataset_dir, "bpe.32000")
        lang = {'src': 'en', 'tgt': 'de'}
        pad_vocab = utils.pad_vocabulary("fp16" if self.fp16 else "fp32")
        tokenizer = Tokenizer(vocab, bpe_codes, lang, pad_vocab)
        data = RawTextDataset(raw_datafile=os.path.join(dataset_dir, 'newstest2014.en'),
                             tokenizer=tokenizer,
                             sort=False)
        loader = data.get_loader(
                batch_size=self.batch_size,
                batch_first=False,
                pad=True,
                repeat=1,
                num_workers=0)
        bos = [[config.BOS]] * (self.batch_size * 5)
        bos = torch.tensor(bos, dtype=torch.int64, device=self.device)
        bos = bos.view(1, -1)
        return loader, bos

    def run_inference(self, src, src_length):
        context = self.model.encode(src, src_length)
        return context
        #context = [context, src_length, None]
        #return self.generator(self.batch_size, self.bos, context)
