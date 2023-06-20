# 导入相关库
import argparse
import shutil
import sys
import time
import os
import torch
from itertools import chain
from typing import List, Union
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
from transformers import (BertTokenizer, PreTrainedModel,
                          PreTrainedTokenizerBase, BertTokenizerFast)

from tools import IEDataset, logger, tqdm, set_seed, SpanEvaluator, EarlyStopping, logging_redirect_tqdm, logger
from model import UIE
from evaluate import evaluate
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3, 4, 5, 6, 7, 8"

# 设置相关参数
class args:
    batch_size = 96
    learning_rate = 1e-5
    train_path = 'datasets/6434c6eaaad2f9ce44d79682-momodel/data/competition_train.txt'
    dev_path = 'datasets/6434c6eaaad2f9ce44d79682-momodel/data/competition_valid.txt'
    save_dir = 'uie_nano_pytorch'
    max_seq_len = 512
    num_epochs = 300
    seed = 42
    logging_steps = 100
    valid_steps = 100
    # device = 'cpu'
    device = 'gpu'
    model = 'uie_nano_pytorch'
    max_model_num = 5

# 设置随机数种子
set_seed(args.seed)
show_bar = True

# 设置分词器，导入预训练模型
tokenizer = BertTokenizerFast.from_pretrained(args.model)
model = UIE.from_pretrained(args.model)

# 设置是否使用gpu
if args.device == 'gpu':
    model = model.cuda()


# 导入微调数据集和验证数据集
train_ds = IEDataset(args.train_path, tokenizer=tokenizer,
                     max_seq_len=args.max_seq_len)
dev_ds = IEDataset(args.dev_path, tokenizer=tokenizer,
                   max_seq_len=args.max_seq_len)
train_data_loader = DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True)
dev_data_loader = DataLoader(
    dev_ds, batch_size=args.batch_size, shuffle=True)

# 查看loader中的一个batch
input_ids, token_type_ids, att_mask, start_ids, end_ids = iter(train_data_loader).next()
print(input_ids)
print(input_ids.shape)

optimizer = torch.optim.AdamW(
    lr=args.learning_rate, params=model.parameters())

criterion = torch.nn.functional.binary_cross_entropy
metric = SpanEvaluator()

# 训练前的参数初始化置零
loss_list = []
loss_sum = 0
loss_num = 0
global_step = 0
best_step = 0
best_f1 = 0
tic_train = time.time()
epoch_iterator = range(1, args.num_epochs + 1)

if show_bar:
    train_postfix_info = {'loss': 'unknown'}
    epoch_iterator = tqdm(
        epoch_iterator, desc='Training', unit='epoch')

# 首先将模型导出为onnx格式
def export_onnx(output_path: Union[Path, str], tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel, device: torch.device, input_names: List[str], output_names: List[str]):
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        model.config.return_dict = True
        model.config.use_cache = False

        output_path = Path(output_path)

        # Create folder
        if not output_path.exists():
            output_path.mkdir(parents=True)
        save_path = output_path / "inference.onnx"

        dynamic_axes = {name: {0: 'batch', 1: 'sequence'}
                        for name in chain(input_names, output_names)}

        # Generate dummy input
        batch_size = 2
        seq_length = 6
        dummy_input = [" ".join([tokenizer.unk_token])
                       * seq_length] * batch_size
        inputs = dict(tokenizer(dummy_input, return_tensors="pt"))

        if save_path.exists():
            logger.warning(f'Overwrite model {save_path.as_posix()}')
            save_path.unlink()

        torch.onnx.export(model,
                          (inputs,),
                          save_path,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes,
                          do_constant_folding=True,
                          opset_version=11
                          )

    if not os.path.exists(save_path):
        logger.error(f'Export Failed!')

    return save_path

input_names = [
    'input_ids',
    'token_type_ids',
    'attention_mask',
]
output_names = [
    'start_prob',
    'end_prob'
]
output_path = './'
model_path = 'uie_nano_pytorch/model_best'

tokenizer = BertTokenizer.from_pretrained(model_path)
model = UIE.from_pretrained(model_path)
device = torch.device('cpu')

save_path = export_onnx(output_path, tokenizer, model, device, input_names, output_names)