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

# 正式开始训练
for epoch in epoch_iterator:
    train_data_iterator = train_data_loader
    if show_bar:
        train_data_iterator = tqdm(train_data_iterator,
                                   desc=f'Training Epoch {epoch}', unit='batch')
        train_data_iterator.set_postfix(train_postfix_info)

    # 迭代训练集
    for batch in train_data_iterator:
        if show_bar:
            epoch_iterator.refresh()

        # 取出每一个batch的输入输出
        input_ids, token_type_ids, att_mask, start_ids, end_ids = batch

        # 如果使用gpu，则将其放入到cuda中
        if args.device == 'gpu':
            input_ids = input_ids.cuda()
            token_type_ids = token_type_ids.cuda()
            att_mask = att_mask.cuda()
            start_ids = start_ids.cuda()
            end_ids = end_ids.cuda()

        # 模型推理预测
        outputs = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=att_mask)
        start_prob, end_prob = outputs[0], outputs[1]

        # 进行反向传播与loss计算
        start_ids = start_ids.type(torch.float32)
        end_ids = end_ids.type(torch.float32)
        loss_start = criterion(start_prob, start_ids)
        loss_end = criterion(end_prob, end_ids)
        loss = (loss_start + loss_end) / 2.0
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(float(loss))
        loss_sum += float(loss)
        loss_num += 1

        if show_bar:
            loss_avg = loss_sum / loss_num
            train_postfix_info.update({
                'loss': f'{loss_avg:.5f}'
            })
            train_data_iterator.set_postfix(train_postfix_info)

        global_step += 1
        if global_step % args.logging_steps == 0:
            time_diff = time.time() - tic_train
            loss_avg = loss_sum / loss_num

            if show_bar:
                with logging_redirect_tqdm([logger.logger]):
                    logger.info(
                        "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, loss_avg,
                           args.logging_steps / time_diff))
            else:
                logger.info(
                    "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, loss_avg,
                       args.logging_steps / time_diff))
            tic_train = time.time()

        # 迭代到一定次数后，对验证集进行评估
        if global_step % args.valid_steps == 0:
            save_dir = os.path.join(
                args.save_dir, "model_%d" % global_step)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model_to_save = model
            model_to_save.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            if args.max_model_num:
                model_to_delete = global_step-args.max_model_num*args.valid_steps
                model_to_delete_path = os.path.join(
                    args.save_dir, "model_%d" % model_to_delete)
                if model_to_delete > 0 and os.path.exists(model_to_delete_path):
                    shutil.rmtree(model_to_delete_path)

            dev_loss_avg, precision, recall, f1 = evaluate(
                model, metric, data_loader=dev_data_loader, device=args.device, loss_fn=criterion)

            if show_bar:
                train_postfix_info.update({
                    'F1': f'{f1:.3f}',
                    'dev loss': f'{dev_loss_avg:.5f}'
                })
                train_data_iterator.set_postfix(train_postfix_info)
                with logging_redirect_tqdm([logger.logger]):
                    logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f, dev loss: %.5f"
                                % (precision, recall, f1, dev_loss_avg))
            else:
                logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f, dev loss: %.5f"
                            % (precision, recall, f1, dev_loss_avg))

            # 如果模型F1指标最优，那么保存该模型
            # Save model which has best F1
            if f1 > best_f1:
                if show_bar:
                    with logging_redirect_tqdm([logger.logger]):
                        logger.info(
                            f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                        )
                else:
                    logger.info(
                        f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                    )
                best_f1 = f1
                save_dir = os.path.join(args.save_dir, "model_best")
                model_to_save = model
                model_to_save.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
            tic_train = time.time()

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
output_path = 'uie_nano_pytorch'
model_path = 'uie_nano_pytorch'

tokenizer = BertTokenizer.from_pretrained(model_path)
model = UIE.from_pretrained(model_path)
device = torch.device('cpu')

save_path = export_onnx(output_path, tokenizer, model, device, input_names, output_names)