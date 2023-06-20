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
output_path = 'uie_nano_pytorch/model_best'
model_path = 'uie_nano_pytorch/model_best'

tokenizer = BertTokenizer.from_pretrained(model_path)
model = UIE.from_pretrained(model_path)
device = torch.device('cpu')

save_path = export_onnx(output_path, tokenizer, model, device, input_names, output_names)