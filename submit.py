###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
from model import UIE
import argparse
from tqdm import tqdm
from functools import partial

import torch
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

from tools import IEMapDataset, SpanEvaluator, IEDataset, convert_example, get_relation_type_dict, logger, tqdm, unify_prompt_name

bmodel_name = 'convert/uie-nano.bmodel' # 提交后的bmodel文件位置
model_name = 'uie_nano_pytorch' # 提交后的torch模型位置