import os
import copy
import pickle

import torch
import transformers

from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

class CorrectionNetwork(nn.Module):

    def __init__(self):
        super(CorrectionNetwork, self).__init__()
        # BERT分词器，作者并没提到自己使用的是哪个中文版的bert，我这里就使用一个比较常用的
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        # BERT
        self.bert = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        # BERT的word embedding，本质就是个nn.Embedding
        self.word_embedding_table = self.bert.get_input_embeddings()
        # 预测层。hidden_size是词向量的大小，len(self.tokenizer)是词典大小
        self.dense_layer = nn.Linear(self.bert.config.hidden_size, len(self.tokenizer))

    def forward(self, inputs, word_embeddings, detect_hidden_states):
        """
        Correction Network的前向传递
        :param inputs: inputs为tokenizer对中文文本的分词结果，
                       里面包含了token对一个的index，attention_mask等
        :param word_embeddings: 使用BERT的word_embedding对token进行embedding后的结果
        :param detect_hidden_states: Detection Network输出hidden state
        :return: Correction Network对个token的预测结果。
        """
        # 1. 使用bert进行前向传递
        bert_outputs = self.bert(token_type_ids=inputs['token_type_ids'],
                                 attention_mask=inputs['attention_mask'],
                                 inputs_embeds=word_embeddings)
        # 2. 将bert的hidden_state和Detection Network的hidden state进行融合。
        hidden_states = bert_outputs['last_hidden_state'] + detect_hidden_states
        # 3. 最终使用全连接层进行token预测
        return self.dense_layer(hidden_states)

    def get_inputs_and_word_embeddings(self, sequences, max_length=128):
        """
        对中文序列进行分词和word embeddings处理
        :param sequences: 中文文本序列。例如: ["鸡你太美", "哎呦，你干嘛！"]
        :param max_length: 文本的最大长度，不足则进行填充，超出进行裁剪。
        :return: tokenizer的输出和word embeddings.
        """
        inputs = self.tokenizer(sequences, padding='max_length', max_length=max_length, return_tensors='pt',
                                truncation=True).to(device)
        # 使用BERT的work embeddings对token进行embedding，这里得到的embedding并不包含position embedding和segment embedding
        word_embeddings = self.word_embedding_table(inputs['input_ids'])
        return inputs, word_embeddings

class DetectionNetwork(nn.Module):

    def __init__(self, position_embeddings, transformer_blocks, hidden_size):
        """
        :param position_embeddings: bert的position_embeddings，本质是一个nn.Embedding
        :param transformer: BERT的前两层transformer_block，其是一个ModuleList对象
        """
        super(DetectionNetwork, self).__init__()
        self.position_embeddings = position_embeddings
        self.transformer_blocks = transformer_blocks

        # 定义最后的预测层，预测哪个token是错误的
        self.dense_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, word_embeddings):
        # 获取token序列的长度，这里为128
        sequence_length = word_embeddings.size(1)
        # 生成position embedding
        position_embeddings = self.position_embeddings(torch.LongTensor(range(sequence_length)).to(device))
        # 融合work_embedding和position_embedding
        x = word_embeddings + position_embeddings
        # 将x一层一层的使用transformer encoder进行向后传递
        for transformer_layer in self.transformer_blocks:
            x = transformer_layer(x)[0]

        # 最终返回Detection Network输出的hidden states和预测结果
        hidden_states = x
        return hidden_states, self.dense_layer(hidden_states)

class MDCSpellModel(nn.Module):

    def __init__(self):
        super(MDCSpellModel, self).__init__()
        # 构造Correction Network
        self.correction_network = CorrectionNetwork()
        self._init_correction_dense_layer()

        # 构造Detection Network
        # position embedding使用BERT的
        position_embeddings = self.correction_network.bert.embeddings.position_embeddings
        # 作者在论文中提到的，Detection Network的Transformer使用BERT的权重
        # 所以我这里直接克隆BERT的前两层Transformer来完成这个动作
        transformer = copy.deepcopy(self.correction_network.bert.encoder.layer[:2])
        # 提取BERT的词向量大小
        hidden_size = self.correction_network.bert.config.hidden_size

        # 构造Detection Network
        self.detection_network = DetectionNetwork(position_embeddings, transformer, hidden_size)

    def forward(self, sequences, max_length=128):
        # 先获取word embedding，Correction Network和Detection Network都要用
        inputs, word_embeddings = self.correction_network.get_inputs_and_word_embeddings(sequences, max_length)
        # Detection Network进行前向传递，获取输出的Hidden State和预测结果
        hidden_states, detection_outputs = self.detection_network(word_embeddings)
        # Correction Network进行前向传递，获取其预测结果
        correction_outputs = self.correction_network(inputs, word_embeddings, hidden_states)
        # 返回Correction Network 和 Detection Network 的预测结果。
        # 在计算损失时`[PAD]`token不需要参与计算，所以这里将`[PAD]`部分全都变为0
        return correction_outputs, detection_outputs.squeeze(2) * inputs['attention_mask']

    def _init_correction_dense_layer(self):
        """
        原论文中提到，使用Word Embedding的weight来对Correction Network进行初始化
        """
        self.correction_network.dense_layer.weight.data = self.correction_network.word_embedding_table.weight.data

model = MDCSpellModel()
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

pretrained_model_path = './drive/MyDrive/models/MDCSpell-finetuned/MDCSpell-model.pt'
checkpoint = torch.load(pretrained_model_path, map_location=device)
model.load_state_dict(checkpoint['model'])
print("成功加载预训练模型权重！")
model = model.to(device)

def score_f_sent(inputs, golds, preds):
    assert len(inputs) == len(golds)
    assert len(golds) == len(preds)
    total_gold_err, total_pred_err, right_pred_err = 0, 0, 0
    check_right_pred_err = 0
    fout = open('sent_pred_result_15.txt', 'w', encoding='utf-8')
    for ori_tags, god_tags, prd_tags in zip(inputs, golds, preds):
        if None in ori_tags or None in god_tags or None in prd_tags:
            continue
        assert len(ori_tags) == len(god_tags)
        assert len(god_tags) == len(prd_tags)
        gold_errs = [idx for (idx, tk) in enumerate(god_tags) if tk != ori_tags[idx]]
        pred_errs = [idx for (idx, tk) in enumerate(prd_tags) if tk != ori_tags[idx]]
        if len(gold_errs) > 0 or len(pred_errs) > 0:
            fout.writelines('\n%s\n%s\n%s\n' % ('|'.join(ori_tags), '|'.join(god_tags),'|'.join(prd_tags)))
        if len(gold_errs) > 0:
            total_gold_err += 1
            fout.writelines('gold_err\n')
        if len(pred_errs) > 0:
            fout.writelines('check_err\n')
            total_pred_err += 1
            if gold_errs == pred_errs:
                check_right_pred_err += 1
                fout.writelines('check_right\n')
            if god_tags == prd_tags:
                right_pred_err += 1
                fout.writelines('correct_right\n')
    fout.close()
    p = 1. * check_right_pred_err / total_pred_err
    r = 1. * check_right_pred_err / total_gold_err
    f = 2 * p * r / (p + r + 1e-13)
    #print(total_gold_err, total_pred_err, right_pred_err, check_right_pred_err)
    print('sent check: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
    p = 1. * right_pred_err / total_pred_err
    r = 1. * right_pred_err / total_gold_err
    f = 2 * p * r / (p + r + 1e-13)
    print('sent correction: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
    return p, r, f

def score_f(ans, print_flg=False, only_check=False):
    fout = open('pred_15.txt', 'w', encoding="utf-8")
    total_gold_err, total_pred_err, right_pred_err = 0, 0, 0
    check_right_pred_err = 0
    inputs, golds, preds = ans
    assert len(inputs) == len(golds)
    assert len(golds) == len(preds)
    for ori, god, prd in zip(inputs, golds, preds):
        ori_txt = str(ori)
        god_txt = str(god) #''.join(list(map(str, god)))
        prd_txt = str(prd) #''.join(list(map(str, prd)))
        if print_flg is True:
            print(ori_txt, '\t', god_txt, '\t', prd_txt)
        if 'UNK' in ori_txt:
            continue
        if ori_txt == god_txt and ori_txt == prd_txt:
            continue
        if prd_txt != god_txt:
            fout.writelines('%s\t%s\t%s\n' % (ori_txt, god_txt, prd_txt)) 
        if ori != god:
            total_gold_err += 1
        if prd != ori:
            total_pred_err += 1
        if (ori != god) and (prd != ori):
            check_right_pred_err += 1
            if god == prd:
                right_pred_err += 1
    fout.close()

    #check p, r, f
    p = 1. * check_right_pred_err / (total_pred_err + 0.001)
    r = 1. * check_right_pred_err / (total_gold_err + 0.001)
    f = 2 * p * r / (p + r +  1e-13)
    print('token check: p=%.3f, r=%.3f, f=%.3f' % (p, r, f))
    if only_check is True:
        return p, r, f

    #correction p, r, f
    #p = 1. * right_pred_err / (total_pred_err + 0.001)
    pc = 1. * right_pred_err / (check_right_pred_err + 0.001)
    rc = 1. * right_pred_err / (total_gold_err + 0.001)
    fc = 2 * pc * rc / (pc + rc + 1e-13) 
    print('token correction: p=%.3f, r=%.3f, f=%.3f' % (pc, rc, fc))
    return p, r, f

def evaluation(test_data):
    all_inputs = []
    all_golds = []
    all_preds = []
    
    inputs = []
    golds = []
    preds = []

    prograss = tqdm(range(len(test_data)))
    for i in prograss:
        src, tgt = test_data[i]['src'], test_data[i]['tgt']

        src_enc = tokenizer(src, return_tensors='pt', max_length=128, truncation=True)
        tgt_enc = tokenizer(tgt, return_tensors='pt', max_length=128, truncation=True)

        src_tokens = src_enc['input_ids'][0][1:-1]  # 去掉 [CLS] 和 [SEP]
        tgt_tokens = tgt_enc['input_ids'][0][1:-1]

        if len(src_tokens) != len(tgt_tokens):
            print("第%d条数据异常" % i)
            continue

        # 模型預測
        correction_outputs, _ = model(src)
        predict_ids = correction_outputs[0][1:len(src_tokens) + 1].argmax(1).detach().cpu()

        # Tensor to tokens
        ori_tokens = tokenizer.convert_ids_to_tokens(src_tokens)
        gold_tokens = tokenizer.convert_ids_to_tokens(tgt_tokens)
        pred_tokens = tokenizer.convert_ids_to_tokens(predict_ids)

        for l in range(len(ori_tokens)):
            inputs.append(ori_tokens[l])
            golds.append(gold_tokens[l])
            preds.append(pred_tokens[l])

        all_inputs.append(ori_tokens)
        all_golds.append(gold_tokens)
        all_preds.append(pred_tokens)
        
    # 評分
    score_f_sent(all_inputs, all_golds, all_preds)

    score_f((inputs, golds, preds), only_check=False)

arr = []
with open("sighan13_test_set_traditional.pkl", mode='br') as f:
    data = pickle.load(f)
    for item in data:
        if item['src'] != item['tgt']:
            arr.append(item)
evaluation(arr)
