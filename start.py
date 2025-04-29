import os
import copy
import json

import torch
import transformers

from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# 句子的长度，作者并没有说明。我这里就按经验取一个
max_length = 128
# 作者使用的batch_size
batch_size = 32
# epoch数，作者并没有具体说明，按经验取一个
epochs = 10

# 每${log_after_step}步，打印一次日志
log_after_step = 20

# 模型存放的位置。
model_path = './drive/MyDrive/models/MDCSpell/'
os.makedirs(model_path, exist_ok=True)
model_path = model_path + 'MDCSpell-model.pt'

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

model = MDCSpellModel().to(device)
# correction_outputs, detection_outputs = model(["鸡你太美", "哎呦，你干嘛！"])
# print("correction_outputs shape:", correction_outputs.size())
# print("detection_outputs shape:", detection_outputs.size())

class MDCSpellLoss(nn.Module):

    def __init__(self, coefficient=0.85):
        super(MDCSpellLoss, self).__init__()
        # 定义Correction Network的Loss函数
        self.correction_criterion = nn.CrossEntropyLoss(ignore_index=0)
        # 定义Detection Network的Loss函数，因为是二分类，所以用Binary Cross Entropy
        self.detection_criterion = nn.BCELoss()
        # 权重系数
        self.coefficient = coefficient

    def forward(self, correction_outputs, correction_targets, detection_outputs, detection_targets):
        """
        :param correction_outputs: Correction Network的输出，Shape为(batch_size, sequence_length, hidden_size)
        :param correction_targets: Correction Network的标签，Shape为(batch_size, sequence_length)
        :param detection_outputs: Detection Network的输出，Shape为(batch_size, sequence_length)
        :param detection_targets: Detection Network的标签，Shape为(batch_size, sequence_length)
        :return:
        """
        # 计算Correction Network的loss，因为Shape维度为3，所以要把batch_size和sequence_length进行合并才能计算
        correction_loss = self.correction_criterion(correction_outputs.view(-1, correction_outputs.size(2)),
                                                    correction_targets.view(-1))
        # 计算Detection Network的loss
        detection_loss = self.detection_criterion(detection_outputs, detection_targets)
        # 对两个loss进行加权平均
        return self.coefficient * correction_loss + (1 - self.coefficient) * detection_loss

class CSCDataset(Dataset):

    def __init__(self):
        super(CSCDataset, self).__init__()
        with open("traditional_wang127k.json", encoding='utf-8') as f:
            train_data = json.load(f)

        self.train_data = train_data

    def __getitem__(self, index):
        src = self.train_data[index]['source']
        tgt = self.train_data[index]['target']
        return src, tgt

    def __len__(self):
        return len(self.train_data)
train_data = CSCDataset()
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
def collate_fn(batch):
    src, tgt = zip(*batch)
    src, tgt = list(src), list(tgt)

    src_tokens = tokenizer(src, padding='max_length', max_length=128, return_tensors='pt', truncation=True)['input_ids']
    tgt_tokens = tokenizer(tgt, padding='max_length', max_length=128, return_tensors='pt', truncation=True)['input_ids']

    correction_targets = tgt_tokens
    detection_targets = (src_tokens != tgt_tokens).float()
    return src, correction_targets, detection_targets, src_tokens  # src_tokens在计算Correction的精准率时要用到
train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
criterion = MDCSpellLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
start_epoch = 0  # 从哪个epoch开始
total_step = 0  # 一共更新了多少次参数
# 恢复之前的训练
if os.path.exists(model_path):
    if not torch.cuda.is_available():
        checkpoint = torch.load(model_path, map_location='cpu')
    else:
        checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    total_step = checkpoint['total_step']
    print("恢复训练，epoch:", start_epoch)
model = model.to(device)
model = model.train()
total_loss = 0.  # 记录loss

d_recall_numerator = 0  # Detection的Recall的分子
d_recall_denominator = 0  # Detection的Recall的分母
d_precision_numerator = 0  # Detection的precision的分子
d_precision_denominator = 0  # Detection的precision的分母
c_recall_numerator = 0  # Correction的Recall的分子
c_recall_denominator = 0  # Correction的Recall的分母
c_precision_numerator = 0  # Correction的precision的分子
c_precision_denominator = 0  # Correction的precision的分母

for epoch in range(start_epoch, epochs):

    step = 0

    for sequences, correction_targets, detection_targets, correction_inputs in train_loader:
        correction_targets, detection_targets = correction_targets.to(device), detection_targets.to(device)
        correction_inputs = correction_inputs.to(device)
        correction_outputs, detection_outputs = model(sequences)
        loss = criterion(correction_outputs, correction_targets, detection_outputs, detection_targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        step += 1
        total_step += 1

        total_loss += loss.detach().item()

        # 计算Detection的recall和precision指标
        # 大于0.5，认为是错误token，反之为正确token
        d_predicts = detection_outputs >= 0.5
        # 计算错误token中被网络正确预测到的数量
        d_recall_numerator += d_predicts[detection_targets == 1].sum().item()
        # 计算错误token的数量
        d_recall_denominator += (detection_targets == 1).sum().item()
        # 计算网络预测的错误token的数量
        d_precision_denominator += d_predicts.sum().item()
        # 计算网络预测的错误token中，有多少是真错误的token
        d_precision_numerator += (detection_targets[d_predicts == 1]).sum().item()

        # 计算Correction的recall和precision
        # 将输出映射成index，即将correction_outputs的Shape由(32, 128, 21128)变为(32,128)
        correction_outputs = correction_outputs.argmax(2)
        # 对于填充、[CLS]和[SEP]这三个token不校验
        correction_outputs[(correction_targets == 0) | (correction_targets == 101) | (correction_targets == 102)] = 0
        # correction_targets的[CLS]和[SEP]也要变为0
        correction_targets[(correction_targets == 101) | (correction_targets == 102)] = 0
        # Correction的预测结果，其中True表示预测正确，False表示预测错误或无需预测
        c_predicts = correction_outputs == correction_targets
        # 计算错误token中被网络正确纠正的token数量
        c_recall_numerator += c_predicts[detection_targets == 1].sum().item()
        # 计算错误token的数量
        c_recall_denominator += (detection_targets == 1).sum().item()
        # 计算网络纠正token的数量
        correction_inputs[(correction_inputs == 101) | (correction_inputs == 102)] = 0
        c_precision_denominator += (correction_outputs != correction_inputs).sum().item()
        # 计算在网络纠正的这些token中，有多少是真正被纠正对的
        c_precision_numerator += c_predicts[correction_outputs != correction_inputs].sum().item()

        if total_step % log_after_step == 0:
            loss = total_loss / log_after_step
            d_recall = d_recall_numerator / (d_recall_denominator + 1e-9)
            d_precision = d_precision_numerator / (d_precision_denominator + 1e-9)
            c_recall = c_recall_numerator / (c_recall_denominator + 1e-9)
            c_precision = c_precision_numerator / (c_precision_denominator + 1e-9)

            print("Epoch {}, "
                  "Step {}/{}, "
                  "Total Step {}, "
                  "loss {:.5f}, "
                  "detection recall {:.4f}, "
                  "detection precision {:.4f}, "
                  "correction recall {:.4f}, "
                  "correction precision {:.4f}".format(epoch, step, len(train_loader), total_step,
                                                       loss,
                                                       d_recall,
                                                       d_precision,
                                                       c_recall,
                                                       c_precision))

            total_loss = 0.
            total_correct = 0
            total_num = 0
            d_recall_numerator = 0
            d_recall_denominator = 0
            d_precision_numerator = 0
            d_precision_denominator = 0
            c_recall_numerator = 0
            c_recall_denominator = 0
            c_precision_numerator = 0
            c_precision_denominator = 0

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1,
        'total_step': total_step,
    }, model_path)
