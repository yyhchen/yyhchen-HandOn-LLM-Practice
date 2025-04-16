from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import zipfile
from PIL import Image
import io
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from typing import List, Dict, Any

class VLMConfig(PretrainedConfig):
    """视觉语言模型(VLM)的配置类
    继承自transformers.PretrainedConfig，用于定义模型的基本配置参数
    
    Attributes:
        vision_model_path (str): 视觉模型路径
        llm_model_path (str): 语言模型路径
        freeze_vision_model (bool): 是否冻结视觉模型参数
        image_pad_num (int): 图像pad token的数量
    """
    model_type = "vlm_model"
    def __init__(self,llm_model_path = '/hpcfs/CNG/yhchen/huggingface_model/Qwen/Qwen2.5-0.5B-Instruct',
                 vision_model_path = '/hpcfs/CNG/yhchen/huggingface_model/google/siglip-so400m-patch14-384',
                 freeze_vision_model = True,
                 image_pad_num = 49,    # 196/4 （压缩）==> (b, 196, d) --> (b, 49, d*4)   16x16
                **kwargs):
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.freeze_vision_model = freeze_vision_model
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs)
        
        
        
class VLM(PreTrainedModel):
    """视觉语言模型(VLM)主类
    实现多模态模型的前向传播逻辑，将视觉特征与文本特征融合
    
    Attributes:
        vision_model: 视觉特征提取模型
        llm_model: 语言模型
        processor: 视觉特征处理器
        tokenizer: 文本tokenizer
        linear1/linear2: 视觉特征投影层
    """
    # 指定该模型使用的配置类为VLMConfig，使transformers库能够正确识别和加载模型配置
    config_class = VLMConfig

    # 这里的config是通过VLMConfig类实例化得到的，用于初始化模型的参数
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path)
        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path)
        self.linear1 = nn.Linear(self.vision_model.config.vision_config.hidden_size*4, self.llm_model.config.hidden_size)
        self.linear2 = nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)
        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        for param in self.llm_model.parameters():   # 无条件冻结llm模型参数(避免破坏预训练模型的知识能力)
            param.requires_grad = False

      
    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        """模型的前向传播函数
        
        Args:
            input_ids (torch.Tensor): 输入文本的token ids
            labels (torch.Tensor): 训练目标标签
            pixel_values (torch.Tensor): 图像像素值
            attention_mask (torch.Tensor, optional): 注意力掩码，用于mask padding tokens
            
        Returns:
            CausalLMOutputWithPast: 包含loss和logits的输出对象
                - loss: 如果提供了labels，则返回计算的损失值
                - logits: 模型的预测logits
        """
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        
        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state 
        b, s, d = image_embeds.shape    # (batch_size, seq_len, hidden_dim)
        
        # (b, 196, d) --> (b, 49, d*4) 压缩图片tokens(降低语言模型要处理的序列长度)，但是要保持大小不变，所以4放到 hidden_dim中
        image_embeds = image_embeds.view(b, -1, d*4)
        image_features = self.linear2(F.silu(self.linear1(image_embeds)))
        
        # 将文本嵌入的数据类型转换为与图像特征相同的数据类型，确保两者可以正确融合
        text_embeds = text_embeds.to(image_features.dtype)
        
        # 将文本嵌入和图像特征合并为一个新的输入嵌入
        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        
        logits = outputs[0] # [batch_size, seq_len, vocab_size]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)    # 忽略padding token的计算
            
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), # 将logits展平为[seq_len*batch_size, vocab_size]
                labels.view(-1).to(logits.device) # 将labels展平为[seq_len*batch_size]
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)
        
    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        """合并图像特征和文本嵌入
        
        将图像特征插入到文本嵌入的特定位置，位置由特殊token '<|image_pad|>' 标记
        
        Args:
            image_features (torch.Tensor): 图像特征张量，形状为 (batch_size, num_patches, embed_dim)
            inputs_embeds (torch.Tensor): 文本的词嵌入张量
            input_ids (torch.Tensor): 输入token的id序列
            
        Returns:
            torch.Tensor: 合并后的多模态特征张量
        """
        num_images, num_image_patches, embed_dim = image_features.shape
        
        # 找到输入序列中所有图像填充token的位置，用 '<|image_pad|>' 好像是因为Qwen的tokenizer可以识别到
        # batch_indices: 表示在batch中的索引
        # image_indices: 表示在序列中的位置索引
        # self.tokenizer('<|image_pad|>')['input_ids'][0]: 获取图像填充token的ID （这么写主要是数据格式问题，tokenizer拿到的是dict）
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        
        inputs_embeds[batch_indices, image_indices] = image_features.view(-1, embed_dim)    # 这里把 num_images, num_image_patches 合并为一个维度
        
        return inputs_embeds


class MyDataset(Dataset):
    """自定义数据集类
    用于加载和预处理多模态训练数据
    
    Attributes:
        data_path (str): 数据json文件路径
        images_path (str): 图片文件夹路径
        tokenizer: 文本tokenizer
        processor: 视觉特征处理器
        config: 模型配置
        datas: 加载的训练数据
    """
    def __init__(self, images_path, data_path, tokenizer, processor, config):
        super().__init__()
        self.data_path = data_path
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)   
        
            
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        sample = self.datas[index]
        try:
            image_name = sample['image']
            conversations = sample['conversations']
            
            # 构建输入文本模版（这部分是 用户 提问）
            q_text = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":conversations[0]['value']}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
            
            # answer（这部分在数据中，是 gpt 的回答）
            a_text = conversations[1]['value'] + self.tokenizer.eos_token

            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids   # 拼接list
            
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]  # causal 任务需要做 一个  shift left（即移除最后一个token）
            labels = labels[1:] # labels 也要做 shift right（移除第一个token，一般是[CLS]）
        
            image = Image.open(os.path.join(self.images_path, image_name)).convert("RGB")
            pixel_values = self.processor(text=None, images=image)['pixel_values']

        except:
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text=None, images=default_image)['pixel_values']

            q_text = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":"图片内容是什么\n<image>"}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
            a_text = '图片内容为空' + self.tokenizer.eos_token
            
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids

            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        } 
     

class MyDataCollator:
    """自定义数据整理器
    用于将batch中的样本padding到相同长度
    
    Attributes:
        tokenizer: 用于获取pad_token_id
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])
            
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'pixel_values': torch.cat(pixel_values, dim=0)}
            
        
        
if __name__ == '__main__':
    config = VLMConfig(vision_model_path='/dg_workfs/CNG/yhchen/hpcfs/huggingface_model/google/siglip-base-patch16-224', image_pad_num=49)
    model = VLM(config).cuda()
    print(model)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    images_path = '/dg_workfs/CNG/yhchen/hpcfs/huggingface_datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K'
    data_path = '/dg_workfs/CNG/yhchen/hpcfs/huggingface_datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions/LLaVA-CC3M-Pretrain-595K/chat-translated.json'
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    output_dir = 'save/pretrain' 
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=8,
        learning_rate=1e-4,
        num_train_epochs=5,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=1
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=MyDataset(images_path, data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer)  
    )
    
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('save/pretrain')
    trainer.save_state()
    

## torchrun --nproc_per_node=4 train.py 启动即可
