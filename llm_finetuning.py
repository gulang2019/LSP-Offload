from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GPT2ForSequenceClassification, GPT2LMHeadModel, RobertaForSequenceClassification, BertForSequenceClassification
from transformers import GPT2Tokenizer
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    set_seed,
    BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
)
import evaluate
import torch
import argparse 
import numpy as np
from tqdm import tqdm
import os 
import time 
import pandas as pd
import logging
import random
from torch.utils.data import DataLoader
from lora import APPROX_MAP
from gradient_compressor import LearnedCountSketchCompressor, SVDGradientCompressor, GradientCompressor, GaussianCompressor
from para_offload_modules import replace_linear_layer_with_parallel_linear, Scheduler, ScheduleEnv
from state import State

dtype = torch.bfloat16
device = 0

glue_task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def calculate_perplexity(model, dataset, tokenizer, prob, max_len = None):
    with torch.no_grad():
        nlls = []
        if max_len is None: max_len = model.config.n_positions
        key = 'validation' if 'validation' in dataset else 'test' 
        for data_point in tqdm(dataset[key]):
            if random.random() > prob: continue
            if args.task == 'alpaca' or args.task == 'self_inst':
                c = InstructionTuningDataset if args.task == 'alpaca' else SelfInstructDataset
                encoded =  tokenizer.encode(c.format(data_point), return_tensors='pt')
            elif args.task == 'oasst':
                encoded = tokenizer.encode(data_point['text'], return_tensors='pt')
            elif args.task == 'sst2':
                encoded =  tokenizer.encode(data_point['text'] + data_point['label_text'], return_tensors='pt')
            else: raise NotImplementedError
            seq_len = encoded.shape[1]
            for begin_pos in range(0, seq_len, max_len):
                input_ids = encoded[:, begin_pos:begin_pos + max_len].cuda(device)
                outputs = model(input_ids = input_ids, labels = input_ids)
                loss = outputs.loss
                nlls.append(loss.item())    
        ppl = np.exp(np.mean(nlls))
    return ppl

def calculate_RougeL(model, dataset, tokenizer, prob, max_len = None):
    from ignite.metrics import Rouge
    m = Rouge(multiref='best')
    data_loader = torch.utils.data.DataLoader(dataset['validation'], batch_size = 16, shuffle = False)
    with torch.no_grad():
        if max_len is None: max_len = model.config.n_positions
        for data_point in tqdm(data_loader):
            batch = []
            references = []
            if random.random() > prob: continue
            if args.task == 'alpaca':
                for x in data_point:
                    inst, input, reference = x['instruction'], x['input'], x['output']
                    batch.append(f'instruction: {inst}\n\ninput: {input}\n\noutput: ')
                    references.append(reference)          
                encoded = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            elif args.task == 'self_inst':
                for x in data_point:
                    prompt, reference = data_point['prompt'], data_point['completion']
                    batch.append(f'prompt: {prompt}\n\ncompletion: ')
                    references.append(reference)
                encoded = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            else: raise NotImplementedError
            seq_len = encoded['input_ids'].shape[1]
            if seq_len > max_len:
                continue
            outputs = model.generate(
                input_ids = encoded['input_ids'].cuda(),
                attention_mask = encoded['attention_mask'].cuda(),
                max_new_tokens = max_len - seq_len)
            generated = tokenizer.decode(outputs[0], skip_special_tokens = True)
            m.update((generated, references))
        score = m.compute()['Rouge-L-F']
    return score

def calculate_ppl(model, dataloader, max_len = None, prob = 1.0):
    with torch.no_grad():
        nlls = []
        for inputs in tqdm(dataloader):
            if random.random() > prob: continue
            inputs = {k: v.cuda(device) for k, v in inputs.items() if k != 'idx'}
            if max_len is not None and inputs['input_ids'].shape[1] > max_len:
                continue
            if 'labels' not in inputs:
                inputs['labels'] = inputs['input_ids']
            outputs = model(**inputs)
            loss = outputs.loss
            nlls.append(loss.mean().item())
        ppl = np.exp(np.mean(nlls))
    return ppl

def calculate_glue(model, dataset, task = None):
    assert task is not None
    with torch.no_grad():
        metric = evaluate.load("glue", task)
        model.eval()
        for batch in tqdm(dataset, desc="Evaluating"):
            inputs = {k: v.cuda(device) for k, v in batch.items() if k != 'idx'}
            if 'labels' not in inputs:
                inputs['labels'] = inputs['input_ids']
            with torch.no_grad():
                outputs = model(**inputs)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            metric.add_batch(
                predictions=predictions.cpu().to(batch['labels'].dtype),
                references=batch["labels"],
            )

        eval_metric = metric.compute()
        if len(eval_metric) > 1:
            eval_metric["combined_score"] = np.mean(list(eval_metric.values())).item()
    return eval_metric

class Trainer:
    def __init__(self, model, tokenizer, dataset, optimizer, task, eval_freq, save_dir,
                 profile_freq = None, 
                 profile_dir = 'profile/', 
                 global_state = {}, 
                 max_seqlen = None, 
                 eval_metric = 'ppl', 
                 **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.optimizer = optimizer
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.task = task
        self.eval_freq = eval_freq
        self.best_score = None 
        self.save_dir = save_dir
        self.profile_freq = profile_freq
        self.profile_dir = profile_dir
        self.global_state = global_state
        self.global_state['profile_dir'] = None
        self.max_seqlen = self.model.config.n_positions if max_seqlen is None else max_seqlen
        self.eval_metric = eval_metric
        for k, v in kwargs.items():
            setattr(self, k, v)
        if self.task in glue_task_to_keys:
            self.eval_func = calculate_glue
            self.eval_metric = 'glue'
        elif self.task in ('self_inst', 'alpaca'):
            self.eval_func = calculate_ppl
        elif self.eval_metric == 'ppl':
            self.eval_func = calculate_perplexity
        elif self.eval_metric == 'rougel':
            self.eval_func = calculate_RougeL
        else:
            raise NotImplementedError
    def train_epoch(self, compress_profile: int = None):
        if self.task == 'oasst':
            encodings = self.tokenizer.encode('\n\n'.join([x['text'] for x in self.dataset['train']]), return_tensors='pt')
            tqdm_struct = tqdm(enumerate(range(0, encodings.size(1), self.model.config.n_positions)), total = encodings.size(1) // self.model.config.n_positions)
        elif self.task in glue_task_to_keys or self.task in ('alpaca', 'self_inst'):
            tqdm_struct = tqdm(enumerate(self.train_dataloader), total = len(self.train_dataloader))
        else: raise NotImplementedError
        acc_loss = None
        ave_seq_len = 0
        acc_weight_norm = 0
        for i, data in tqdm_struct:
            if self.early_stop is not None and i >= self.early_stop:
                break
            is_timeout = (time.time() - self.train_start_time) > (self.timeout * 3600)
            if i != 0 and i % self.eval_freq == 0 or is_timeout:
                self.scheduler.clear()
                self.model.eval()
                is_better = False
                eval_start_time = time.time()
                if self.task in ('alpaca', 'self_inst'):
                    score = self.eval_func(self.model, self.eval_dataloader, max_len = self.max_seqlen, prob = self.eval_prob)
                    print(f'epoch {self.epoch}, iter {i}, eval {self.eval_metric}: {score}')
                    is_better = self.best_score is None or score > self.best_score
                    self.best_score = score if is_better else self.best_score
                    self.eval_losses.append((self.epoch, i, time.time() - self.start_time, score))
                if self.task in ('oasst'):
                    score = self.eval_func(self.model, self.dataset, self.tokenizer, prob = 0.1, max_len = self.max_seqlen)
                    print(f'epoch {self.epoch}, iter {i}, eval {self.eval_metric}: {score}')
                    is_better = self.best_score is None or score > self.best_score
                    self.best_score = score if is_better else self.best_score
                    self.eval_losses.append((self.epoch, i, time.time() - self.start_time, score))
                elif self.task in glue_task_to_keys:
                    score_ = self.eval_func(self.model, self.eval_dataloader, task = self.task)
                    score = score_['combined_score'] if len(score_) > 2 else list(score_.values())[0]
                    print(f'epoch {self.epoch}, iter {i}, eval: {score_}')
                    is_better = self.best_score is None or score > self.best_score
                    self.best_score = score if is_better else self.best_score
                    self.eval_losses.append((self.epoch, i, time.time() - self.start_time, score))
                if self.save_dir is not None:
                    if is_better:
                        torch.save(self.model.state_dict(), f'{self.save_dir}/model.pth')
                    pd.DataFrame(self.losses, columns = ['epoch', 'iter', 'time', 'loss']).to_csv(f'{self.save_dir}/losses.csv')
                    pd.DataFrame(self.eval_losses, columns = ['epoch', 'iter', 'time', self.eval_metric]).to_csv(f'{self.save_dir}/eval_losses.csv')
                self.start_time += time.time() - eval_start_time
                torch.cuda.empty_cache()
                if is_timeout:
                    print('exit due to timeout')
                    exit(0)
            self.model.train()
            if self.compress is not None and\
                i % compress_profile == 0 and\
                self.gradient_compressor.need_profile():
                profile_start_t = time.time()
                for module in self.global_state['offload_modules']:
                    module.to('cpu')
                self.global_state['state'] = State.RUNNING_PROFILE
                for data in tqdm(self.dev_dataloader, desc = 'compress profile'):
                    inputs = {k: v.cuda() for k, v in data.items() if k != 'idx'}
                    if 'labels' not in inputs:
                        inputs['labels'] = inputs['input_ids']
                    if inputs['labels'].dtype in (torch.float32, torch.float16, torch.bfloat16):
                        inputs['labels'] = inputs['labels'].to(dtype)
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    loss.backward()
                self.model = self.model.cpu()
                self.gradient_compressor.fit(self.global_state['offload_modules'])
                self.model = self.model.cuda()
                self.global_state['state'] = State.RUNNING
                print(f'Compress profile takes: {time.time() - profile_start_t} s')
            
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            if self.task == 'oasst':
                encoded =  encodings[:, data:data + self.max_seqlen].cuda(device)
                input_ids = encoded
                labels = input_ids
                sequence_length = encoded.shape[1]
                inputs = {'input_ids': input_ids, 'labels': labels}
            elif self.task in glue_task_to_keys or self.task in ('alpaca', 'self_inst'):
                inputs = {k: v.cuda(device) for k, v in data.items() if k != 'idx'}
                sequence_length = inputs['input_ids'].shape[1]
                if 'labels' not in inputs:
                    inputs['labels'] = inputs['input_ids']
                if inputs['labels'].dtype in (torch.float32, torch.float16, torch.bfloat16):
                    inputs['labels'] = inputs['labels'].to(dtype)
            else: raise NotImplementedError
            
            if sequence_length > self.max_seqlen: 
                logging.warning(f'sequence length {sequence_length} is longer than model config n_positions {self.max_seqlen}')
                continue  
            
            outputs = self.model(**inputs)
            l2_regularization = 0
            for param in self.model.parameters():
                l2_regularization += torch.norm(param)
            loss = outputs.loss + self.l2_reg * l2_regularization
            loss.backward()
            
            if self.optimizer is not None:
                self.optimizer.step()
                
            if torch.isnan(loss).any():
                if self.load_ckpt()[0]:
                    continue
                else: 
                    print('nan loss')
                    exit(0)
                
            acc_loss = loss.item() if acc_loss is None else 0.9 * acc_loss + 0.1 * loss.item()
            acc_loss = round(acc_loss, 2)
            acc_weight_norm = l2_regularization.item() if acc_weight_norm is None else 0.9 * acc_weight_norm + 0.1 * l2_regularization.item()
            ave_seq_len = round((ave_seq_len * i + sequence_length) / (i + 1), 2)
            tqdm_struct.desc = f'epoch: {self.epoch}, loss: {acc_loss}, seqlen: {ave_seq_len}, acc_reg: {acc_weight_norm}, memory: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB'
            self.losses.append((self.epoch, i, time.time() - self.start_time, loss.item()))
    def load_ckpt(self):
        if os.path.exists(f'{self.save_dir}/model.pth'):
            self.model.load_state_dict(torch.load(f'{self.save_dir}/model.pth'))
            start_epoch = 0
            if os.path.exists(f'{self.save_dir}/losses.csv'):
                self.losses = pd.read_csv(f'{self.save_dir}/losses.csv', index_col=0).values.tolist()
                if len(self.losses):
                    start_epoch = int(self.losses[-1][0])
            if os.path.exists(f'{self.save_dir}/eval_losses.csv'):
                self.eval_losses = pd.read_csv(f'{self.save_dir}/eval_losses.csv', index_col = 0).values.tolist()
            self.losses = list(filter(lambda x: x[0] < start_epoch, self.losses))
            self.eval_losses = list(filter(lambda x: x[0] < start_epoch, self.eval_losses))
            print(f'model loaded from {self.save_dir}/model.pth, epoch {start_epoch}, len(losses): {len(self.losses)}, len(eval_losses): {len(self.eval_losses)}')
            return True, start_epoch
        print(f'no checkpoint found in {self.save_dir}')
        return False, 0
    def train(self,
             epochs = 1,
             from_ckpt = False,
             early_stop = None,
             compress_profile: int = None
             ):
        start_epoch = 0
        self.losses = []
        self.eval_losses = []
        if from_ckpt:
            loaded, start_epoch = self.load_ckpt()
        self.model.train()
        self.start_time = time.time()
        self.early_stop = early_stop
        self.train_start_time = time.time()
        for epoch in range(start_epoch, epochs):
            self.epoch = epoch
            with ScheduleEnv(self.scheduler):
                self.train_epoch(compress_profile)

    def offload_profile(self, offload_device, max_profile_bs, n_repeat = 10):
        self.model.train()
        if self.compress is not None:
            if self.gradient_compressor.need_profile():
                start = time.perf_counter()
                self.global_state['state'] = State.RUNNING_PROFILE
                for data in tqdm(self.dev_dataloader, desc = 'compress profile'):
                    inputs = {k: v.cuda() for k, v in data.items() if k != 'idx'}
                    if 'labels' not in inputs:
                        inputs['labels'] = inputs['input_ids']
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    loss.backward()
                    break
                iter_time = time.perf_counter() - start
            else: iter_time = 0
            start = time.perf_counter()
            self.model = self.model.cpu()
            self.gradient_compressor.fit(self.global_state['offload_modules'])
            self.model = self.model.cuda()
            self.global_state['state'] = State.RUNNING
            profile_time = len(self.dev_dataloader) * iter_time + time.perf_counter() - start
            print(f'profile time: {profile_time:.2f} s')

        encoded = torch.randint(0, self.tokenizer.vocab_size, (self.batch_size, self.max_seqlen), device = 'cuda')
        labels = torch.randint(self.model.num_labels, (self.batch_size,), device = 'cuda') if self.task in glue_task_to_keys else encoded
        
        end2end_times = {}
        fwd_times = []
        for state, tag in zip([State.PROFILE_COMM, State.PROFILE, State.PROFILE_COMPUTE], ['Comm', 'Tot', 'GPU_Compute']):
        # for state, tag in zip([State.PROFILE_COMM, State.PROFILE, State.PROFILE_COMPUTE], ['Comm', 'Tot']):
        # for state, tag in zip([State.PROFILE_COMM], ['Comm']):
            if state == State.PROFILE_COMM: self.gradient_compressor.init_profile()
            global_state['state'] = state
            t = time.perf_counter()
            with ScheduleEnv(self.scheduler):
                for i in tqdm(range(n_repeat), desc = tag):
                    if state == State.PROFILE_COMPUTE:
                        torch.cuda.synchronize()
                        fwd_start_t = time.perf_counter()
                    y = model(encoded, labels = labels)
                    err = y.loss
                    if state == State.PROFILE_COMPUTE: 
                        torch.cuda.synchronize()
                        fwd_times.append(time.perf_counter() - fwd_start_t)
                    err.backward()
            torch.cuda.synchronize()
            end2end_times[tag] = (time.perf_counter() - t) / n_repeat
        global_state['state'] = State.RUNNING
        self.gradient_compressor.report_and_end_profile(n_repeat)
        # profile adam
        example_params = {}
        for param in model.parameters():
            if param.requires_grad:
                if param.size() not in example_params:
                    example_params[param.size()] = [param, 1]
                else:
                    example_params[param.size()] = [param, example_params[param.size()][1] + 1]
        optim_time = 0
        for k, v in example_params.items():
            t = time.perf_counter()
            for _ in range(n_repeat):
                optim = torch.optim.Adam([v[0]])
                optim.zero_grad()
                optim.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t
            optim_time += elapsed / n_repeat * v[1]
        
        # {'upload_amount': upload_amount, 'upload_time': upload_time, 'offload_amount': offload_amount, 'offload_time': offload_time, 'upload_wait': np.mean(upload_wait), 'offload_wait': np.mean(offload_wait), 'compute_time': np.mean(compute_time)}
        stats = self.scheduler.report()
        upload_bw = stats['upload_amount'] / stats['upload_time'] / 1e3
        offload_bw = stats['offload_amount'] / stats['offload_time'] / 1e3
        reference_time = end2end_times['GPU_Compute'] + optim_time
        slow_down = (end2end_times['Tot']) / reference_time
        print(f'optim time: {optim_time:.2f}')
        print(f'upload_bw: {upload_bw:.2f}GB/s')
        print(f'offload_bw: {offload_bw:.2f}GB/s')
        print(f'reference_time: {reference_time:.2f}')
        print(f'slow_down: {slow_down:.2f}')
        print(f'fwd time: {np.mean(fwd_times):.2f}')
        print(f'bwd time: {end2end_times["GPU_Compute"] - np.mean(fwd_times):.2f}')
        for k, v in end2end_times.items():
            print(k, v)
        print(torch.cuda.memory_summary(device = 'cuda:0'))
        return
        
        # profile offload:
        self.global_state['ProfileState'] = 'Offload'
        offload_times, optim_times, upload_times, offload_comms, upload_comms, num_params, offload_bandwidth, upload_bandwidth, compute_speed = [], [], [], [], [], [], [], [], []
        detailed_offload_times = {}
        detailed_optim_times = {}
        detailed_upload_times = {}
        
        for _ in range(n_repeat):
            self.global_state['OffloadTime'] = self.global_state['OptimTime'] = self.global_state['UploadTime'] = self.global_state['UploadComm'] = self.global_state['OffloadComm'] = self.global_state['NumParams'] = 0.0 
            self.global_state['OffloadTimeStamp'] = {}
            self.global_state['OptimTimeStamp'] = {}
            self.global_state['UploadTimeStamp'] = {}
            outputs = self.model(input_ids = encoded, labels = labels)
            l = outputs.loss
            l.backward() 
            offload_times.append(self.global_state['OffloadTime'])
            optim_times.append(self.global_state['OptimTime'])
            upload_times.append(self.global_state['UploadTime'])
            offload_comms.append(self.global_state['OffloadComm'] / 1e6)
            upload_comms.append(self.global_state['UploadComm'] / 1e6)
            num_params.append(self.global_state['NumParams'] / 1e6)
            offload_bandwidth.append(self.global_state['OffloadComm'] / (self.global_state['OffloadTime'] + 1e-6) / 1e9)
            upload_bandwidth.append(self.global_state['UploadComm'] / (self.global_state['UploadTime'] + 1e-6) / 1e9)
            compute_speed.append(self.global_state['NumParams'] / (self.global_state['OptimTime'] + 1e-6) / 1e9)
            for k, v in self.global_state['OffloadTimeStamp'].items():
                detailed_offload_times[k] = detailed_offload_times.get(k, []) + [v]
            for k, v in self.global_state['OptimTimeStamp'].items():
                detailed_optim_times[k] = detailed_optim_times.get(k, []) + [v]
            for k, v in self.global_state['UploadTimeStamp'].items():
                detailed_upload_times[k] = detailed_upload_times.get(k, []) + [v]

        batch_sizes = list(range(1,max_profile_bs+1))
        # profile compute:
        self.global_state['ProfileState'] = 'Compute'
        compute_profiles = []
        for bs in tqdm(batch_sizes, desc = 'benchmark Compute'):
            fake_inputs = torch.randint(0, self.tokenizer.vocab_size, (bs, self.max_seqlen), device = 'cuda')
            fake_labels = torch.randint(self.model.num_labels, (bs,), device = 'cuda') if self.task in glue_task_to_keys else fake_inputs
            compute_times = []
            fwd_times = []
            time_stamps = []
            torch.cuda.synchronize()
            for _ in range(n_repeat):
                self.global_state['FWDTimeStamp'] = []
                self.global_state['BWDTimeStamp'] = []
                start_time = time.time()
                outputs = self.model(input_ids = fake_inputs, labels = fake_labels)
                torch.cuda.synchronize()
                fwd_times.append(time.time() - start_time)
                l = outputs.loss
                l.backward()
                torch.cuda.synchronize()
                compute_times.append(time.time() - start_time)

                # Because of gradient checkpointing, the forward and backward time stamps may not match
                # self.global_state['FWDTimeStamp'] = self.global_state['FWDTimeStamp'][:len(self.global_state['BWDTimeStamp'])]
                # if time_stamps == []:
                #     for tag, fwd_t in self.global_state['FWDTimeStamp']:
                #         time_stamps.append((tag, []))

                # t0 = self.global_state['FWDTimeStamp'][0][1]
                # for i, ((tag, fwd_t), (bwd_tag, bwd_t)) in enumerate(zip(self.global_state['FWDTimeStamp'],\
                #     reversed(self.global_state['BWDTimeStamp']))):
                #     assert time_stamps[i][0] == tag and tag == bwd_tag
                #     time_stamps[i][1].append([fwd_t - t0, bwd_t - t0])
            compute_profiles.append([bs, np.mean(compute_times), np.mean(fwd_times), time_stamps])
        
        self.global_state['ProfileState'] = None
        tot_times = []
        for bs in tqdm(batch_sizes, desc = 'benchmark Total'):
            fake_inputs = torch.randint(0, self.tokenizer.vocab_size, (bs, self.max_seqlen), device = 'cuda')
            fake_labels = torch.randint(self.model.num_labels, (bs,), device = 'cuda') if self.task in glue_task_to_keys else fake_inputs
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(n_repeat):
                outputs = self.model(input_ids = fake_inputs, labels = fake_labels)
                l = outputs.loss
                l.backward()
            torch.cuda.synchronize()
            tot_times.append((bs, (time.time() - start_time) / n_repeat))
        
        print(f'Profile (saved to {self.profile_dir}/profile.csv)')


        comm_compute_ratio = []
        upload_time = np.mean(upload_times)
        offload_time = np.mean(offload_times)
        upload_time_std = np.std(upload_times)
        offload_time_std = np.mean(offload_times)
        
        for (bs, compute_time, fwd_time, time_stamps), (bs_, tot_time) in zip(compute_profiles, tot_times):
            assert bs == bs_
            ts_data = []
            for tag, times in time_stamps:
                times = np.array(times)
                ts_data.append([tag, np.mean(times[:, 0]), np.mean(times[:, 1]), np.mean(detailed_offload_times[tag]), np.mean(detailed_optim_times[tag]), np.mean(detailed_upload_times[tag])])
            df_ts = pd.DataFrame(ts_data, columns = ['tag', 'fwd', 'bwd', 'offload', 'optim', 'upload'])
            df_ts.to_csv(f'{self.profile_dir}/time_stamps_{bs}.csv')
            
            print(f'bs: {bs}, compute: {compute_time:.6f}, tot/compute: {tot_time / compute_time}, fwd: {fwd_time:.6f}, bwd: {compute_time - fwd_time:.6f}, upload/compute: {upload_time / compute_time:.6f}, offload/compute: {offload_time / compute_time:.6f}')
            comm_compute_ratio.append([bs, tot_time / compute_time, upload_time / compute_time, offload_time / compute_time, upload_time_std / compute_time, offload_time_std / compute_time])
        df = pd.DataFrame(comm_compute_ratio, columns = ['bs', 'tot/compute', 'upload/compute', 'offload/compute', 'upload_std/compute', 'offload_std/compute'])
        df.to_csv(f'{self.profile_dir}/comm_compute_ratio.csv')            
        
        for name, times in zip(['offload (s)', 'optim (s)', 'upload (s)', '#upload (MB)', '#offload (MB)', '#Params (M)', 'BW upload (GB/s)', 'BW offload (GB/s)', 'Compute (G/s)']\
            , [offload_times, optim_times, upload_times, upload_comms, offload_comms, num_params, upload_bandwidth, offload_bandwidth, compute_speed]):
            print(f'{name}: {np.mean(times):.6f} +/- {np.std(times):.6f}')
        for bs, compute_time in tot_times:
            print(f'bs: {bs}, tot(s): {compute_time:.6f}')

        df = pd.DataFrame({'offload': offload_times, 'optim': optim_times, 'upload': upload_times, 'compute': compute_times})
        df.to_csv(f'{self.profile_dir}/profile.csv')

    def init_compress(self,
                    compress,
                    compress_profile,
                    compress_profile_data,
                    compress_svd_energy_thresh,
                    compress_svd_min_rank,
                    compress_svd_max_rank,
                    compress_svd_approx,
                    compress_cs_size,
                    compress_cs_n_unempty,
                    compress_cs_n_iter,
                    compress_cs_lr,
                    compress_cs_n_ft_iter,
                    compress_cs_ft_lr,
                    compress_cs_reuse,
                    compress_cs_init,
                    compress_cs_thresh,
                    compress_gaussian_rank,
                    compress_gaussian_freq):
        self.compress = compress
        self.compress_profile = compress_profile
        self.compress_profile_data = compress_profile_data
        if compress == 'CS':
            self.gradient_compressor = LearnedCountSketchCompressor(compress_cs_size, compress_cs_n_unempty, compress_cs_n_iter, compress_cs_n_ft_iter, compress_cs_lr, compress_cs_ft_lr, compress_cs_reuse, compress_cs_init, compress_cs_thresh)
        elif compress == 'SVD':
            self.gradient_compressor = SVDGradientCompressor(compress_svd_energy_thresh, compress_svd_min_rank, compress_svd_max_rank, compress_svd_approx)
        elif compress == 'GAUSSIAN':
            self.gradient_compressor = GaussianCompressor(compress_gaussian_rank, compress_gaussian_freq)
        else:
            self.gradient_compressor = GradientCompressor()
            self.gradient_compressor.fit(self.global_state['offload_modules'])
        for module in self.global_state['offload_modules']:
            module.gradient_compressor = self.gradient_compressor
        print('Compressor:', self.gradient_compressor)

    def init_scheduler(self, **kwargs):
        self.scheduler = Scheduler(self.global_state['offload_modules'], **kwargs)
        print(self.scheduler)

def layer_profile(model, bs, seq_len, n_repeat, dtype):
        if isinstance(model, GPT2LMHeadModel):
            _layer = model.transformer.h[0]
        elif isinstance(model, RobertaForSequenceClassification):
            _layer = model.roberta.encoder.layer[0]
        elif isinstance(model, BertForSequenceClassification):
            _layer = model.bert.encoder.layer[0]
        elif isinstance(model, LlamaForCausalLM):
            _layer = model.model.layers[0]
        else: raise NotImplementedError
        print(f'profile layer {_layer}')
        print(f'bs: {bs}, seq_len: {seq_len}, n_repeat: {n_repeat}, dtype: {dtype}')
        model = model.cpu()
        for device in ['cuda', 'cpu']:
            if device == 'cuda':
                n_repeat *= 10
            layer = _layer.to(device)
            # if device == 'cpu': 
            #     layer = layer.float()
            #     dtype = torch.float32
            fake_input = torch.randn((bs, seq_len, model.config.hidden_size), device = device, dtype = dtype)
            fake_attns = torch.ones((bs, seq_len), device = device, dtype = torch.int32)
            fake_positions = torch.arange(seq_len).expand(bs, seq_len).to(device)
            optim = torch.optim.Adam(layer.parameters(), lr = 1e-3)
            for i in tqdm(range(5), desc = f'warm up {device}'):
                output = layer(fake_input, None, fake_positions, output_attentions = False, use_cache = False)
                loss = output[0].mean()
                loss.backward()
            t = time.perf_counter()
            fwd_times = []
            for i in tqdm(range(n_repeat), desc = f'layer profile FWD BWD {device}'):
                if device == 'cuda': torch.cuda.synchronize()
                fwd_start_t = time.perf_counter()
                output = layer(fake_input, None, fake_positions, output_attentions = False, use_cache = False)
                loss = output[0].mean()
                if device == 'cuda': torch.cuda.synchronize()
                fwd_times.append(time.perf_counter() - fwd_start_t)
                loss.backward()
            if device == 'cuda': torch.cuda.synchronize()
            elapsed = time.perf_counter() - t
            optim_times = []
            for _ in tqdm(range(n_repeat), desc = f'layer profile optim {device}'):
                output = layer(fake_input, None, fake_positions, output_attentions = False, use_cache = False)
                loss = output[0].mean()
                optim.zero_grad()
                loss.backward()
                if device == 'cuda': torch.cuda.synchronize()
                t = time.perf_counter()
                optim.step()
                if device == 'cuda': torch.cuda.synchronize()
                optim_times.append(time.perf_counter() - t)
            optim_time = np.mean(optim_times)
            fwd_time = np.mean(fwd_times)
            bwd_time = elapsed / n_repeat - fwd_time
            tot_time = elapsed / n_repeat
            print(f'device: {device}, dtype: {dtype}, fwd: {fwd_time:.4f}, bwd: {bwd_time:.4f}, optim_time: {optim_time:.4f}, tot_time: {tot_time:.4f}')
        
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='llama2', choices = ['llama2', 'gpt2', 'gpt2-medium', 'gpt2-large',\
        'gpt2-xl', "bert-base-uncased", 'roberta-base', 'bert-xl', 'llama-3b', 'roberta-large', 'bert-large-uncased', 'deepseek-ai/deepseek-coder-1.3b-base'], help = 'the model to be pruned')
    parser.add_argument('--task', choices = ['self_inst', 'alpaca', 'wikitext', 'oasst', 'sqlcode'] + list(glue_task_to_keys.keys()))
    parser.add_argument('--peft', choices = list(APPROX_MAP.keys()) + ['lora', 'drop'], default = None)
    parser.add_argument('--use_deepspeed', action = 'store_true')
    parser.add_argument('--rank', type=int, default = 32)
    parser.add_argument('--optim-approx', choices = ['lora_approx', 'lora_svd', 'power', 'lora_gaussian', 'topk', 'CountSketch'], default = None)
    parser.add_argument('--approx-rank', type = int, default = 32)
    parser.add_argument('--modules', choices = ['attn.qkv', 'attn.o', 'mlp.gate', 'mlp.up', 'mlp.down'], default = [], nargs = '+')
    parser.add_argument('--stop', type = int, default = None)
    parser.add_argument('--verbose', action = 'store_true')
    parser.add_argument('--save', action = 'store_true')
    parser.add_argument('--output', type = str, default = None)
    parser.add_argument('--profile', type = int, default = None)
    parser.add_argument('--n_epoch', type=int, default=1)
    parser.add_argument('--eval_freq', type=int, default=1000)
    parser.add_argument('--offload', type = str, choices = ['cpu', 'cuda:0', 'cuda:1'], default = None)
    parser.add_argument('--lr', type = float, default = 1e-4)
    parser.add_argument('--max_seqlen', type = int, default = None)
    parser.add_argument('--offload-profile', action = 'store_true')
    parser.add_argument('--layer-profile', action = 'store_true')
    parser.add_argument('--max-profile-bs', type = int, default = 4)
    parser.add_argument('--memory-cap', type = float, default = .0)
    parser.add_argument('--dtype', choices = ['float32', 'float16', 'bfloat16'], default = 'bfloat16')
    parser.add_argument('--eval-metric', choices = ['ppl', 'rougel'], default = 'ppl')
    parser.add_argument('--batch-size', type = int, default = 8)
    parser.add_argument('--optim-dtype', choices=['float32', 'bfloat16', 'float16'], default = 'bfloat16')
    parser.add_argument('--gradient-checkpointing', action = 'store_true')
    parser.add_argument('--cpu-threads', type=int, default=14)
    parser.add_argument('--cpu-inter-threads', type=int, default = 2)
    parser.add_argument('--profile-repeat', type=int, default = 10)
    parser.add_argument('--keep-layers', type=int, default = None)
    parser.add_argument('--scheduler-zero', action = 'store_true')
    parser.add_argument('--sch-fcfs-point', type=int, default=0)
    parser.add_argument('--sch-fcfs-process-delay', type=int, default=0)
    parser.add_argument('--sch-fcfs-h2d-delay', type=int, default=0)
    parser.add_argument('--sch-lcfs-h2d-delay', type=int, default=0)
    parser.add_argument('--sch-lcfs-process-delay', type=int, default=0)
    parser.add_argument('--sch-lcfs-d2h-delay', type=int, default=0)
    parser.add_argument('--seed', type=int, default = None)
    parser.add_argument('--eval-prob', type=float, default = 1.0)
    parser.add_argument('--from-ckpt', action = 'store_true')
    parser.add_argument('--timeout', type=int, default = 1000)
    parser.add_argument('--l2-reg', type=float, default = 0.0)
    
    parser.add_argument('--compress', choices = ['CS', 'SVD', 'GAUSSIAN'])
    parser.add_argument('--compress-profile', type = int, default = None)
    parser.add_argument('--compress-profile-data', type = int, default=0)
    parser.add_argument('--compress-svd-energy-thresh', type=float, default = 0.99)
    parser.add_argument('--compress-svd-min-rank', type = int, default=32)
    parser.add_argument('--compress-svd-max-rank', type = int, default=256)
    parser.add_argument('--compress-svd-approx', action = 'store_true')
    parser.add_argument('--compress-cs-size', type = int, default = None)
    parser.add_argument('--compress-cs-n_unempty', type = int, default = None)
    parser.add_argument('--compress-cs-n_iter', type = int, default = None)
    parser.add_argument('--compress-cs-lr', type = float, default = 1e-1)
    parser.add_argument('--compress-cs-n_ft_iter', type = int, default = None)
    parser.add_argument('--compress-cs-ft_lr', type = float, default = 1e-1)
    parser.add_argument('--compress-cs-reuse', action = 'store_true')
    parser.add_argument('--compress-cs-thresh', type = float, default = 0.8)
    parser.add_argument('--compress-cs-init', type= str, choices = ['gaussian', 'binary'], default = 'gaussian')
    parser.add_argument('--compress-gaussian-rank', type = int, default = 32)
    parser.add_argument('--compress-gaussian-freq', type = int, default = 10)
    
    args  = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(args.cpu_inter_threads)
    if args.dtype == 'float32':
        dtype = torch.float32
    elif args.dtype == 'float16':
        dtype = torch.float16
    elif args.dtype == 'bfloat16':
        dtype = torch.bfloat16
    
    if args.optim_dtype == 'float32':
        optim_dtype = torch.float32
    elif args.optim_dtype == 'float16':
        optim_dtype = torch.float16
    elif args.optim_dtype == 'bfloat16':
        optim_dtype = torch.bfloat16
        
    if args.seed is not None:
        torch.manual_seed(args.seed)

    tag = f'{args.model}-{args.task}'
    tag += f'-dtype-{args.dtype}-lr-{args.lr}'
    if args.peft is not None:
        tag += f'-peft-{args.peft}-{args.rank}'
    if args.optim_approx is not None:
        tag += f'-opt-{args.optim_approx}-{args.approx_rank}'
    if args.offload is not None:
        tag += f'-offload-{args.offload.replace(":", "_")}'
    if args.compress is not None:
        tag += f'-compress-{args.compress}'
        if args.compress == 'SVD':
            tag += f'-svd-{args.compress_svd_energy_thresh}-{args.compress_svd_min_rank}-{args.compress_svd_max_rank}'
        elif args.compress == 'CS':
            tag += f'-cs-{args.compress_cs_size}-{args.compress_cs_n_unempty}-{args.compress_cs_n_iter}-{args.compress_cs_n_ft_iter}-{args.compress_cs_init}-{args.compress_cs_lr}-{args.compress_cs_ft_lr}'
        elif args.compress == 'GAUSSIAN':
            tag += f'-gaussian-{args.compress_gaussian_rank}-{args.compress_gaussian_freq}'
    ### Load the Model
    use_fast = True
    if args.model == 'llama2':
        model_dir = 'meta-llama/Llama-2-7b-hf'
    elif args.model == 'llama-3b':
        model_dir = 'openlm-research/open_llama_3b'
        use_fast = False
        # model_dir = 'meta-llama/Llama-2-7b-hf'
    else: model_dir = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=use_fast)
    max_seq_length = min(args.max_seqlen, tokenizer.model_max_length)
    
    profile_dir = None
    if args.profile is not None or args.offload_profile:
        profile_dir = f'profile/{tag}/'
        os.makedirs(profile_dir, exist_ok = True)
        
    save_dir = f'models/{tag}/' if args.output is None else f'{args.output}'
    os.makedirs(save_dir, exist_ok = True)
    print('save_dir:', save_dir)
    
    # Load dataset 
    from datasets import load_dataset
    train_dataloader = eval_dataloader = None
    extra_kwargs = {}
    if args.task in glue_task_to_keys:
        dataset = load_dataset(
            "nyu-mll/glue",
            args.task
        )
        is_regression = args.task == "stsb"
        if not is_regression:
            label_list = dataset["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
            
        extra_kwargs['num_labels'] = num_labels
            
        print('num_labels:', num_labels)
    elif args.task in ('alpaca', 'self_inst'):
        from data import InstructionTuningDataset, SelfInstructDataset
        data_class = InstructionTuningDataset if args.task == 'alpaca' else SelfInstructDataset
        def preprocess_function(data_point):
            # Tokenize the texts
            args = data_class.format(data_point)
            result = tokenizer(args, padding=False, max_length=max_seq_length, truncation=True)
            return result
        dataset = data_class(tokenizer.__class__.__name__)
        dataset.map(preprocess_function, use_cache = True, load_from_cache = True)
        train_size = int(len(dataset) * 0.9)
        val_size = args.compress_profile_data
        test_size = len(dataset) - train_size - val_size
        assert train_size > 0
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        dev_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
        eval_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, train_size + val_size + test_size))
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, padding="longest") # default_data_collator #
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size
        )
        dev_dataloader = DataLoader(dev_dataset, collate_fn=data_collator, batch_size=args.batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.batch_size)
    elif args.task == 'oasst':
        dataset = load_dataset('timdettmers/openassistant-guanaco')
    elif args.task == 'mrpc':
        dataset = load_dataset('nyu-mll/glue', 'mrpc')
    elif args.task == 'sqlcode':
        dataset = load_dataset('b-mc2/sql-create-context', split="train")
        
    config = AutoConfig.from_pretrained(
        model_dir,
        finetuning_task=args.task,
        use_cache = False,
        **extra_kwargs
        # _attn_implementation = 'flash_attention_2'
    )
    
    if args.task in glue_task_to_keys:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config, torch_dtype=dtype).cuda(device)
        sentence1_key, sentence2_key = glue_task_to_keys[args.task]
        label_to_id = None
        if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and not is_regression
        ):
            # Some have all caps in their config, some don't.
            label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
            if sorted(label_name_to_id.keys()) == sorted(label_list):
                label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
            else:
                logging.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                    "\nIgnoring the model labels as a result.",
                )
        elif args.task is None and not is_regression:
            label_to_id = {v: i for i, v in enumerate(label_list)}
        if label_to_id is not None:
            model.config.label2id = label_to_id
            model.config.id2label = {id: label for label, id in config.label2id.items()}
        elif args.task is not None and not is_regression:
            model.config.label2id = {l: i for i, l in enumerate(label_list)}
            model.config.id2label = {id: label for label, id in config.label2id.items()}
        
        def preprocess_function(examples):
            
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, padding=False, max_length=max_seq_length, truncation=True)
            
            # Map labels to IDs (not necessary for GLUE tasks)
            if label_to_id is not None and "label" in examples:
                result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
            # print('result', result)
            # exit(0)
            return result
        
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
        dataset = dataset.remove_columns([k for k in dataset['train'][0].keys() if k not in ['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'label']])
                
        train_dataset = dataset["train"]
        assert args.compress_profile_data < len(train_dataset)
        dev_dataset = torch.utils.data.Subset(train_dataset, range(args.compress_profile_data))
        train_dataset = torch.utils.data.Subset(train_dataset, range(args.compress_profile_data, len(train_dataset)))
        eval_dataset = dataset["validation_matched" if args.task == "mnli" else "validation"]
        # if args.dtype in ('float16', 'bfloat16'):
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, padding="longest") # default_data_collator #
        # else: 
            # data_collator = None
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size
        )
        dev_dataloader = DataLoader(dev_dataset, collate_fn=data_collator, batch_size=args.batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.batch_size * 4)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, config=config).cuda(device)
    
    if args.layer_profile:
        layer_profile(model, args.batch_size, max_seq_length, args.profile_repeat, dtype)
        exit(0)
    
    if args.keep_layers is not None:
        if isinstance(model, GPT2LMHeadModel):
            model.transformer.h.__delitem__(slice(args.keep_layers, None))
        elif isinstance(model, RobertaForSequenceClassification):
            model.roberta.encoder.layer.__delitem__(slice(args.keep_layers, None))
        elif isinstance(model, BertForSequenceClassification):
            model.bert.encoder.layer.__delitem__(slice(args.keep_layers, None))
        elif isinstance(model, LlamaForCausalLM):
            model.model.layers.__delitem__(slice(args.keep_layers, None))
        else: raise NotImplementedError    
        
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else: 
        logging.warning('no eos token found, using <pad> as pad token')
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))

    if len(args.modules) and (args.offload is not None) or args.peft == 'lora':
        for name, param in model.named_parameters():
            if 'classifier' not in name: 
                param.requires_grad = False
              
    if train_dataloader is not None:
        for batch in train_dataloader:
            print('train data example:', batch)
            break  # Print only the first batch for brevity
        print('#train:', len(train_dataloader))
        print('#dev:', len(dev_dataloader))
        print('#eval:', len(eval_dataloader))
    else:
        print('train data example:', dataset['train'][0])
        if 'validation' in dataset: print('val data example', dataset['validation'][0])
        if 'validation_matched' in dataset: print('val data example', dataset['validation_matched'][0])
        if 'test' in dataset: print('test data example', dataset['test'][0])
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    global_state = args.__dict__
    global_state['state'] = State.RUNNING
    print('peft:', args.peft, 'rank:', args.rank)
    modules = []
    if args.peft == 'lora' or (args.offload is not None):
        modules = replace_linear_layer_with_parallel_linear(
            model = model,
            modules = args.modules,
            use_lora = args.peft == 'lora',
            lora_rank = args.rank,
            int_compress_method=args.peft,
            int_compress_rank=args.rank,
            offload = args.offload,
            lr = args.lr,
            optim_dtype = optim_dtype,
            use_grad_ckpt=args.gradient_checkpointing,
            global_state = global_state
        )
        print(f'{len(modules)} modules replaced')
    global_state['offload_modules'] = modules
    
    print('model:', model)
    print(f'#params: {sum(p.numel() for p in model.parameters()) / 1e9}G:.2f, {sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9:.2f}GB')
    print(f'Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9}G:.2f, {sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / 1e9:.2f}GB')
    
    if args.model == 'llama2' or args.model == 'llama-3b':
        model.config.n_positions = model.config.max_position_embeddings
        model.config.n_layer = model.config.num_hidden_layers

    print (f'train {args.modules} of {args.model} on {args.task} with precision {dtype} optim: {optim_dtype}')
    if args.peft: 
        print('peft:', args.peft, 'rank:', args.rank)
    if args.optim_approx:
        print('optim_approx:', args.optim_approx, 'approx_rank:', args.approx_rank)
    if args.offload:
        print('offload:', args.offload)
    print('gradient_checkpointing:', args.gradient_checkpointing)

    if args.compress is not None:
        print(f'Compress Method: {args.compress}')
        print(f'Compress per {args.compress_profile} iterations, {args.compress_profile_data} data')
        assert args.compress_profile_data > 0
        if args.compress == 'CS':
            init_method = 'Binary' if args.compress_cs_init else 'Guassian'
            print(f'CS size: {args.compress_cs_size}, n_unempty: {args.compress_cs_n_unempty}, n_iter: {args.compress_cs_n_iter}, n_ft_iter: {args.compress_cs_n_ft_iter} lr: {args.compress_cs_lr}, ft_lr: {args.compress_cs_ft_lr}, reuse: {args.compress_cs_reuse}, init: {args.compress_cs_init}')
        elif args.compress == 'SVD':
            print(f'SVD energy threshold: {args.compress_svd_energy_thresh}, min rank: {args.compress_svd_min_rank}, max rank: {args.compress_svd_max_rank}, approx: {args.compress_svd_approx}, ')
        elif args.compress == 'GAUSSIAN':
            print(f'Gaussian rank: {args.compress_gaussian_rank}, freq: {args.compress_gaussian_freq}')
    optimizer = None
    if args.offload is None:
        print(f'train: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}G:.2f, {sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / 1e9:.2f}GB')
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, eps = 1e-6)
    else:
        classifier_params = [param for name, param in model.named_parameters() if 'classifier' in name]
        if len(classifier_params):
            optimizer = torch.optim.Adam(classifier_params, lr = args.lr, eps = 1e-6)

    trainer = Trainer(model, tokenizer, dataset, optimizer, args.task\
        , args.eval_freq, save_dir, args.profile, profile_dir, global_state, args.max_seqlen, args.eval_metric, 
        batch_size = args.batch_size, train_dataloader = train_dataloader, eval_dataloader = eval_dataloader,
        dev_dataloader = dev_dataloader, eval_prob = args.eval_prob, timeout = args.timeout, dtype = dtype,
        l2_reg = args.l2_reg)
    trainer.init_compress(args.compress,
                         args.compress_profile,
                         args.compress_profile_data,
                         args.compress_svd_energy_thresh,
                         args.compress_svd_min_rank,
                         args.compress_svd_max_rank,
                         args.compress_svd_approx,
                         args.compress_cs_size,
                         args.compress_cs_n_unempty,
                         args.compress_cs_n_iter,
                         args.compress_cs_lr,
                         args.compress_cs_n_ft_iter,
                         args.compress_cs_ft_lr,
                         args.compress_cs_reuse,
                         args.compress_cs_init,
                         args.compress_cs_thresh,
                         args.compress_gaussian_rank,
                         args.compress_gaussian_freq)
    trainer.init_scheduler( fcfs_point = args.sch_fcfs_point,
                            fcfs_process_delay = args.sch_fcfs_process_delay,
                            fcfs_h2d_delay = args.sch_fcfs_h2d_delay,
                            lcfs_h2d_delay = args.sch_lcfs_h2d_delay,
                            lcfs_process_delay = args.sch_lcfs_process_delay,
                            lcfs_d2h_delay = args.sch_lcfs_d2h_delay,
                            zero = args.scheduler_zero,
                            verbose = args.verbose)
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    if args.offload_profile:
        trainer.offload_profile(args.offload, args.max_profile_bs, n_repeat = args.profile_repeat)
    else:
        trainer.train(args.n_epoch, from_ckpt = args.from_ckpt, early_stop = args.stop, compress_profile = args.compress_profile)

'''
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --capture-range-end stop --cudabacktrace=true -x true -o my_profile python main.py

nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none --capture-range-end stop --capture-range=cudaProfilerApi --cudabacktrace=true -x true poetry run
'''