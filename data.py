import json 
from torch.utils.data import Dataset
import tqdm
import diskcache as dc

CACHE = dc.Cache('datasets/cache')

class DatasetWithMap(Dataset):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag
        
    def map(self, func, desc = 'Mapping', use_cache = True, load_from_cache = False):
        if use_cache:
            if load_from_cache and self.tag in CACHE:
                self.data = CACHE[self.tag]
            else:
                self.data = [func(x) for x in tqdm.tqdm(self.data, desc = desc)]
                CACHE[self.tag] = self.data
            return
        self.data = [func(x) for x in tqdm.tqdm(self.data, desc = desc)]

class InstructionTuningDataset(DatasetWithMap):
    @staticmethod
    def format(data_point):
        return f"instruction: {data_point['instruction']}\n\ninput: {data_point['input']}\n\noutput: {data_point['output']}"    
    def __init__(self, tag):
        super().__init__(f"{self.__class__.__name__}+{tag}")
        with open('datasets/alpaca_gpt4_data.json') as f:
            self.data = [x for x in json.load(f)]
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
class SelfInstructDataset(DatasetWithMap):
    @staticmethod 
    def format(data_point):
        return f"prompt: {data_point['prompt']}\n\ncompletion: {data_point['completion']}"
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.data = []
        with open('datasets/gpt3_finetuning_data.jsonl', 'r') as f:
            for line in f:
                # Convert line to JSON
                json_obj = json.loads(line)
                self.data.append(json_obj)
        # self.data = [SelfInstructDataset.format(x) for x in self.data]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
    def example_data(self):
        return '{"prompt": "Make a list of 10 ways to help students improve their study skills.\n\nOutput:", "completion": " 1. Make a schedule for studying and stick to it.\n2. Study in the same place every time.\n3. Set goals for yourself.\n4. Take breaks when you need them.\n5. Don\'t cram before an exam.\n6. Get enough sleep.\n7. Eat healthy food.\n8. Exercise regularly.\n9. Find a study partner.\n10. Reward yourself after completing a task.<|endoftext|>"}'

_formatting = {    
    'alpaca': InstructionTuningDataset.format,
    'self_inst': SelfInstructDataset.format
}

# class DataCollatorWithPaddingAndFormatting:    
#     def __init__(self, 
#                  tokenizer, 
#                  padding = True,
#                  max_length = None,
#                  pad_to_multiple_of = None,
#                  return_tensors = None,
#                  task: str = None
#                  ):
#         self.tokenizer = tokenizer
#         self.padding = padding
#         self.max_length = max_length
#         self.pad_to_multiple_of = pad_to_multiple_of
#         self.return_tensors = return_tensors
#         self.task = task
#         if task in _formatting:
#             self.formatter = _formatting[task]
#         else: raise ValueError(f"Task {task} not supported")

#     def __call__(self, examples):
#         examples = [self.formatter(x) for x in examples]
#         batch = self.tokenizer.pad(
#             examples,
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors=self.return_tensors,
#         )
        
#         if "label" in batch:
#             batch["labels"] = batch["label"]
#             del batch["label"]
#         if "label_ids" in batch:
#             batch["labels"] = batch["label_ids"]
#             del batch["label_ids"]
#         return batch

if __name__ == '__main__':
    # for c in [InstructionTuningDataset, SelfInstructDataset]:
    #     print(c.__name__)
    #     dataset = c()
    #     print(len(dataset))
    #     print(dataset[0])
    #     print(dataset[1])
    #     print(dataset[2])
    #     print(dataset[3])
    #     print(dataset[4])
    #     print(dataset[5])
    #     print(dataset[6])
    #     print(dataset[7])
    #     print(dataset[8])
    #     print(dataset[9])
    #     print('---' * 10)
    from transformers import DataCollatorWithPadding
    from torch.utils.data import DataLoader
    from transformers import GPT2TokenizerFast
    def preprocess_function(data_point):
        # Tokenize the texts
        args = InstructionTuningDataset.format(data_point)
        result = tokenizer(args, padding=False, max_length=512, truncation=True)

        
        # print('result', result)
        # exit(0)
        return result
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, padding="longest")
    dataset = InstructionTuningDataset()
    dataset.map(
        preprocess_function,
        load_from_cache = False
    )
    for batch in dataset:
        print(batch)
        break
    loader = DataLoader(dataset, batch_size=2, collate_fn=data_collator)
    for batch in loader:
        print(batch)
        break