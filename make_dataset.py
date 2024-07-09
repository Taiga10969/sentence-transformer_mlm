import re
import json
from tqdm import tqdm
from torch.utils.data import Dataset

class ChemRxiv_hypo_conc_Dataset(Dataset):
    def __init__(self, json_file):
        self.data = []
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        hypothesis = self.data[idx]['hypothesis']
        conclusion = self.data[idx]['conclusion']
        label = self.data[idx]['label']
        hypothesis_score = self.data[idx]['hypothesis_score']
        conclusion_score = self.data[idx]['conclusion_score']
        
        return hypothesis, conclusion, label, hypothesis_score, conclusion_score

# テキストのクリーンアップ関数
def clean_text(text):
    # "" などの記号や改行を取り除く正規表現
    #text = re.sub(r'[^\w\s]', '', text)  # 句読点を除去
    text = text.replace("\n", " ")       # 改行をスペースに置換
    text = text.strip()                  # 前後の空白を削除
    return text




if __name__ == '__main__':
    
    min_length = 200 #文字数が200以下の文章は削除
    json_file_path = '/taiga/Datasets/chemrxiv/hypo_conc_summary_data/chemrxiv_hypo_conc_test.jsonl' # train => train.text or test => dev.text
    dataset = ChemRxiv_hypo_conc_Dataset(json_file_path)
    
    num = 0
    with open("./dataset/dev.txt", "w") as file:
        for i in tqdm(range(len(dataset)), desc="Writing train.txt"):
            hypothesis, conclusion, label, hypothesis_score, conclusion_score = dataset[i]

            cleaned_hypothesis = clean_text(hypothesis)
            cleaned_conclusion = clean_text(conclusion)
            
            if len(cleaned_hypothesis) >= min_length:
                file.write(cleaned_hypothesis + "\n")
                num += 1
            if len(cleaned_conclusion) >= min_length:
                file.write(cleaned_conclusion + "\n")
                num += 1

    print(f"train.txt が作成されました。data_num : {num}")
