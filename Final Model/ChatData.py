from torch.utils.data import Dataset
import json


class ChatData(Dataset):
    """
    Dataset class for the chat data
    """
    def __init__(self, path: str, tokenizer):
        """
        Initialize the dataset
        """
        self.data = json.load(open(path, "r"))

        self.X = []
        for i in self.data:
            for j in i['dialog']:
                self.X.append(j['text'])

        X_new = []

        for idx in range(0, len(self.X), 2):
            try:
                i = self.X[idx]

                if((len(self.X[idx].split(' ')) + len(self.X[idx+1].split(' '))) > 100):
                    
                    to_be_truncated = len(self.X[idx].split(' ')) + len(self.X[idx+1].split(' ')) - 100
                    self.X[idx] = " ".join(
                        self.X[idx].split()[:-to_be_truncated])
                    i = self.X[idx]

                string_feed = "<START> " + \
                    i + " <bot>: " + \
                    " ".join(self.X[idx+1].split()) + " <END>"

                X_new.append(string_feed)

            except:
                print("Error at index: "+str(idx))
                break
            

        self.X = X_new
        print(len(self.X))

        self.X = self.X[:9358]
        
        
        # maxi = 0
        # for x in self.X:
        #     maxi = max(maxi, len(tokenizer(x)['input_ids']))
        # print("Max length  : "+str(maxi)+"\n\n\n")

        self.X_encoded = tokenizer(
            self.X, max_length=100, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Return the input_ids and attention_mask for the given index
        """
        return (self.input_ids[idx], self.attention_mask[idx])
