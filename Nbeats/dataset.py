from torch.utils.data import Dataset
import torch


class WindowDataset(Dataset):
    '''
    Build a custom dataset 
    
    --Return--
    Y : 
    Yf : 
    ----------
    
    '''
    
    def __init__(self, data, input_window, output_window, stride=1):
        # total data length 
        L = data.shape[0]
        
        # total number of samples with stride
        num_samples = (L - input_window - output_window) // stride + 1
        
        # input, output 
        Y = []
        Yf = []

        for i in range(num_samples):
            start_x = stride*i
            end_x = start_x + input_window 
            Y.append(data.iloc[start_x: end_x, 0].values)

            start_y = stride*i + input_window
            end_y = start_y + output_window 
            Yf.append(data.iloc[start_y:end_y, 0].values)

        self.Y = Y
        self.Yf = Yf
        
        self.len = len(Y)
            
    def __len__(self):
        return self.len 
    
    def __getitem__(self, idx):
        Y = torch.FloatTensor(self.Y[idx])
        Y = Y.unsqueeze(-1)
        Yf = torch.FloatTensor(self.Yf[idx])
        Yf = Yf.unsqueeze(-1)
        
        return Y, Yf