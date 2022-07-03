from torch.utils.data import Dataset
import torch

class WindowDataset(Dataset):
    '''
    Build a custom dataset 
    
    --Return--
    
    x: inputs
    y: targets 
    z: features
    ----------
    
    '''
    
    def __init__(self, data, input_window, output_window, stride=1):
        # total data length 
        L = data.shape[0]
        
        # total number of samples with stride
        num_samples = (L - input_window - output_window) // stride + 1
        
        # input, output 
        X = []
        Y = []
        Xf = []
        Yf = []
        
        for i in range(num_samples):
            start_x = stride*i
            end_x = start_x + input_window 
            Y.append(data.iloc[start_x: end_x, 0].values)
            X.append(data.iloc[start_x: end_x, 1:].values)
            start_y = stride*i + input_window
            end_y = start_y + output_window 
            Yf.append(data.iloc[start_y:end_y, 0].values)
            Xf.append(data.iloc[start_y:end_y, 1:].values)
            
        self.X = X
        self.Y = Y
        self.Xf = Xf
        self.Yf = Yf
        
        self.len = len(X)
            
    def __len__(self):
        return self.len 
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.X[idx])
        Y = torch.FloatTensor(self.Y[idx])
        Y = Y.unsqueeze(-1)
        Xf = torch.FloatTensor(self.Xf[idx])
        Yf = torch.FloatTensor(self.Yf[idx])
        Yf = Yf.unsqueeze(-1)
        
        return X, Y, Xf, Yf