import torch
import time 
import datetime
import numpy as np 
from tqdm import tqdm 
import matplotlib.pyplot as plt 
from utils import *
import os


class ModelTest:
    '''
    Model Testing
    '''

    def __init__(
        self, 
        model, 
        test_dataloader,
        yscaler,
        n_samples, 
        metric,
        input_window,
        output_window,
        log_dir
        ):   

        self.model = model
        self.testloader = test_dataloader
        self.yscaler = yscaler
        self.n_samples = n_samples
        self.metric = metric
        self.input_window = input_window
        self.output_window = output_window
        self.log_dir = log_dir

        #-- Test iteration 
        test_loss = self.test()

        #-- Save history
        self.history = {}
        self.history['test'] = test_loss


    def test(self):
        # save directory
        save_directory = f'{self.log_dir}/predict_img'
        os.makedirs(save_directory, exist_ok=True)

        self.model.eval()

        loss_lst = []

        for i, (X_te, Y_te, Xf_te, Yf_te) in enumerate(self.testloader):
            result = []
            for _ in tqdm(range(self.n_samples)):
                y_pred, _, _ = self.model(X_te, Y_te, Xf_te)
                y_pred = y_pred.data.numpy()
                y_pred = self.yscaler.inverse_transform(y_pred)

                result.append(y_pred.reshape((-1,1)))

            result = np.concatenate(result, axis=1)
            p50 = np.quantile(result, 0.5, axis=1)
            p60 = np.quantile(result, 0.6, axis=1)
            p40 = np.quantile(result, 0.4, axis=1)   

            y_true = self.yscaler.inverse_transform(Yf_te.data.numpy())

            # metric option 
            if self.metric == 'MAPE':
                loss = MAPEval(p50, y_true)
            else:
                pass

            print("P50 {}: {}".format(self.metric, loss))
            loss_lst.append(loss)  

            # 20번 마다 plot 저장 
            if i % 20 == 0:
                Y_total = torch.concat([Y_te, Yf_te], dim=1)
                Y_total = self.yscaler.inverse_transform(Y_total.data.numpy())
                plt.figure(1, figsize=(20, 5))
                plt.plot([k + self.output_window + self.input_window - self.output_window \
                    for k in range(self.output_window)], p50, "r-")
                plt.fill_between(x=[k + self.output_window + self.input_window - self.output_window for k in range(self.output_window)], \
                    y1=p40, y2=p60, alpha=0.5)
                plt.title('Prediction uncertainty')
                yplot = Y_total[-1, -self.output_window-self.input_window:]
                plt.plot(range(len(yplot)), yplot, "k-")
                plt.legend(["P50 forecast", "P40-P60 quantile", "true"], loc="upper left")
                ymin, ymax = plt.ylim()
                plt.vlines(self.output_window + self.input_window - self.output_window, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
                plt.ylim(ymin, ymax)
                plt.xlabel("Periods")
                plt.ylabel("Y")
                # plt.show()
                plt.savefig(f'{save_directory}/deepar_pred_{i}.png')
                plt.clf() # Clear the current figure            
            
        return loss_lst