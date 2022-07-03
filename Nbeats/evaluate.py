import torch
import time 
import datetime
import numpy as np 
from tqdm import tqdm 
import matplotlib.pyplot as plt 
from utils import *
import os

#-- Viz
import plotly.express as px
import plotly.io as pio
import kaleido
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class ModelTest:
    '''
    Model Testing
    '''

    def __init__(
        self, 
        model, 
        test_dataloader,
        yscaler,
        metric,
        input_window,
        output_window,
        log_dir
        ):   

        self.model = model
        self.testloader = test_dataloader
        self.yscaler = yscaler
        self.metric = metric
        self.input_window = input_window
        self.output_window = output_window
        self.log_dir = log_dir

        #-- Test iteration 
        test_loss, true_list, y_pred_list, trend_list, seasonal_list  = self.test()

        #-- Plot prediction 
        self.plot_prediction(true_list, y_pred_list, trend_list, seasonal_list)

        #-- Save history
        self.history = {}
        self.history['test'] = test_loss


    def test(self):
        self.model.eval()

        loss_lst = []

        true_list=[]
        y_pred_list=[]
        trend_list=[]
        seasonal_list=[]    

        for i, (x_test, y_test) in enumerate(self.testloader):
            backcast, y_pred, trend_forecast, seasonal_forecast = self.model(x_test.to(self.model.device))

            y_pred = y_pred.squeeze(0).detach().cpu().numpy()
            y_test = y_test.squeeze(0).detach().cpu().numpy()
            x_test = x_test.squeeze(0).detach().cpu().numpy() 
            trend_forecast = trend_forecast.squeeze(0).detach().cpu().numpy()
            seasonal_forecast = seasonal_forecast.squeeze(0).detach().cpu().numpy()

            pred = self.yscaler.inverse_transform(y_pred)
            y_test = self.yscaler.inverse_transform(y_test)
            x_test = self.yscaler.inverse_transform(x_test)
            true = np.concatenate([x_test, y_test])
            
            trend_forecast = self.yscaler.inverse_transform(trend_forecast)
            seasonal_forecast = self.yscaler.inverse_transform(seasonal_forecast)

            # metric option 
            if self.metric == 'MAPE':
                loss = MAPEval(pred, y_test)
            else:
                pass

            print("loss {}: {}".format(self.metric, loss))
            loss_lst.append(loss)

            # 20번 마다 예측값, 실제값, trend, seasonality 리스트로 저장
            if i % 20 == 0:
                true_list.append(true)
                y_pred_list.append(pred)
                trend_list.append(trend_forecast)
                seasonal_list.append(seasonal_forecast)      

        return loss_lst, true_list, y_pred_list, trend_list, seasonal_list    

    def plot_prediction(self, true_list, y_pred_list, trend_list, seasonal_list):
        # save directory
        save_directory = f'{self.log_dir}/predict_img'
        os.makedirs(save_directory, exist_ok=True)

        for i in range(len(true_list)):
            
            fig = make_subplots(
                subplot_titles=['True Vs Predicted','Trend','Seasonality'],
                rows=2, cols=2,
                vertical_spacing=0.1,
                horizontal_spacing=0.05,
                column_widths=[0.9, 0.6],
                row_heights=[0.8, 0.8],
                specs=[[{"rowspan": 2}, {}], [None, {}]])        

            fig.add_trace(go.Scatter(x = list(range(len(true_list[i]))), y = np.array(list(map(float, true_list[i]))),
                                    name = "Real value"),row=1,col=1)
            fig.add_trace(go.Scatter(x = list(range(len(true_list[i])-len(y_pred_list[i]), len(true_list[i]))), y = y_pred_list[i], 
                                    name = "Prediction", line=dict(color="red")),row=1,col=1)
            fig.add_trace(go.Scatter(x = list(range(len(true_list[i])-len(y_pred_list[i]), len(true_list[i]))), y = trend_list[i], 
                                    name = "Trend"), row=1,col=2)
            fig.add_trace(go.Scatter(x = list(range(len(true_list[i])-len(y_pred_list[i]), len(true_list[i]))), y = seasonal_list[i], 
                                    name = "Seasonality"), row=2,col=2)
            # dash line
            full_fig = fig.full_figure_for_development()
            fig.add_shape(type="line", xref='x', yref='paper',
                        x0=len(true_list[i])-len(y_pred_list[i]), y0 = full_fig.layout.yaxis.range[0],
                        x1=len(true_list[i])-len(y_pred_list[i]), y1 = full_fig.layout.yaxis.range[1], 
                        line=dict(color="black", width=1, dash="dash"),row=1,col=1)

            fig.update_layout(height=550, width=1200, title_text="Bitcoin Price Prediction")
            fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True) # 테두리
            fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True) # 테두리
            fig.write_html(f'{save_directory}/nbeats_pred_{i}.html')
            pio.write_image(fig, f'{save_directory}/nbeats_pred_{i}.png', engine='kaleido')
        
        fig.show()
