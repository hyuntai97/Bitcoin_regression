import torch
import time 
import datetime
import numpy as np 
from tqdm import tqdm 
import matplotlib.pyplot as plt 
from utils import *
import os
from scipy.stats import norm
from scipy.stats import t

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
        n_samples, 
        input_window,
        output_window,
        log_dir
        ):   

        self.model = model
        self.testloader = test_dataloader
        self.yscaler = yscaler
        self.metric = metric
        self.n_samples = n_samples
        self.input_window = input_window
        self.output_window = output_window
        self.log_dir = log_dir

        #-- Test iteration 
        test_loss, true_list, y_pred_list, trend_list, seasonal_list  = self.test()

        #-- Save history
        self.history = {}
        self.history['test'] = test_loss

        #-- Plot prediction 
        true, y_pred, trend, seasonality = self.reshape_data(true_list, y_pred_list, trend_list, seasonal_list)
        self.plot_prediction(true, y_pred, trend, seasonality)


    def test(self):
        self.model.eval()

        loss_lst = []

        true_list=[]
        y_pred_list=[]
        trend_list=[]
        seasonal_list=[]    

        for n in tqdm(range(self.n_samples)):
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

                if n == 0:
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

    def plot_prediction(self, true, y_pred, trend, seasonality):
        # save directory
        save_directory = f'{self.log_dir}/predict_img'
        os.makedirs(save_directory, exist_ok=True)

        for i in tqdm(range(len(true[0]))):
            fig = make_subplots(
                subplot_titles=['True Vs Predicted','Trend','Seasonality'],
                rows=2, cols=2,
                vertical_spacing=0.1,
                horizontal_spacing=0.05,
                column_widths=[0.9, 0.6],
                row_heights=[0.8, 0.8],
                specs=[[{"rowspan": 2}, {}], [None, {}]])
            
            # 99% confidence interval
            pred_mean, pred_min_interval, pred_max_interval = self.confidence_interval(y_pred[:,i], 0.01)
            trend_mean, trend_min_interval, trend_max_interval = self.confidence_interval(trend[:,i], 0.01)
            seasonal_mean, seasonal_min_interval, seasonal_max_interval = self.confidence_interval(seasonality[:,i], 0.01)
            
            # plot(1,1) - prediction vs real_value
            ## CI
            fig.add_trace(go.Scatter(name = 'Upper Bound',
                                    x = list(range(len(true[0,0])-len(y_pred[0,0]), len(true[0,0]))),
                                    y = pred_max_interval,
                                    mode = 'lines',
                                    marker = dict(color="rgb(179,226,205)"),
                                    line = dict(width=0),
                                    showlegend = False), row=1,col=1)
            fig.add_trace(go.Scatter(name = 'Confidence Interval',
                                    x = list(range(len(true[0,0])-len(y_pred[0,0]), len(true[0,0]))),
                                    y = pred_min_interval,
                                    marker = dict(color="rgb(179,226,205)"),
                                    line = dict(width=0),
                                    mode = 'lines',
                                    fillcolor = 'rgba(179,226,205,0.7)',
                                    fill = 'tonexty',
                                    showlegend = True), row=1,col=1)
            ## mean value
            fig.add_trace(go.Scatter(x = list(range(len(true[0,0]))), y = list(true[0,i]),
                                    name = "Real value", line=dict(color="#636EFA")), row=1,col=1)
            fig.add_trace(go.Scatter(x = list(range(len(true[0,0])-len(y_pred[0,0]), len(true[0,0]))), y = pred_mean, 
                                    name = "Prediction average", line=dict(color="red")), row=1,col=1)
            
            # plot(1,2) - trend
            ## CI
            fig.add_trace(go.Scatter(name = 'Upper Bound',
                                    x = list(range(len(true[0,0])-len(y_pred[0,0]), len(true[0,0]))),
                                    y = trend_max_interval,
                                    mode = 'lines',
                                    marker = dict(color="rgb(179,226,205)"),
                                    line = dict(width=0),
                                    showlegend = False), row=1,col=2)
            fig.add_trace(go.Scatter(name = 'Confidence Interval',
                                    x = list(range(len(true[0,0])-len(y_pred[0,0]), len(true[0,0]))),
                                    y = trend_min_interval,
                                    marker = dict(color="rgb(179,226,205)"),
                                    line = dict(width=0),
                                    mode = 'lines',
                                    fillcolor = 'rgba(179,226,205,0.7)',
                                    fill = 'tonexty',
                                    showlegend = False), row=1,col=2)
            ## mean value
            fig.add_trace(go.Scatter(x = list(range(len(true[0,0])-len(y_pred[0,0]), len(true[0,0]))), y = trend_mean, 
                                    name = "Trend average", line=dict(color="red"), showlegend = False), row=1,col=2)
            
            # plot(2,2) - seasonality
            ## CI
            fig.add_trace(go.Scatter(name = 'Upper Bound',
                                    x = list(range(len(true[0,0])-len(y_pred[0,0]), len(true[0,0]))),
                                    y = seasonal_max_interval,
                                    mode = 'lines',
                                    marker = dict(color="rgb(179,226,205)"),
                                    line = dict(width=0),
                                    showlegend = False), row=2,col=2)
            fig.add_trace(go.Scatter(name = 'Confidence Interval',
                                    x = list(range(len(true[0,0])-len(y_pred[0,0]), len(true[0,0]))),
                                    y = seasonal_min_interval,
                                    marker = dict(color="rgb(179,226,205)"),
                                    line = dict(width=0),
                                    mode = 'lines',
                                    fillcolor = 'rgba(179,226,205,0.7)',
                                    fill = 'tonexty',
                                    showlegend = False), row=2,col=2)
            ## mean value
            fig.add_trace(go.Scatter(x = list(range(len(true[0,0])-len(y_pred[0,0]), len(true[0,0]))), y = seasonal_mean, 
                                    name = "Seasonality average", line=dict(color="red"), showlegend = False), row=2,col=2)
            # # dash line
            # full_fig = fig.full_figure_for_development()
            # fig.add_shape(type="line", xref='x', yref='paper',
            #             x0=len(true[0,0])-len(y_pred[0,0]), y0 = full_fig.layout.yaxis.range[0],
            #             x1=len(true[0,0])-len(y_pred[0,0]), y1 = full_fig.layout.yaxis.range[1], 
            #             line=dict(color="black", width=1, dash="dash"),row=1,col=1)

            fig.update_layout(height=550, width=1200, title_text="Bitcoin Price Prediction")
            fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True) # 테두리
            fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True) # 테두리
            fig.write_html(f'{save_directory}/nbeats_pred_{i}.html')
            pio.write_image(fig, f'{save_directory}/nbeats_pred_{i}.png', format='png', engine='kaleido')        
        
        fig.show()



    ##-- 결과값 reshape 및 신뢰구간 구축 
    def reshape_data(self, true_list, y_pred_list, trend_list, seasonal_list):

        true = np.array(true_list).reshape(self.n_samples, -1, self.output_window + self.input_window)
        y_pred = np.array(y_pred_list).reshape(self.n_samples, -1, self.output_window)
        trend = np.array(trend_list).reshape(self.n_samples, -1, self.output_window)
        seasonality = np.array(seasonal_list).reshape(self.n_samples, -1, self.output_window)
        
        return true, y_pred, trend, seasonality

    def confidence_interval(self, sample, alpha = 0.01):
        mean = np.mean(sample, axis=0)
        std_error = np.std(sample, axis=0)

        max_interval = mean + norm.ppf(alpha/2, loc = 0, scale = 1) * std_error/np.sqrt(len(sample))
        min_interval = mean - norm.ppf(alpha/2, loc = 0, scale = 1) * std_error/np.sqrt(len(sample))
        
        return mean, min_interval, max_interval