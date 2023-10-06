
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import (RationalQuadratic, Exponentiation, RBF, ConstantKernel)

from sklearn.gaussian_process import GaussianProcessRegressor



def quantiles_check(df, nutrients):
    q_low = df[nutrients[0::]].quantile(0.01)
    q_hi  = df[nutrients[0::]].quantile(0.99)
    
    df_return = df.copy()
    df_return[nutrients[0::]] = np.where( ((df[nutrients[0::]] < q_hi) & (df[nutrients[0::]] > q_low)), df[nutrients[0::]], np.nan)
    return df_return


def sum_year(df, notation, nutrients_to_sum):
    if np.size(notation)==1:
        df = df.sort_values(by="Date", ascending=True).reset_index()
        date = df[(df["sample.samplingPoint.notation"]==notation)]["Date"]
        sum_year = df[(df["sample.samplingPoint.notation"]==notation)][nutrients_to_sum].sum(min_count=1, axis=1)
        
    else:
        df = df.sort_values(by="Date", ascending=True).reset_index()
        date = df[(df["sample.samplingPoint.notation"].isin(notation))]["Date"]
        sum_year = df[(df["sample.samplingPoint.notation"].isin(notation))][nutrients_to_sum].sum(min_count=1, axis=1)
        
        

    return(pd.concat([date, sum_year], axis=1, keys=['Date', 'sum']))

def sum_seasons(df, notation, nutrients_to_sum):
    df = df.sort_values(by="Date", ascending=True).reset_index()

    if np.size(notation)==1:
        date_summer = df[(df["sample.samplingPoint.notation"]==notation)&(df["season"]==0)]["Date"]
        dain_summer = df[(df["sample.samplingPoint.notation"]==notation)&(df["season"]==0)][nutrients_to_sum].sum(min_count=1, axis=1)

        date_winter = df[(df["sample.samplingPoint.notation"]==notation)&(df["season"]==1)]["Date"]
        dain_winter = df[(df["sample.samplingPoint.notation"]==notation)&(df["season"]==1)][nutrients_to_sum].sum(min_count=1, axis=1)
    else:

        date_summer = df[(df["sample.samplingPoint.notation"].isin(notation))&(df["season"]==0)]["Date"]
        dain_summer = df[(df["sample.samplingPoint.notation"].isin(notation))&(df["season"]==0)][nutrients_to_sum].sum(min_count=1, axis=1)

        date_winter = df[(df["sample.samplingPoint.notation"].isin(notation))&(df["season"]==1)]["Date"]
        dain_winter = df[(df["sample.samplingPoint.notation"].isin(notation))&(df["season"]==1)][nutrients_to_sum].sum(min_count=1, axis=1)
    return(pd.concat([date_summer, dain_summer], axis=1, keys=['Date', 'sum']), pd.concat([date_winter, dain_winter], axis=1, keys=['Date', 'sum']))

def plot_sum_summer(df, notation, nutrients_to_sum, ax=None, plt_kwargs={}, sct_kwargs={}, xlabel=None, ylabel=None, plot_title=None, label_str=None):
	df = df.sort_values(by="Date", ascending=True).reset_index()

	df_summer, df_winter = sum_seasons(df, notation, nutrients_to_sum)
		
	if ax is None:
		ax = plt.gca()		
	ax.scatter(df_summer["Date"], df_summer["sum"], zorder=2, **sct_kwargs, label=label_str)

	if ylabel != None:
		ax.set_ylabel(str(ylabel))
	if xlabel != None:
		ax.set_xlabel(str(xlabel))
	if label_str != None:
		ax.legend(fontsize=18)
	if plot_title != None:
		ax.set_title(label=str(plot_title), fontsize=22)
			
	return(ax)

def plot_sum_winter(df, notation, nutrients_to_sum, ax=None,  plt_kwargs={}, sct_kwargs={}, xlabel=None, ylabel=None, plot_title=None, label_str=None):

	df_summer, df_winter = sum_seasons(df, notation, nutrients_to_sum)
		
	if ax is None:
		ax = plt.gca()		
	ax.scatter(df_winter["Date"], df_winter["sum"], zorder=2, **sct_kwargs, label=label_str)

	if ylabel != None:
		ax.set_ylabel(str(ylabel))
	if xlabel != None:
		ax.set_xlabel(str(xlabel))
	if label_str != None:
		ax.legend(fontsize=18)
	if plot_title != None:
		ax.set_title(label=str(plot_title), fontsize=22)
			
	return(ax)
	
	
	
def pred_linreg(x,y, date):
    m, c = np.polyfit(x[(np.isfinite(x))&(np.isfinite(y))], y[(np.isfinite(x))&(np.isfinite(y))], 1)
    y_model = m * date + c
    return y_model

def date_to_sec(df1, df2):


    return(np.array([(df1.reset_index().loc[i, "Date"]-df2.reset_index()["Date"][0]).total_seconds()*10**-8 for i in range(len(df1["Date"]))]))

def LinearRegression_boot_summer(df, notation, nutrients_to_sum , ax=None):
    df_summer, df_winter = sum_seasons(df, notation, nutrients_to_sum)
    
    x_df_sec = date_to_sec(df_summer, df)
    date = date_to_sec(df, df)
    
    x_df = np.array(df_summer["Date"])
    y_df = np.array(df_summer["sum"])
    
    lr_boot = []
    for i in range(0, 500):
        sample_index = np.random.choice(range(0, len(y_df)), len(y_df))
        X_samples = x_df_sec[sample_index]
        y_samples = y_df[sample_index]
        
        lr_boot.append(pred_linreg(X_samples, y_samples, date))
    

    lr_boot = np.array(lr_boot)
    lr = pred_linreg(x_df_sec, y_df, date)
    q_16, q_84 = np.percentile(lr_boot, (16, 84), axis=0)
    
    if ax is None:
        ax = plt.gca()
    ax.fill_between(df["Date"], q_16, q_84, alpha=0.2, color="grey")

    ax.plot(df["Date"], lr, color='darkblue', zorder=5)
    
    return ax

def LinearRegression_boot_winter(df, notation, nutrients_to_sum , ax=None):
    df_summer, df_winter = sum_seasons(df, notation, nutrients_to_sum)
    
    x_df_sec = date_to_sec(df_winter, df)
    date = date_to_sec(df, df)
    
    x_df = np.array(df_winter["Date"])
    y_df = np.array(df_winter["sum"])
    
    lr_boot = []
    for i in range(0, 500):
        sample_index = np.random.choice(range(0, len(y_df)), len(y_df))
        X_samples = x_df_sec[sample_index]
        y_samples = y_df[sample_index]
        
        lr_boot.append(pred_linreg(X_samples, y_samples, date))
    

    lr_boot = np.array(lr_boot)
    lr = pred_linreg(x_df_sec, y_df, date)
    q_16, q_84 = np.percentile(lr_boot, (16, 84), axis=0)
    
    if ax is None:
        ax = plt.gca()
    ax.fill_between(df["Date"], q_16, q_84, alpha=0.2, color="grey")

    ax.plot(df["Date"], lr, color='darkblue', zorder=5)
    
    return ax
    
def GP_missing_date_Xy(df, notation, nutrients_to_sum):

    df_sum = sum_year(df, notation, nutrients_to_sum)
    df_sum = df_sum.dropna().reset_index()
    df_sum_sec = np.array([(df_sum.loc[i, "Date"]-df_sum["Date"][0]).total_seconds()*10**-8 for i in range(len(df_sum["Date"]))])

    fitting = np.polyfit(df_sum_sec, df_sum["sum"], 
                     deg=2)

    p = np.poly1d(fitting)

    y_mean = p(df_sum_sec)

    X = df_sum_sec.reshape(-1, 1)
    y = df_sum["sum"]

    return(X, y, y_mean, p)

def find_missing_years(df, notation, nutrients_to_sum):
    df_year = np.unique(df.sort_values(by="Date", ascending=True).reset_index()["Date"])

    missing_years = [y for y in df_year if y not in np.unique(sum_year(df, notation, nutrients_to_sum).dropna()["Date"])]
    
    return(missing_years)

def find_all_years(df, notation, nutrients_to_sum):
    df_year = np.unique(df.sort_values(by="Date", ascending=True).reset_index()["Date"])
    
    return(df_year)
    
def GP_imputing(df, notation, nutrients_to_sum, kernel):
    
    X, y, y_mean, p = GP_missing_date_Xy(df, notation, nutrients_to_sum)
    
    gaussian_process = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gaussian_process.fit(X, y-y_mean)

    missing_years = find_missing_years(df, notation, nutrients_to_sum)
    
    dates_random = pd.to_datetime(missing_years) 
    df_sum = sum_year(df, notation, nutrients_to_sum).reset_index()

    X_test = np.array((dates_random-pd.to_datetime(df_sum["Date"][0])).total_seconds()*10**-8)

    mean_y_pred, std_y_pred = gaussian_process.predict(X_test.reshape(-1,1), return_std=True)
    
    
    new_values = np.random.normal(mean_y_pred, std_y_pred, len(missing_years))+p(X_test)
    new_values[new_values<0] = 0.

    return(missing_years, new_values)
    
def GP_imputing(df, notation, nutrients_to_sum, kernel):
    
    X, y, y_mean, p = GP_missing_date_Xy(df, notation, nutrients_to_sum)
    
    gaussian_process = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gaussian_process.fit(X, y-y_mean)

    missing_years = find_missing_years(df, notation, nutrients_to_sum)
    
    dates_random = pd.to_datetime(missing_years) 
    df_sum = sum_year(df, notation, nutrients_to_sum).reset_index()

    X_test = np.array((dates_random-pd.to_datetime(df_sum["Date"][0])).total_seconds()*10**-8)

    mean_y_pred, std_y_pred = gaussian_process.predict(X_test.reshape(-1,1), return_std=True)
    
    
    new_values = np.random.normal(mean_y_pred, std_y_pred, len(missing_years))+p(X_test)
    new_values[new_values<0] = 0.

    return(missing_years, new_values)
    
def GP_kernel (noise_level_=5, length_scale_ = 0.25, alpha_ = 1):

    noise_kernel = WhiteKernel(noise_level=noise_level_, noise_level_bounds=(1e-10, 1e+3))


    irregularities_kernel = RationalQuadratic(length_scale=length_scale_, alpha=alpha_, length_scale_bounds=(1e-5, 1e5))

    kernel =   noise_kernel + irregularities_kernel

    return(kernel)
    

