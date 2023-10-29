"""
-*- coding: utf-8 -*-

@Author : Aoran,Li
@Time : 2023/10/25 13:42
@File : backtesting.py
"""
import pandas as pd

from __init__ import *

def back_test_plot(df):
    df['Date'] = pd.to_datetime(df['Date'])
    fig, ax = plt.subplots(dpi=150)
    # plt.figure(dpi=150)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # 设置x轴刻度间隔为5天
    # ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    for col in ['equal_ret', 'long_ret', 'short_ret', 'long_short']:
        y = (1 + df[col]).cumprod()
        plt.plot(df['Date'], y, label=col)
    ax.axhline(y=1, color = 'gray', linestyle='--')
    plt.legend(loc='best')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    # plt.title('CNN20D2C Model Performance')
    plt.show()

def back_testing(df: pd.DataFrame, label_='Ret_20d'):
    df_res = pd.DataFrame()
    for date_g, df_g in df.groupby('Date'):
        df_g = df_g.dropna()
        equal_ret = np.mean(df_g[label_]).round(6)
        market_ret = np.mean((df_g[label_] * df_g['MarketCap']) / sum(df_g['MarketCap'])).round(8)
        long_ret = np.mean(df_g[df_g['label_1_prob'] > df_g['label_1_prob'].quantile(0.9)][label_]).round(6)
        short_ret = np.mean(df_g[df_g['label_0_prob'] > df_g['label_0_prob'].quantile(0.9)][label_]).round(6)
        long_short = round(long_ret - short_ret, 6)

        df_res.loc[len(df_res), ['Date', 'equal_ret', 'market_ret', 'long_ret', 'short_ret', 'long_short']] = (
        date_g,
        equal_ret,
        market_ret,
        long_ret,
        short_ret,
        long_short)
    return df_res.sort_values(by=['Date'], ascending=True)



def test_result(df):
    # loss 自己算
    acc = accuracy_score(df['label'], df['label_pre']).__round__(4)
    t_num = len(df[df['label'] == 1])
    t_p_num = len(df[(df['label'] == 1) & (df['label_pre'] == 1)])
    t_p_r = (t_p_num / t_num).__round__(4)

    f_num = len(df[df['label'] == 0])
    t_n_num = len(df[(df['label'] == 0) & (df['label_pre'] == 0)])
    t_n_r = (t_n_num / f_num).__round__(4)


    acc_tn = accuracy_score(df[df['label'] == 0]['label'], df[df['label']==0]['label_pre']).__round__(4)

    f1 = f1_score(df['label'], df['label_pre'], average='macro').__round__(4)

    t_acc = []
    df['Date'] = pd.to_datetime(df['Date'])
    for t_r in [pd.date_range('2002-01-01', '2006-01-01'), pd.date_range('2006-01-01', '2012-01-01'), pd.date_range('2012-01-01', '2020-01-01')]:
        df_t = df[df['Date'].isin(t_r)]
        t_acc.append(accuracy_score(df_t['label'], df_t['label_pre']).__round__(4))

    df_res = pd.DataFrame()
    df_res.loc[0, 'acc'] = acc
    df_res.loc[0, 'acc_tp'] = f'{t_p_r} ({t_p_num}/{t_num})'
    df_res.loc[0, 'acc_tn'] = f'{t_n_r} ({t_n_num}/{f_num})'
    df_res.loc[0, 'f1_score'] = f1
    df_res.loc[0, ['acc(02-05)', 'acc(06-11)', 'acc(12-19)']] = t_acc

    return df_res



if __name__ == '__main__':
    df_l = {}
    for res_f in os.listdir(r'./result'):
        df_model = pd.read_csv(os.path.join(r'./result', res_f), index_col=0)
        df_l[f'{res_f[:-11]}'] = test_result(df_model)
        del df_model
        gc.collect()
    df_test_res = pd.concat(df_l.values(), keys=df_l.keys())

    df_5D2C = pd.read_csv(r'result/CNN5D2C_results_new.csv', index_col=0)
    CNN5D2C_backtest = back_testing(df_5D2C, label_='Ret_5d')

    CNN5D2C_backtest.to_csv('CNN5D2C_backtesting.csv')
    df_5d2c = pd.read_csv('CNN5D2C_backtesting.csv', index_col=0)

    back_test_plot(df_5d2c)

