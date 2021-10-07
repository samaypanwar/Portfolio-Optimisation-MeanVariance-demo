import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quant_risk as qr

def plot_series(dataframe):

    plt.figure(figsize=(14, 7))
    for c in dataframe.columns.values:
        plt.plot(dataframe.index, dataframe[c], lw=3, alpha=0.8,label=c)
    plt.legend(loc='upper left', fontsize=12)
    plt.ylabel('price in $')


def portfolio_annualised_performance(weights, returns):

    annualised_portfolio_returns = qr.statistics.annualize.annualised_returns(np.dot(returns,weights))
    annualised_portfolio_var = qr.statistics.annualize.annualised_volatility(np.dot(returns, weights))

    return annualised_portfolio_returns, annualised_portfolio_var

def random_portfolios(num_portfolios, returns, risk_free_rate):

    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):

        weights = np.random.random(len(returns.columns))
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return, portfolio_std_dev = portfolio_annualised_performance(weights, returns)

        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev

    return results, weights_record


def display_simulated_ef_with_random(dataframe, num_portfolios, risk_free_rate):

    returns = dataframe.pct_change().dropna()

    results, weights = random_portfolios(num_portfolios, returns, risk_free_rate)

    pf = qr.portfolio.mean_variance.MeanVariance(dataframe)
    max_sharpe_allocation = pf.fit('max_sharpe')
    max_sharpe_returns, max_sharpe_vol, max_sharpe = pf.stats(verbose=False)

    max_sharpe_allocation = pd.DataFrame.from_dict(max_sharpe_allocation, orient='index', columns=['allocation']).T

    pf = qr.portfolio.mean_variance.MeanVariance(dataframe)
    min_vol_allocation = pf.fit('min_volatility')
    min_vol_returns, min_vol, min_vol_sharpe = pf.stats(verbose=False)

    min_vol_allocation = pd.DataFrame.from_dict(min_vol_allocation, orient='index', columns=['allocation']).T
    # min_vol_allocation.allocation = [round(i*100, 2)for i in min_vol_allocation.allocation]
    # min_vol_allocation = min_vol_allocation.T

    print("-"*80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return: ", round(max_sharpe_returns,2))
    print("Annualised Volatility: ", round(max_sharpe_vol,2))
    print("Sharpe Ratio: ", round(max_sharpe, 2))
    print()
    print(max_sharpe_allocation)

    print("-"*80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return: ", round(min_vol_returns, 2))
    print("Annualised Volatility: ", round(min_vol, 2))
    print("Sharpe Ratio: ", round(min_vol_sharpe, 2))
    print()
    print(min_vol_allocation)

    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='YlGnBu', marker='o', s=10, alpha=0.3)

    plt.colorbar()
    plt.scatter(max_sharpe_vol, max_sharpe_returns, marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(min_vol, min_vol_returns,marker='*',color='g',s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)