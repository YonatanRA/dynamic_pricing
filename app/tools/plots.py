import json

import pandas as pd
import numpy as np
import plotly
import plotly.express as px


PRODUCTS = {'surrogate': {'alpha': -0.02473518, 'beta': 5.47213038, 'mean': 28.8},
            'extracash': {'alpha': -0.05629794, 'beta': 5.64341742, 'mean': 21.85},
            'plussurrogate': {'alpha': -0.22629794, 'beta': 5.64341742, 'mean': 6.4}
            }

x = np.linspace(1, 100, 100)


def demand_plot(product):

    global x, PRODUCTS

    prod = PRODUCTS[product]

    df_plot = pd.DataFrame()
    df_plot['Price'] = x
    df_plot['Demand'] = np.exp(prod['alpha']*x + prod['beta'])*100

    mean_demand = np.exp(prod['alpha']*prod['mean'] + prod['beta'])*100

    title = f'Demand function of {product.title()} with average price {prod["mean"]} and demand {round(mean_demand, 2)}'

    fig = px.line(df_plot, x='Price', y='Demand', title=title)

    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                      'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def profit_plot(product):

    global x, PRODUCTS

    prod = PRODUCTS[product]

    df_plot = pd.DataFrame()
    df_plot['Price'] = x
    df_plot['Demand'] = np.exp(prod['alpha']*x + prod['beta'])*100
    df_plot['Profit'] = df_plot.Price * df_plot.Demand

    product_title = product.title()
    profit_title = round(
        np.exp(prod['alpha']*prod["mean"] + prod['beta'])*100*prod["mean"], 2)
    best_price_title = round(
        df_plot[df_plot.Profit == df_plot.Profit.max()].Price.values[0], 2)
    best_profit_title = round(
        df_plot[df_plot.Profit == df_plot.Profit.max()].Profit.values[0], 2)
    increment = round((best_profit_title - profit_title) / profit_title * 100, 2)ﬂ

    title = f'Profit function of {product_title} with average price {prod["mean"]} and profit {profit_title} -- Best Price {best_price_title} with and profit {best_profit_title} -- {increment}% increment'

    fig = px.line(df_plot, x='Price', y='Profit', title=title)

    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                      'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


PLOT_FUNCTIONS = {'demand': demand_plot,
                  'profit': profit_plot,
                  }
