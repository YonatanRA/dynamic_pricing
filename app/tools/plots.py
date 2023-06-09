import json

import pandas as pd
import numpy as np
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import os

PATH = os.path.dirname(os.path.abspath(__file__))


PRODUCTS = {'purchase': {'alpha': -0.02473518, 'beta': 5.47213038, 'mean': 28.8},
            'cash': {'alpha': -0.05629794, 'beta': 5.64341742, 'mean': 21.85},
            'debt': {'alpha': -0.22629794, 'beta': 5.64341742, 'mean': 6.4}
            }

x = np.linspace(1, 100, 1000)


def demand_plot(product):

    global x, PRODUCTS

    prod = PRODUCTS[product]

    df_plot = pd.DataFrame()
    df_plot['Price'] = x
    df_plot['Demand'] = np.exp(prod['alpha']*x + prod['beta'])*100
    df_plot['Profit'] = df_plot.Price * df_plot.Demand

    max_price = df_plot[df_plot.Profit == df_plot.Profit.max()].Price.values[0]
    max_demand = df_plot[df_plot.Profit ==
                         df_plot.Profit.max()].Demand.values[0]

    mean_demand = np.exp(prod['alpha']*prod['mean'] + prod['beta'])*100

    title = f'Demand function of {product.title()} with average Price {prod["mean"]}€ and Demand {round(mean_demand, 2)} operations per day'

    fig = px.line(df_plot, x='Price', y='Demand', title=title)

    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                      'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

    fig.add_shape(type="rect",
                  x0=prod['mean']-0.1, y0=mean_demand,
                  x1=prod['mean']+0.1, y1=mean_demand,
                  line=dict(color="red", width=20,),
                  fillcolor="red",
                  )

    fig.add_shape(type="rect",
                  x0=max_price-0.1, y0=max_demand,
                  x1=max_price+0.1, y1=max_demand,
                  line=dict(color="green", width=20,),
                  fillcolor="green",
                  )

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
    increment = round((best_profit_title - profit_title) /
                      profit_title * 100, 2)

    title = f'Profit function of {product_title} with average Price {prod["mean"]}€ and Profit {profit_title}€ -- Best Price {best_price_title}€ with Profit {best_profit_title}€ -- {increment}% Increment'

    fig = px.line(df_plot, x='Price', y='Profit', title=title)

    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                      'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

    fig.add_shape(type="rect",
                  x0=prod['mean']-0.1, y0=profit_title,
                  x1=prod['mean']+0.1, y1=profit_title,
                  line=dict(color="red", width=20,),
                  fillcolor="red",
                  )

    fig.add_shape(type="rect",
                  x0=best_price_title-0.1, y0=best_profit_title,
                  x1=best_price_title+0.1, y1=best_profit_title,
                  line=dict(color="green", width=20,),
                  fillcolor="green",
                  )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def both_plot(product):

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
    increment = round((best_profit_title - profit_title) /
                      profit_title * 100, 2)

    max_price = df_plot[df_plot.Profit == df_plot.Profit.max()].Price.values[0]
    max_demand = df_plot[df_plot.Profit ==
                         df_plot.Profit.max()].Demand.values[0]

    mean_demand = np.exp(prod['alpha']*prod['mean'] + prod['beta'])*100

    title = f'Profit function of {product_title} with average Price {prod["mean"]}€ and Profit {profit_title}€ -- Best Price {best_price_title}€ with Profit {best_profit_title}€ -- {increment}% Increment'

    fig = make_subplots(rows=2, cols=1)

    fig0 = go.Scatter(x=df_plot['Price'], y=df_plot['Demand'], name='demand')

    fig1 = go.Scatter(x=df_plot['Price'], y=df_plot['Profit'],
                      name='profit', line=dict(color='green'))

    fig.add_trace(fig0, row=1, col=1)
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                      'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

    fig.add_shape(type="rect",
                  x0=prod['mean']-0.1, y0=mean_demand,
                  x1=prod['mean']+0.1, y1=mean_demand,
                  line=dict(color="red", width=20,),
                  fillcolor="red",
                  row=1, col=1
                  )

    fig.add_shape(type="rect",
                  x0=max_price-0.1, y0=max_demand,
                  x1=max_price+0.1, y1=max_demand,
                  line=dict(color="green", width=20,),
                  fillcolor="green",
                  row=1, col=1
                  )
    fig.add_trace(fig1, row=2, col=1)

    fig.add_shape(type="rect",
                  x0=prod['mean']-0.1, y0=profit_title,
                  x1=prod['mean']+0.1, y1=profit_title,
                  line=dict(color="red", width=20,),
                  fillcolor="red",
                  row=2, col=1
                  )

    fig.add_shape(type="rect",
                  x0=best_price_title-0.1, y0=best_profit_title,
                  x1=best_price_title+0.1, y1=best_profit_title,
                  line=dict(color="green", width=20,),
                  fillcolor="green",
                  row=2, col=1
                  )

    fig.update_layout(title=dict(text=title)
                      )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def line_overview():

    df_plot = pd.read_parquet(PATH + '/../data/line_plot.parquet')
    
    fig = px.area(df_plot, x='date', y='# Count', title='', width=1400, height=450)

    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                      'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

    fig.update_traces(marker_color='#1966AD', stackgroup=None, fillcolor='#1966AD',
                      fill='none', hoveron='points+fills')
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def client_plot(product, client):

    global x, PRODUCTS

    prod = PRODUCTS[product]
    clients = {'043534': [0.002, -1.], '043563': [-0.002, 1.], '123982': [0.01, .4], '293498': [-0.01, -.5]}

    adition = clients[client]

    df_plot = pd.DataFrame()
    df_plot['Price'] = x
    df_plot['Demand'] = np.exp((prod['alpha']+adition[0])*x + prod['beta'] + adition[1])
    df_plot['Profit'] = df_plot.Price * df_plot.Demand

    product_title = product.title()
    profit_title = round(
        np.exp((prod['alpha']+adition[0])*prod["mean"] + prod['beta'] + adition[1])*prod["mean"], 2)
    best_price_title = round(
        df_plot[df_plot.Profit == df_plot.Profit.max()].Price.values[0], 2)
    best_profit_title = round(
        df_plot[df_plot.Profit == df_plot.Profit.max()].Profit.values[0], 2)
    increment = round((best_profit_title - profit_title) /
                      profit_title * 100, 2)

    max_price = df_plot[df_plot.Profit == df_plot.Profit.max()].Price.values[0]
    max_demand = df_plot[df_plot.Profit ==
                         df_plot.Profit.max()].Demand.values[0]

    mean_demand = np.exp((prod['alpha']+adition[0])*prod['mean'] + prod['beta'] + adition[1])

    title = f'Product {product_title} for Client {client} with average Price {prod["mean"]}€ and Profit {profit_title}€ -- Best Price {best_price_title}€ with Profit {best_profit_title}€ -- {increment}% Increment'

    fig = make_subplots(rows=2, cols=1)

    fig0 = go.Scatter(x=df_plot['Price'], y=df_plot['Demand'], name='demand')

    fig1 = go.Scatter(x=df_plot['Price'], y=df_plot['Profit'],
                      name='profit', line=dict(color='green'))

    fig.add_trace(fig0, row=1, col=1)
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                      'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

    fig.add_shape(type="rect",
                  x0=prod['mean']-0.1, y0=mean_demand,
                  x1=prod['mean']+0.1, y1=mean_demand,
                  line=dict(color="red", width=20,),
                  fillcolor="red",
                  row=1, col=1
                  )

    fig.add_shape(type="rect",
                  x0=max_price-0.1, y0=max_demand,
                  x1=max_price+0.1, y1=max_demand,
                  line=dict(color="green", width=20,),
                  fillcolor="green",
                  row=1, col=1
                  )
    fig.add_trace(fig1, row=2, col=1)

    fig.add_shape(type="rect",
                  x0=prod['mean']-0.1, y0=profit_title,
                  x1=prod['mean']+0.1, y1=profit_title,
                  line=dict(color="red", width=20,),
                  fillcolor="red",
                  row=2, col=1
                  )

    fig.add_shape(type="rect",
                  x0=best_price_title-0.1, y0=best_profit_title,
                  x1=best_price_title+0.1, y1=best_profit_title,
                  line=dict(color="green", width=20,),
                  fillcolor="green",
                  row=2, col=1
                  )

    fig.update_layout(title=dict(text=title)
                      )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)



PLOT_FUNCTIONS = {'demand': demand_plot,
                  'profit': profit_plot,
                  'd/p': both_plot,
                  'line_overview': line_overview,
                  'client': client_plot,
                  }
