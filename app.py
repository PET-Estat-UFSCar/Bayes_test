# -*- coding: utf-8 -*-
import numpy as np
import plotly.graph_objects as go
from scipy.stats import beta, binom, gamma, norm, invgamma
from dash import Dash, html, dcc, Input, Output, State, exceptions
import dash
import math

app = Dash(__name__)
server = app.server

# Funções de probabilidade e gráficos
def plot_beta_distribution(a, b,v="Priori"):
    if a <= 0 or b <= 0: return go.Figure(layout={"title": "Parâmetros a e b devem ser > 0", "template": "plotly_white"})
    theta = np.linspace(0, 1, 1000)
    if a == 1 and b == 1:
        beta_pdf = np.ones_like(theta)
    else:
        from scipy.stats import beta
        beta_pdf = beta.pdf(theta, a, b)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theta, y=beta_pdf, mode='lines', name=f'Beta({a}, {b})',
        line=dict(color='royalblue', width=2)
    ))
    fig.update_layout(
        title=f'{v} Beta({a}, {b})', xaxis_title='Suporte do parâmetro',
        yaxis_title='Densidade', template='plotly_white', showlegend=True
    )
    return fig

def plot_gamma_distribution(a, b,v="Priori"):
    if a <= 0 or b <= 0: return go.Figure(layout={"title": "Parâmetros a e b devem ser > 0", "template": "plotly_white"})
    if a > 1: mode = (a - 1) / b
    else: mode = 0
    x_min = max(0, mode - 3*np.sqrt(a/b**2))
    x_max = mode + 3*np.sqrt(a/b**2)
    x = np.linspace(x_min + 1e-9, x_max, 1000)
    gamma_pdf = gamma.pdf(x, a, scale=1/b)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=gamma_pdf, mode='lines', name=f'Gama({a}, {b})',
        line=dict(color='royalblue', width=2)
    ))
    fig.update_layout(
        title=f'{v} Gama({a}, {b})', xaxis_title='Suporte do parâmetro',
        yaxis_title='Densidade', template='plotly_white', showlegend=True
    )
    return fig

def plot_normal_distribution(mu, sigma2,v="Priori"):
    if sigma2 <= 0: return go.Figure(layout={"title": "Variância deve ser > 0", "template": "plotly_white"})
    sigma = np.sqrt(sigma2)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    normal_pdf = norm.pdf(x, mu, sigma)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=normal_pdf, mode='lines', name=f'Normal({mu}, {sigma2})',
        line=dict(color='royalblue', width=2)
    ))
    fig.update_layout(
        title=f'{v} Normal({mu}, {sigma2})', xaxis_title='Suporte do parâmetro',
        yaxis_title='Densidade', template='plotly_white', showlegend=True
    )
    return fig

def verossimilhanca_poisson_aproximada(x,n):
    figura=plot_gamma_distribution(round(n*x+1,3),n)
    figura.data[0].name = 'Verossim.: Poisson'
    figura.data[0].line.color = 'red'
    figura.update_layout(title="Verossimilhança reescalada da Poisson", yaxis_title="Verossimilhança", showlegend=True)
    return figura

def verossimilhanca_binomial_negativa_aproximada(x,r,n):
    figura=plot_beta_distribution(round(n*r+1,3),round(n*(x-r)+1,3))
    figura.data[0].name = 'Verossim.: Binomial Negativa'
    figura.data[0].line.color = 'red'
    figura.update_layout(title="Verossimilhança reescalada da Binomial negativa", yaxis_title="Verossimilhança", showlegend=True)
    return figura

def verossimilhanca_gama_aproximada(x,a,n):
    figura=plot_gamma_distribution(round(n*a+1,3),round(n*x,3))
    figura.data[0].name = 'Verossim.: Gama'
    figura.data[0].line.color = 'red'
    figura.update_layout(title="Verossimilhança reescalada da Gama", yaxis_title="Verossimilhança", showlegend=True)
    return figura

def verossimilhanca_exponencial_aproximada(x,n):
    figura=plot_gamma_distribution(n+1,round(n*x,3))
    figura.data[0].name = 'Verossim.: Exponencial'
    figura.data[0].line.color = 'red'
    figura.update_layout(title="Verossimilhança reescalada da Exponencial", yaxis_title="Verossimilhança", showlegend=True)
    return figura

def verossimilhanca_normal_aproximada(x,sigma2,n):
    figura=plot_normal_distribution(x,round(sigma2/n,3))
    figura.data[0].name = 'Verossim.: Normal'
    figura.data[0].line.color = 'red'
    figura.update_layout(title="Verossimilhança reescalada da Normal", yaxis_title="Verossimilhança", showlegend=True)
    return figura

def verossimilhanca_bernoulli_aproximada(x,n):
    figura=plot_beta_distribution(round(n*x+1,3),round(n*(1-x)+1,3))
    figura.data[0].name = 'Verossim.: Bernoulli'
    figura.data[0].line.color = 'red'
    figura.update_layout(title="Verossimilhança reescalada da Bernoulli", yaxis_title="Verossimilhança", showlegend=True)
    return figura

def verossimilhanca_geometrica_aproximada(x,n):
    figura=plot_beta_distribution(n+1,round(n*(x-1)+1,3))
    figura.data[0].name = 'Verossim.: Geométrica'
    figura.data[0].line.color = 'red'
    figura.update_layout(title="Verossimilhança reescalada da Geométrica", yaxis_title="Verossimilhança", showlegend=True)
    return figura

def beta_pdf(a,b):
    if a <= 0 or b <=0: return np.zeros(1000)
    theta = np.linspace(0, 1, 1000)
    if a == 1 and b == 1: return np.ones_like(theta)
    else: from scipy.stats import beta; return beta.pdf(theta, a, b)

def beta_bernoulli(a,b,x,n):
    fig = go.Figure()
    eixo_x=np.linspace(0,1,1000)
    y_priori=beta_pdf(a,b)
    y_verossimilhanca=beta_pdf(round(n*x+1,3),round(n*(1-x)+1,3))
    y_posteriori=beta_pdf(round(a+n*x,3),round(b+n*(1-x),3))
    fig.add_trace(go.Scatter(x=eixo_x, y=y_priori, mode='lines', name=f'Priori: Beta({a}, {b})', line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=eixo_x, y=y_verossimilhanca, mode='lines', name='Verossim.: Bernoulli', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=eixo_x, y=y_posteriori, mode='lines', name=f'Posteriori: Beta({round(a+n*x,3)}, {round(b+n*(1-x),3)})', line=dict(color='green', width=2)))
    fig.update_layout(title='Distribuição Beta-Bernoulli', xaxis_title='Suporte do parâmetro', yaxis_title='Densidade', template='plotly_white')
    return fig

def beta_geometrica(a,b,x,n):
    fig = go.Figure()
    eixo_x=np.linspace(0,1,1000)
    y_priori=beta_pdf(a,b)
    y_verossimilhanca=beta_pdf(n+1,round(n*(x-1)+1,3))
    y_posteriori=beta_pdf(a+n,round(b+n*(x-1),3))
    fig.add_trace(go.Scatter(x=eixo_x, y=y_priori, mode='lines', name=f'Priori: Beta({a}, {b})', line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=eixo_x, y=y_verossimilhanca, mode='lines', name='Verossim.: Geométrica', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=eixo_x, y=y_posteriori, mode='lines', name=f'Posteriori: Beta({a+n}, {round(b+n*(x-1),3)})', line=dict(color='green', width=2)))
    fig.update_layout(title='Distribuição Beta-Geométrica', xaxis_title='Suporte do parâmetro', yaxis_title='Densidade', template='plotly_white')
    return fig

def beta_binomial_negativa(a,b,x,r,n):
    fig = go.Figure()
    eixo_x=np.linspace(0,1,1000)
    y_priori=beta_pdf(a,b)
    y_verossimilhanca=beta_pdf(round(n*r+1,3),round(n*(x-r)+1,3))
    y_posteriori=beta_pdf(round(a+n*r,3),round(b+n*(x-r),3))
    fig.add_trace(go.Scatter(x=eixo_x, y=y_priori, mode='lines', name=f'Priori: Beta({a}, {b})', line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=eixo_x, y=y_verossimilhanca, mode='lines', name='Verossim.: Binomial Negativa', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=eixo_x, y=y_posteriori, mode='lines', name=f'Posteriori: Beta({round(a+n*r,3)}, {round(b+n*(x-r),3)})', line=dict(color='green', width=2)))
    fig.update_layout(title='Distribuição Beta-Binomial negativa', xaxis_title='Suporte do parâmetro', yaxis_title='Densidade', template='plotly_white')
    return fig

def gama_pdf(a, b):
    if a > 1: mode = (a - 1) / b
    else: mode = 0
    x_min = max(0, mode - 3*np.sqrt(a/b**2))
    x_max = mode + 3*np.sqrt(a/b**2)
    return x_min,x_max

def gama_exponencial(a,b,x,n):
    x_min_priori,x_max_priori=gama_pdf(a,b)
    x_min_verossimilhanca,x_max_verossimilhanca=gama_pdf(n+1,n*x)
    x_min_posteriori,x_max_posteriori=gama_pdf(a+n,b+n*x)
    x_min=min(x_min_priori,x_min_verossimilhanca,x_min_posteriori)
    x_max=max(x_max_priori,x_max_verossimilhanca,x_max_posteriori)
    eixo_x=np.linspace(x_min,x_max,1000)
    y_priori=gamma.pdf(eixo_x,a,scale=1/b)
    y_verossimilhanca=gamma.pdf(eixo_x,n+1,scale=round(1/(n*x),3))
    y_posteriori=gamma.pdf(eixo_x,a+n,scale=round(1/(b+n*x),3))
    figura=go.Figure()
    figura.add_trace(go.Scatter(x=eixo_x, y=y_priori, mode='lines', name=f'Priori: Gama({a}, {b})', line=dict(color='royalblue', width=2)))
    figura.add_trace(go.Scatter(x=eixo_x, y=y_verossimilhanca, mode='lines', name='Verossim.: Exponencial', line=dict(color='red', width=2)))
    figura.add_trace(go.Scatter(x=eixo_x, y=y_posteriori, mode='lines', name=f'Posteriori: Gama({a+n}, {round(b+n*x,3)})', line=dict(color='green', width=2)))
    figura.update_layout(title='Distribuição Gama-Exponencial', xaxis_title='Suporte do parâmetro', yaxis_title='Densidade', template='plotly_white', showlegend=True)
    return figura

def gama_poisson(a,b,x,n):
    x_min_priori,x_max_priori=gama_pdf(a,b)
    x_min_verossimilhanca,x_max_verossimilhanca=gama_pdf(n*x+1,n)
    x_min_posteriori,x_max_posteriori=gama_pdf(n*x+a,b+n)
    x_min=min(x_min_priori,x_min_verossimilhanca,x_min_posteriori)
    x_max=max(x_max_priori,x_max_verossimilhanca,x_max_posteriori)
    eixo_x=np.linspace(x_min,x_max,1000)
    y_priori=gamma.pdf(eixo_x,a,scale=1/b)
    y_verossimilhanca=gamma.pdf(eixo_x,round(n*x+1,3),scale=round(1/(n),3))
    y_posteriori=gamma.pdf(eixo_x,round(n*x+a,3),scale=round(1/(b+n),3))
    figura=go.Figure()
    figura.add_trace(go.Scatter(x=eixo_x, y=y_priori, mode='lines', name=f'Priori: Gama({a}, {b})', line=dict(color='royalblue', width=2)))
    figura.add_trace(go.Scatter(x=eixo_x, y=y_verossimilhanca, mode='lines', name='Verossim.: Poisson', line=dict(color='red', width=2)))
    figura.add_trace(go.Scatter(x=eixo_x, y=y_posteriori, mode='lines', name=f'Posteriori: Gama({round(n*x+a,3)}, {b+n})', line=dict(color='green', width=2)))
    figura.update_layout(title='Distribuição Gama-Poisson', xaxis_title='Suporte do parâmetro', yaxis_title='Densidade', template='plotly_white', showlegend=True)
    return figura

def gama_gama(a,b,x,conhecido,n):
    x_min_priori,x_max_priori=gama_pdf(a,b)
    x_min_verossimilhanca,x_max_verossimilhanca=gama_pdf(n*conhecido+1,n*x)
    x_min_posteriori,x_max_posteriori=gama_pdf(a+n*conhecido,b+n*x)
    x_min=min(x_min_priori,x_min_verossimilhanca,x_min_posteriori)
    x_max=max(x_max_priori,x_max_verossimilhanca,x_max_posteriori)
    eixo_x=np.linspace(x_min,x_max,1000)
    y_priori=gamma.pdf(eixo_x,a,scale=1/b)
    y_verossimilhanca=gamma.pdf(eixo_x,round(n*conhecido+1,3),scale=round(1/(n*x),3))
    y_posteriori=gamma.pdf(eixo_x,round(a+n*conhecido,3),scale=round(1/(b+n*x),3))
    figura=go.Figure()
    figura.add_trace(go.Scatter(x=eixo_x, y=y_priori, mode='lines', name=f'Priori: Gama({a}, {b})', line=dict(color='royalblue', width=2)))
    figura.add_trace(go.Scatter(x=eixo_x, y=y_verossimilhanca, mode='lines', name='Verossim.: Gama', line=dict(color='red', width=2)))
    figura.add_trace(go.Scatter(x=eixo_x, y=y_posteriori, mode='lines', name=f'Posteriori: Gama({round(a+n*conhecido,3)}, {round(b+n*x,3)})', line=dict(color='green', width=2)))
    figura.update_layout(title='Distribuição Gama-Gama', xaxis_title='Suporte do parâmetro', yaxis_title='Densidade', template='plotly_white', showlegend=True)
    return figura

def normal_pdf(mu,sigma2):
    if sigma2 <= 0: return mu-4, mu+4
    sigma = np.sqrt(sigma2)
    return mu - 4*sigma,  mu + 4*sigma

def normal_normal(mu,sigma2,x,conhecido,n):
    x_min_priori,x_max_priori=normal_pdf(mu,sigma2)
    x_min_verossimilhanca,x_max_verossimilhanca=normal_pdf(x,conhecido/n)
    x_min_posteriori,x_max_posteriori=normal_pdf((n*sigma2*x+conhecido*mu)/(n*sigma2+conhecido),sigma2*conhecido/(n*sigma2+conhecido))
    x_min=min(x_min_priori,x_min_verossimilhanca,x_min_posteriori)
    x_max=max(x_max_priori,x_max_verossimilhanca,x_max_posteriori)
    eixo_x=np.linspace(x_min,x_max,1000)
    y_priori=norm.pdf(eixo_x,mu,np.sqrt(sigma2))
    y_verossimilhanca=norm.pdf(eixo_x,x,round(np.sqrt(conhecido/n),3))
    y_posteriori=norm.pdf(eixo_x,round((n*sigma2*x+conhecido*mu)/(n*sigma2+conhecido),3),round(np.sqrt(sigma2*conhecido/(n*sigma2+conhecido)),3))
    figura = go.Figure()
    figura.add_trace(go.Scatter(x=eixo_x, y=y_priori, mode='lines', name=f'Priori: Normal({mu}, {sigma2})', line=dict(color='royalblue', width=2)))
    figura.add_trace(go.Scatter(x=eixo_x, y=y_verossimilhanca, mode='lines', name='Verossim.: Normal', line=dict(color='red', width=2)))
    figura.add_trace(go.Scatter(x=eixo_x, y=y_posteriori, mode='lines', name=f'Posteriori: Normal({round((n*sigma2*x+conhecido*mu)/(n*sigma2+conhecido),3)}, {round(sigma2*conhecido/(n*sigma2+conhecido),3)})', line=dict(color='green', width=2)))
    figura.update_layout(title='Distribuição Normal-Normal', xaxis_title='Suporte do parâmetro', yaxis_title='Densidade', template='plotly_white')
    return figura

def posteriori_beta(a,b,v="Posteriori"):
    figura=plot_beta_distribution(a,b,v)
    figura.data[0].line.color = 'green'
    return figura

def posteriori_gama(a,b,v="Posteriori"):
    figura=plot_gamma_distribution(a,b,v)
    figura.data[0].line.color = 'green'
    return figura

def posteriori_normal(a,b,v="Posteriori"):
    figura=plot_normal_distribution(a,b,v)
    figura.data[0].line.color = 'green'
    return figura

def normal_gama_pdf(x, tau, mu, lambda_, alpha, beta):
    tau = np.maximum(tau, 1e-9)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        term1 = (beta**alpha)
        term2 = np.sqrt(lambda_)
        term3 = tau**(alpha-0.5)
        term4 = np.exp(-beta*tau)
        term5 = np.exp(-lambda_*tau*(x-mu)**2/2)
        denominador = (math.gamma(alpha)*np.sqrt(2*np.pi))
        densidade = (term1 * term2 * term3 * term4 * term5) / denominador
    return np.nan_to_num(densidade)


colorscale_blue = [[0.0, "lightblue"],[0.5, "blue"],[1.0, "darkblue"]]
colorscale_red = [[0.0, "lightcoral"],[0.5, "red"],[1.0, "darkred"]]
colorscale_green = [[0.0, "lightgreen"],[0.5, "green"],[1.0, "darkgreen"]]

def plot_normal_gama_distribution(mu,lambda_,alpha,beta,cor=colorscale_blue,v="Priori"):
    if alpha <= 1 or lambda_ <= 0 or beta <=0: return go.Figure(layout={"title": f"Parâmetros Inválidos para {v} Normal-Gama", "template": "plotly_white"})
    mode_tau=(alpha-0.5)/beta
    desvio_x=np.sqrt(beta/(lambda_*(alpha-1)))
    desvio_tau=np.sqrt(alpha/beta**2)
    x_vals=np.linspace(mu-3*desvio_x,mu+3*desvio_x,100)
    tau_vals=np.linspace(max(mode_tau-3*desvio_tau,0.001),mode_tau+3*desvio_tau,100)
    X, T = np.meshgrid(x_vals, tau_vals)
    Z = normal_gama_pdf(X, T, mu, lambda_, alpha, beta)
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=T, colorscale=cor)])
    fig.update_layout(title=f"{v} Normal-Gama({round(mu,3)}, {round(lambda_,3)}, {round(alpha,3)}, {round(beta,3)})",
                        scene=dict(xaxis_title="μ", yaxis_title="τ", zaxis_title="Densidade"))
    return fig

def verossimilhanca_normal_gama_aproximada(x,s,n):
    alpha_vero = (n-1)/2
    beta_vero = n*s/2
    if alpha_vero <= 0 or beta_vero <= 0: return go.Figure(layout={"title": "Parâmetros da Verossimilhança Inválidos", "template": "plotly_white"})
    figura=plot_normal_gama_distribution(x,n,alpha_vero, beta_vero ,colorscale_red, "Verossimilhança")
    figura.update_layout(title="Verossimilhança aproximada da Normal",
                            scene=dict(xaxis_title="μ", yaxis_title="τ", zaxis_title="Verossimilhança"))
    return figura

def posteriori_Normal_Gama(mu,lambda_,alpha,beta,x,s,n,v="Posteriori"):
    mu_post = (lambda_*mu+n*x)/(lambda_+n)
    lambda_post = lambda_+n
    alpha_post = alpha+n/2
    beta_post = beta+(n*s+(lambda_*n*(x-mu)**2)/(lambda_+n))/2
    if alpha_post <= 1: return go.Figure(layout={"title": "Parâmetros da Posteriori Inválidos", "template": "plotly_white"})
    figura=plot_normal_gama_distribution(mu_post, lambda_post, alpha_post, beta_post, colorscale_green,v)
    return figura

def xvals_tauvals(mu,lambda_,alpha,beta):
    if alpha <= 1 or lambda_ <= 0 or beta <= 0: return (np.linspace(mu-3, mu+3, 100), np.linspace(0.001, 3, 100))
    mode_tau=(alpha-0.5)/beta
    desvio_x=np.sqrt(beta/(lambda_*(alpha-1)))
    desvio_tau=np.sqrt(alpha/beta**2)
    x_vals=np.linspace(mu-3*desvio_x,mu+3*desvio_x,100)
    tau_vals=np.linspace(max(mode_tau-3*desvio_tau,0.001),mode_tau+3*desvio_tau,100)
    return x_vals,tau_vals

def Normal_Gama_final(mu,lambda_,alpha,beta,x,s,n):
    if any(p is None for p in [mu, lambda_, alpha, beta, x, s, n]): return go.Figure(layout={"title": "Aguardando todos os parâmetros", "template": "plotly_white"})
    if any(p <= 0 for p in [lambda_, alpha, beta, n, s]) or alpha <= 1: return go.Figure(layout={"title": "Parâmetros da Priori Inválidos", "template": "plotly_white"})
    
    mu_post = (lambda_*mu+n*x)/(lambda_+n)
    lambda_post = lambda_+n
    alpha_post = alpha+n/2
    beta_post = beta+(n*s+(lambda_*n*(x-mu)**2)/(lambda_+n))/2

    if alpha_post <=1 : return go.Figure(layout={"title": "Parâmetros da Posteriori Inválidos", "template": "plotly_white"})
    if (n-1)/2 <=0 : return go.Figure(layout={"title": "Parâmetros da Verossimilhança Inválidos", "template": "plotly_white"})

    x_priori,tau_priori=xvals_tauvals(mu,lambda_,alpha,beta)
    x_verossimilhanca,tau_verossimilhanca=xvals_tauvals(x,n,(n-1)/2,n*s/2)
    x_posteriori,tau_posteriori=xvals_tauvals(mu_post,lambda_post,alpha_post,beta_post)

    x_min_val = min(x_priori[0],x_verossimilhanca[0],x_posteriori[0])
    x_max_val = max(x_priori[-1],x_verossimilhanca[-1],x_posteriori[-1])
    tau_min_val = min(tau_priori[0],tau_verossimilhanca[0],tau_posteriori[0])
    tau_max_val = max(tau_priori[-1],tau_verossimilhanca[-1],tau_posteriori[-1])

    x_final=np.linspace(x_min_val, x_max_val, 70)
    tau_final=np.linspace(tau_min_val, tau_max_val, 70)

    X, T = np.meshgrid(x_final, tau_final)
    Z_priori = normal_gama_pdf(X, T, mu, lambda_, alpha, beta)
    Z_verossimilhanca = normal_gama_pdf(X, T, x, n, (n-1)/2, n*s/2)
    Z_posteriori = normal_gama_pdf(X, T, mu_post, lambda_post, alpha_post, beta_post)

    fig = go.Figure(go.Surface(z=Z_priori, x=X, y=T, colorscale=colorscale_blue,name="Priori",showscale=False))
    fig.add_trace(go.Surface(z=Z_verossimilhanca, x=X, y=T, colorscale=colorscale_red,name="Verossimilhança",showscale=False, opacity=0.7))
    fig.add_trace(go.Surface(z=Z_posteriori, x=X, y=T, colorscale=colorscale_green,name="Posteriori",showscale=False, opacity=0.7))

    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(color='blue'), name="Priori"))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(color='red'), name="Verossimilhança"))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(color='green'), name="Posteriori"))

    fig.update_layout(title="Distribuição Normal-Gama", showlegend=True,
                        scene=dict(xaxis_title="μ", yaxis_title="τ", zaxis_title="Densidade"))
    return fig


lista_prioris=list(["Beta","Gama","Normal","Normal-Gama"])

# --- ESTILOS ---
colors = {'background': '#F3F6FA', 'text': '#333333', 'primary': '#007BFF', 'light_grey': '#E9ECEF'}
header_style = {"backgroundColor": "white", "padding": "20px", "textAlign": "center", "borderBottom": f"1px solid {colors['light_grey']}", "display": "flex", "alignItems": "center", "justifyContent": "space-between", "boxShadow": "0 2px 4px rgba(0,0,0,0.05)"}
logo_style = {"height": "60px"}
nav_buttons_style = {"display": "flex", "gap": "15px"}
main_content_style = {"padding": "30px", "backgroundColor": colors['background'], "fontFamily": "Roboto, sans-serif"}
card_style = {"backgroundColor": "white", "padding": "20px", "borderRadius": "10px", "boxShadow": "0 2px 4px rgba(0,0,0,0.05)", "marginBottom": "20px"}

# --- LAYOUT ---
header = html.Div(style=header_style, children=[
    html.Img(src="/assets/logopet.png", style=logo_style),
    html.H1("Dashboard de Análise Bayesiana", style={'color': colors['text'], 'fontSize': '24px', 'margin': '0'}),
    html.Div(style=nav_buttons_style, children=[
        html.Button("Análise de Conjugadas", id="btn-conjugadas", n_clicks=0, className="nav-button"),
        html.Button("Teorema de Bayes", id="btn-teorema", n_clicks=0, className="nav-button")
    ])
])

app.layout = html.Div([
    header,
    html.Div(id="content-div", style=main_content_style)
])

def layout_teorema():
    return html.Div(className="container", children=[
        html.Div(style=card_style, children=[html.H2("Teorema de Bayes — Quadrado de Bayes (2 hipóteses)", style={'textAlign': 'center'})]),
        html.Div(className="row", children=[
            html.Div(className="col", children=[
                html.Div(style=card_style, children=[
                    html.Label("Probabilidade a priori A:"),
                    dcc.Slider(id='pa_slider', min=0, max=1, step=0.01, value=0.5, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                    dcc.Input(id='pa_input', type='number', value=0.5, min=0, max=1, step=0.01, className="styled-input"),
                    html.Hr(),
                    html.Label("Probabilidade condicional E|A:"),
                    dcc.Slider(id='pea_slider', min=0, max=1, step=0.01, value=0.7, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                    dcc.Input(id='pea_input', type='number', value=0.7, min=0, max=1, step=0.01, className="styled-input"),
                    html.Hr(),
                    html.Label("Probabilidade condicional E|B:"),
                    dcc.Slider(id='peb_slider', min=0, max=1, step=0.01, value=0.4, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                    dcc.Input(id='peb_input', type='number', value=0.4, min=0, max=1, step=0.01, className="styled-input"),
                ])
            ]),
            html.Div(className="col", children=[
                html.Div(style=card_style, children=[dcc.Graph(id='bayes_graph')]),
                html.Div(id='posterior_text', style={**card_style, 'textAlign': 'center', 'fontWeight': 'bold'}),
            ])
        ])
    ])

def layout_conjugadas():
    return html.Div(children=[
        html.Div(style=card_style, children=[
            html.H2('Análise de Distribuições Conjugadas', style={'textAlign': 'center'}),
            html.P('Selecione as distribuições da Priori e do Modelo para visualizar a Posteriori resultante.', style={'textAlign': 'center'})
        ]),
        html.Div(className="row", children=[
            html.Div(className="col", children=[
                html.Div(style=card_style, children=[
                    html.H4('Configuração da Priori'),
                    html.Label('Distribuição da Priori:'),
                    dcc.Dropdown(lista_prioris, value="Beta", id="prioris"),
                    html.Div(id="priori-params-div", style={'marginTop': '15px'}),
                ]),
                html.Div(style=card_style, children=[
                    html.H4('Configuração da Verossimilhança'),
                    html.Label('Modelo Estatístico:'),
                    dcc.Dropdown(id="verossimilhancas"),
                    html.Div(id="verossimilhanca-params-div", style={'marginTop': '15px'}),
                ]),
            ]),
            html.Div(className="col", children=[
                html.Div(style=card_style, children=[dcc.Graph(id='densidade_priori')]),
                html.Div(style=card_style, id='aparencia_verossimilhanca', children=[dcc.Graph(id='densidade_verossimilhanca')]),
                html.Div(style=card_style, children=[dcc.Graph(id='densidade_posteriori')]),
            ])
        ]),
        html.Div(style=card_style, children=[dcc.Graph(id="grafico_conjunto")]),
        html.Div(style=card_style, children=[
            html.Button("Ver Fórmulas Gerais", id="botao", n_clicks=0, className="action-button"),
            html.Div(id="texto_formula_div")
        ])
    ])

# --- CALLBACKS ---

# Callback para navegação entre páginas
@app.callback(
    Output("content-div", "children"),
    [Input("btn-conjugadas", "n_clicks"), Input("btn-teorema", "n_clicks")]
)
def display_page(n_clicks_conjugadas, n_clicks_teorema):
    ctx = dash.callback_context
    if not ctx.triggered:
        return layout_conjugadas()
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "btn-teorema":
        return layout_teorema()
    else:
        return layout_conjugadas()

# Callback para atualizar o dropdown de verossimilhança
@app.callback(
    [Output("verossimilhancas", "options"),
     Output("verossimilhancas", "value")],
    Input("prioris", "value")
)
def update_verossimilhanca_dropdown(priori):
    if priori == "Beta":
        # REMOVIDA A OPÇÃO "Binomial" CONFORME SOLICITADO
        options = ["Bernoulli", "Geométrica", "Binomial negativa"]
        return options, options[0]
    elif priori == "Gama":
        options = ["Exponencial", "Poisson", "Gama (b desconhecido)"]
        return options, options[0]
    elif priori == "Normal-Gama":
        options = ["Normal (média e precisão desconhecidas)"]
        return options, options[0]
    elif priori == "Normal":
        options = ["Normal (média desconhecida)"]
        return options, options[0]
    return [], None

# Callbacks para renderizar os campos de parâmetros
@app.callback(
    Output("priori-params-div", "children"),
    Input("prioris", "value")
)
def render_priori_params(priori):
    if priori == "Beta":
        return html.Div([
            html.Label("Parâmetro de forma a (> 0):"),
            dcc.Input(id='input-a', type='number', value=2, min=0.001, className="styled-input"),
            html.Div(id='error-msg-a', className="error-message"),
            html.Label("Parâmetro de forma b (> 0):"),
            dcc.Input(id='input-b', type='number', value=4, min=0.001, className="styled-input"),
            html.Div(id='error-msg-b', className="error-message"),
            # Hidden inputs to maintain callback structure and prevent crashes
            html.Div(dcc.Input(id='input-c', value=3), style={'display':'none'}),
            html.Div(dcc.Input(id='input-d', value=2), style={'display':'none'})
        ])
    elif priori == "Gama":
        return html.Div([
            html.Label("Parâmetro de forma α (> 0):"),
            dcc.Input(id='input-a', type='number', value=2, min=0.001, className="styled-input"),
            html.Div(id='error-msg-a', className="error-message"),
            html.Label("Parâmetro de taxa β (> 0):"),
            dcc.Input(id='input-b', type='number', value=4, min=0.001, className="styled-input"),
            html.Div(id='error-msg-b', className="error-message"),
            html.Div(dcc.Input(id='input-c', value=3), style={'display':'none'}),
            html.Div(dcc.Input(id='input-d', value=2), style={'display':'none'})
        ])
    elif priori == "Normal":
        return html.Div([
            html.Label("Média da priori μ (ℝ):"),
            dcc.Input(id='input-a', type='number', value=0, className="styled-input"),
            html.Div(id='error-msg-a', className="error-message"),
            html.Label("Variância da priori σ² (> 0):"),
            dcc.Input(id='input-b', type='number', value=1, min=0.001, className="styled-input"),
            html.Div(id='error-msg-b', className="error-message"),
            html.Div(dcc.Input(id='input-c', value=3), style={'display':'none'}),
            html.Div(dcc.Input(id='input-d', value=2), style={'display':'none'})
        ])
    elif priori == "Normal-Gama":
        return html.Div([
            html.Label("Parâmetro μ da priori (ℝ):"),
            dcc.Input(id='input-a', type='number', value=0, className="styled-input"),
            html.Div(id='error-msg-a', className="error-message"),
            html.Label("Parâmetro λ da priori (> 0):"),
            dcc.Input(id='input-b', type='number', value=1, min=0.001, className="styled-input"),
            html.Div(id='error-msg-b', className="error-message"),
            html.Label("Parâmetro de forma α (> 1):"),
            dcc.Input(id='input-c', type='number', value=3, min=1.001, className="styled-input"),
            html.Div(id='error-msg-c', className="error-message"),
            html.Label("Parâmetro de taxa β (> 0):"),
            dcc.Input(id='input-d', type='number', value=2, min=0.001, className="styled-input"),
            html.Div(id='error-msg-d', className="error-message"),
        ])
    return []

@app.callback(
    Output("verossimilhanca-params-div", "children"),
    Input("verossimilhancas", "value")
)
def render_verossimilhanca_params(verossimilhanca):
    if verossimilhanca is None:
        raise exceptions.PreventUpdate

    tamanho_min = 2 if verossimilhanca == "Normal (média e precisão desconhecidas)" else 1
    
    # Base inputs definition
    base_inputs = [
        html.Label(f"Tamanho amostral (n ≥ {tamanho_min}):"),
        dcc.Input(id='input-tamanho', type='number', step=1, value=10, className="styled-input"),
        html.Div(id='error-msg-tamanho', className="error-message"),
    ]

    # Reusable component definitions
    input_x = dcc.Input(id='input-x', type='number', value=1, className="styled-input")
    input_x_bernoulli = dcc.Input(id='input-x-bernoulli', type='number', step=0.01, min=0, max=1, value=0.5, className="styled-input")
    input_m = dcc.Input(id='input-m', type='number', min=1, step=1, value=10, className="styled-input")
    input_conhecido = dcc.Input(id='input-conhecido', type='number', value=1, min=0.001, className="styled-input")

    if verossimilhanca == "Bernoulli":
        return html.Div([
            html.Label("Média amostral (0 ≤ x̄ ≤ 1):"),
            input_x_bernoulli,
            html.Div(id='error-msg-x-bernoulli', className="error-message"),
            *base_inputs,
            # Hidden inputs for compatibility
            html.Div(input_x, style={'display': 'none'}),
            html.Div(input_m, style={'display': 'none'}),
            html.Div(input_conhecido, style={'display': 'none'}),
        ])

    specific_section, hidden_inputs = [], []
    media_label = "Média amostral (x̄):"
    
    if verossimilhanca == "Binomial negativa":
        media_label = "Média amostral (x̄ > r):"
        specific_section = [html.Label("Número de sucessos (r ≥ 1):"), input_m, html.Div(id='error-msg-m', className="error-message")]
        hidden_inputs = [html.Div(input_conhecido, style={'display': 'none'})]
    elif verossimilhanca == "Gama (b desconhecido)":
        media_label = "Média amostral (x̄ ≥ 0):"
        specific_section = [html.Label("Parâmetro 'a' conhecido: (> 0)"), input_conhecido, html.Div(id='error-msg-conhecido', className="error-message")]
        hidden_inputs = [html.Div(input_m, style={'display': 'none'})]
    elif verossimilhanca == "Normal (média desconhecida)":
        media_label = "Média amostral (x̄ ∈ ℝ):"
        specific_section = [html.Label("Variância populacional (σ²) conhecida: (> 0)"), input_conhecido, html.Div(id='error-msg-conhecido', className="error-message")]
        hidden_inputs = [html.Div(input_m, style={'display': 'none'})]
    elif verossimilhanca == "Normal (média e precisão desconhecidas)":
        media_label = "Média amostral (x̄ ∈ ℝ):"
        specific_section = [html.Label("Variância amostral (s² > 0):"), input_conhecido, html.Div(id='error-msg-conhecido', className="error-message")]
        hidden_inputs = [html.Div(input_m, style={'display': 'none'})]
    else:
        hidden_inputs = [html.Div(input_m, style={'display': 'none'}), html.Div(input_conhecido, style={'display': 'none'})]
        if verossimilhanca == "Geométrica":
            media_label = "Média amostral (x̄ ≥ 1):"
        elif verossimilhanca in ["Exponencial", "Poisson"]:
            media_label = "Média amostral (x̄ ≥ 0):"

    return html.Div([
        html.Label(media_label), input_x,
        html.Div(id='error-msg-x', className="error-message"),
        *specific_section,
        *base_inputs,
        html.Div(input_x_bernoulli, style={'display': 'none'}),
        *hidden_inputs
    ])

# --- CALLBACKS DE VALIDAÇÃO VISUAL ---
def validate_param(value, min_val=None, max_val=None, min_exclusive=False, max_exclusive=False):
    invalid_style = {'borderColor': 'red'}
    valid_style = {'borderColor': ''}
    if value is None:
        return invalid_style, 'Campo obrigatório.'
    
    if min_val is not None:
        if min_exclusive and value <= min_val:
            return invalid_style, f'Valor deve ser > {min_val}.'
        if not min_exclusive and value < min_val:
            return invalid_style, f'Valor deve ser ≥ {min_val}.'

    if max_val is not None:
        if max_exclusive and value >= max_val:
            return invalid_style, f'Valor deve ser < {max_val}.'
        if not max_exclusive and value > max_val:
            return invalid_style, f'Valor deve ser ≤ {max_val}.'
            
    return valid_style, ''

@app.callback(
    [Output('input-a', 'style'), Output('error-msg-a', 'children')],
    Input('input-a', 'value'),
    State('prioris', 'value'),
    prevent_initial_call=True
)
def validate_input_a(value, priori):
    if priori in ["Beta", "Gama"]:
        return validate_param(value, min_val=0, min_exclusive=True)
    return {'borderColor': ''}, ''

@app.callback(
    [Output('input-b', 'style'), Output('error-msg-b', 'children')],
    Input('input-b', 'value'),
    State('prioris', 'value'),
    prevent_initial_call=True
)
def validate_input_b(value, priori):
    if priori in ["Beta", "Gama", "Normal", "Normal-Gama"]:
        return validate_param(value, min_val=0, min_exclusive=True)
    return {'borderColor': ''}, ''

@app.callback(
    [Output('input-c', 'style'), Output('error-msg-c', 'children')],
    Input('input-c', 'value'),
    State('prioris', 'value'),
    prevent_initial_call=True
)
def validate_input_c(value, priori):
    if priori == "Normal-Gama":
        return validate_param(value, min_val=1, min_exclusive=True)
    return {'borderColor': ''}, ''

@app.callback(
    [Output('input-d', 'style'), Output('error-msg-d', 'children')],
    Input('input-d', 'value'),
    State('prioris', 'value'),
    prevent_initial_call=True
)
def validate_input_d(value, priori):
    if priori == "Normal-Gama":
        return validate_param(value, min_val=0, min_exclusive=True)
    return {'borderColor': ''}, ''

@app.callback(
    [Output('input-tamanho', 'style'), Output('error-msg-tamanho', 'children')],
    Input('input-tamanho', 'value'),
    State('verossimilhancas', 'value'),
    prevent_initial_call=True
)
def validate_input_tamanho(value, verossimilhanca):
    if verossimilhanca is None: return {'borderColor': ''}, ''
    min_val = 2 if verossimilhanca == "Normal (média e precisão desconhecidas)" else 1
    return validate_param(value, min_val=min_val)

@app.callback(
    [Output('input-x-bernoulli', 'style'), Output('error-msg-x-bernoulli', 'children')],
    Input('input-x-bernoulli', 'value'),
    State('verossimilhancas', 'value'),
    prevent_initial_call=True
)
def validate_input_x_bernoulli(value, verossimilhanca):
    if verossimilhanca == "Bernoulli":
        return validate_param(value, min_val=0, max_val=1)
    return {'borderColor': ''}, ''

@app.callback(
    [Output('input-conhecido', 'style'), Output('error-msg-conhecido', 'children')],
    Input('input-conhecido', 'value'),
    State('verossimilhancas', 'value'),
    prevent_initial_call=True
)
def validate_input_conhecido(value, verossimilhanca):
    if verossimilhanca in ["Gama (b desconhecido)", "Normal (média desconhecida)", "Normal (média e precisão desconhecidas)"]:
        return validate_param(value, min_val=0, min_exclusive=True)
    return {'borderColor': ''}, ''
    
@app.callback(
    [Output('input-m', 'style'), Output('error-msg-m', 'children')],
    Input('input-m', 'value'),
    State('verossimilhancas', 'value'),
    prevent_initial_call=True
)
def validate_input_m(value, verossimilhanca):
    if verossimilhanca in ["Binomial negativa"]:
        return validate_param(value, min_val=1)
    return {'borderColor': ''}, ''

@app.callback(
    [Output('input-x', 'style'), Output('error-msg-x', 'children')],
    [Input('input-x', 'value'), Input('input-m', 'value')],
    State('verossimilhancas', 'value'),
    prevent_initial_call=True
)
def validate_input_x_and_m(x, m, verossimilhanca):
    if verossimilhanca is None: return {'borderColor': ''}, ''
    
    ctx = dash.callback_context
    if not ctx.triggered: return {'borderColor': ''}, ''
    
    triggered_id_prop = ctx.triggered[0]['prop_id']
    
    if x is None:
        if 'input-x.value' in triggered_id_prop:
            return {'borderColor': 'red'}, 'Campo obrigatório.'
        else:
            return dash.no_update
    
    invalid_style = {'borderColor': 'red'}
    valid_style = {'borderColor': ''}

    if verossimilhanca == "Binomial negativa":
        style, msg = validate_param(x, min_val=0)
        if msg != '': return style, msg
        if m is not None and x <= m:
            return invalid_style, f'Média (x̄) deve ser > r ({m}).'
        return valid_style, ''

    elif verossimilhanca == "Geométrica":
        return validate_param(x, min_val=1)
        
    elif verossimilhanca in ["Exponencial", "Poisson", "Gama (b desconhecido)"]:
        return validate_param(x, min_val=0)

    return valid_style, ''

# --- FIM DOS CALLBACKS DE VALIDAÇÃO ---

# Callbacks para Teorema de Bayes
@app.callback(Output('pa_slider', 'value'), Input('pa_input', 'value'))
def sync_slider_pa(val): return val
@app.callback(Output('pea_slider', 'value'), Input('pea_input', 'value'))
def sync_slider_pea(val): return val
@app.callback(Output('peb_slider', 'value'), Input('peb_input', 'value'))
def sync_slider_peb(val): return val
@app.callback(Output('pa_input', 'value'), Input('pa_slider', 'value'))
def sync_input_pa(val): return val
@app.callback(Output('pea_input', 'value'), Input('pea_slider', 'value'))
def sync_input_pea(val): return val
@app.callback(Output('peb_input', 'value'), Input('peb_slider', 'value'))
def sync_input_peb(val): return val

@app.callback(
    [Output('bayes_graph', 'figure'), Output('posterior_text', 'children')],
    [Input('pa_input', 'value'), Input('pea_input', 'value'), Input('peb_input', 'value')]
)
def update_teorema(PA, PEA, PEB):
    if PA is None or PEA is None or PEB is None: raise exceptions.PreventUpdate
    PB = 1 - PA
    denom = PEA * PA + PEB * PB
    PAE = (PEA * PA) / denom if denom != 0 else 0
    PBE = (PEB * PB) / denom if denom != 0 else 0
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=1, y0=0, y1=1, line=dict(color="black"))
    fig.add_shape(type="rect", x0=0, x1=PEA, y0=0, y1=PA, fillcolor=colors['primary'], opacity=0.7, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=PEB, y0=PA, y1=1, fillcolor="green", opacity=0.7, line_width=0)
    fig.update_layout(
        xaxis=dict(range=[0, 1], showticklabels=False), yaxis=dict(range=[0, 1], showticklabels=False),
        width=400, height=400, title="Quadrado de Bayes", showlegend=False, margin=dict(l=20, r=20, t=40, b=20)
    )
    text = f"Posterior A: {PAE:.2f} | Posterior B: {PBE:.2f}"
    return fig, text

def validate_all_params(a, b, c, d, m, x, x_bernoulli, n, conhecido, prioris, verossimilhancas):
    # Check for None values based on which inputs should exist
    if any(p is None for p in [a, b, n, prioris, verossimilhancas]): return False
    
    if prioris == "Normal-Gama" and any(p is None for p in [c, d]): return False
    
    if verossimilhancas == "Bernoulli" and x_bernoulli is None: return False
    if verossimilhancas == "Binomial negativa" and (x is None or m is None): return False
    if verossimilhancas in ["Geométrica", "Exponencial", "Poisson"] and x is None: return False
    if verossimilhancas in ["Gama (b desconhecido)", "Normal (média desconhecida)", "Normal (média e precisão desconhecidas)"] and (x is None or conhecido is None): return False

    # Priori validation
    if prioris == "Beta" and (a <= 0 or b <= 0): return False
    if prioris == "Gama" and (a <= 0 or b <= 0): return False
    if prioris == "Normal" and b <= 0: return False
    if prioris == "Normal-Gama" and (b <= 0 or c <= 1 or d <= 0): return False

    # Likelihood validation
    if verossimilhancas == "Bernoulli" and (not (0 <= x_bernoulli <= 1) or n < 1): return False
    if verossimilhancas == "Geométrica" and (x < 1 or n < 1): return False
    if verossimilhancas == "Binomial negativa" and (x <= m or n < 1 or m < 1): return False
    if verossimilhancas in ["Exponencial", "Poisson"] and (x < 0 or n < 1): return False
    if verossimilhancas == "Gama (b desconhecido)" and (x < 0 or conhecido <= 0 or n < 1): return False
    if verossimilhancas == "Normal (média desconhecida)" and (conhecido <= 0 or n < 1): return False
    if verossimilhancas == "Normal (média e precisão desconhecidas)" and (n < 2 or conhecido <= 0): return False
    
    return True

# Callbacks para atualização dos gráficos
@app.callback(
    Output('densidade_priori', 'figure'),
    [Input('input-a', 'value'), Input('input-b', 'value'), Input('input-c', 'value'), Input('input-d', 'value'), Input("prioris","value")]
)
def update_priori_graph(a, b, c, d, prioris):
    if any(v is None for v in [a, b, prioris]):
        return go.Figure(layout={"template": "plotly_white"})

    invalid_fig = go.Figure(layout={"title": "Parâmetros da Priori Inválidos", "template": "plotly_white"})

    if prioris == "Beta":
        if a <= 0 or b <= 0: return invalid_fig
        return plot_beta_distribution(a,b)
    elif prioris == "Gama":
        if a <= 0 or b <= 0: return invalid_fig
        return plot_gamma_distribution(a,b)
    elif prioris == "Normal":
        if b <= 0: return invalid_fig
        return plot_normal_distribution(a,b)
    elif prioris == "Normal-Gama":
        if any(v is None for v in [c, d]) or b <= 0 or c <= 1 or d <= 0: return invalid_fig
        return plot_normal_gama_distribution(a,b,c,d)
    
    return go.Figure(layout={"template": "plotly_white"})

@app.callback(
    [Output('densidade_verossimilhanca', 'figure'), Output('aparencia_verossimilhanca', 'style')],
    [Input('input-m', 'value'), Input('input-x', 'value'), Input('input-x-bernoulli','value'), Input('input-tamanho','value'), Input('input-conhecido','value'), Input('verossimilhancas','value')]
)
def update_likelihood_graph(m, x, x_bernoulli, n, conhecido, verossimilhanca):
    if verossimilhanca is None or n is None:
        return go.Figure(), {'display': 'none'}

    style = {**card_style}
    invalid_fig = go.Figure(layout={"title": "Parâmetros da Verossimilhança Inválidos", "template": "plotly_white"})
    
    if not validate_all_params(1, 1, 1, 1, m, x, x_bernoulli, n, conhecido, "Beta", verossimilhanca):
        if verossimilhanca == "Normal (média e precisão desconhecidas)":
             if not validate_all_params(1, 1, 2, 1, m, x, x_bernoulli, n, conhecido, "Normal-Gama", verossimilhanca):
                 return invalid_fig, style
        else:
            return invalid_fig, style

    if verossimilhanca=="Geométrica":
        return verossimilhanca_geometrica_aproximada(x,n), style
    elif verossimilhanca=="Bernoulli":
        return verossimilhanca_bernoulli_aproximada(x_bernoulli,n), style
    elif verossimilhanca=="Normal (média desconhecida)":
        return verossimilhanca_normal_aproximada(x,conhecido,n), style
    elif verossimilhanca=="Exponencial":
        return verossimilhanca_exponencial_aproximada(x,n), style
    elif verossimilhanca=="Poisson":
        return verossimilhanca_poisson_aproximada(x,n), style
    elif verossimilhanca=="Binomial negativa":
        return verossimilhanca_binomial_negativa_aproximada(x,m,n), style
    elif verossimilhanca=="Gama (b desconhecido)":
        return verossimilhanca_gama_aproximada(x,conhecido,n), style
    elif verossimilhanca=="Normal (média e precisão desconhecidas)":
        return verossimilhanca_normal_gama_aproximada(x,conhecido,n), style
    else: return go.Figure(), {'display': 'none'}

@app.callback(
    Output('densidade_posteriori', 'figure'),
    [Input('input-a', 'value'), Input('input-b', 'value'), Input('input-c', 'value'), Input('input-d', 'value'), Input('input-m', 'value'), Input('input-x', 'value'), Input('input-x-bernoulli','value'), Input('input-tamanho', 'value'), Input('input-conhecido', 'value'), Input("prioris","value"), Input("verossimilhancas","value")]
)
def update_posterior_graph(a,b,c,d,m,x,x_bernoulli,n,conhecido,prioris,verossimilhancas):
    if not validate_all_params(a, b, c, d, m, x, x_bernoulli, n, conhecido, prioris, verossimilhancas):
        return go.Figure(layout={"title": "Parâmetros Inválidos", "template": "plotly_white"})

    if verossimilhancas=="Bernoulli": return posteriori_beta(round(a+n*x_bernoulli,3),round(b+n*(1-x_bernoulli),3))
    elif verossimilhancas=="Poisson": return posteriori_gama(round(n*x+a,3),b+n)
    elif verossimilhancas=="Exponencial": return posteriori_gama(a+n,round(b+n*x,3))
    elif verossimilhancas=="Gama (b desconhecido)": return posteriori_gama(round(a+n*conhecido,3),round(b+n*x,3))
    elif verossimilhancas=="Geométrica": return posteriori_beta(a+n,round(b+n*(x-1),3))
    elif verossimilhancas=="Binomial negativa": return posteriori_beta(round(a+n*m,3),round(b+n*(x-m),3))
    elif verossimilhancas=="Normal (média e precisão desconhecidas)":
        return posteriori_Normal_Gama(a,b,c,d,x,conhecido,n)
    else: # Normal (média desconhecida)
        return posteriori_normal(round((n*b*x+a*conhecido)/(n*b+conhecido),3),round(conhecido*b/(n*b+conhecido),3))

@app.callback(
    Output('grafico_conjunto', 'figure'),
    [Input('input-a', 'value'), Input('input-b', 'value'), Input('input-c', 'value'), Input('input-d', 'value'), Input('input-m', 'value'), Input('input-x', 'value'), Input('input-x-bernoulli','value'), Input('input-tamanho', 'value'), Input('input-conhecido', 'value'), Input("prioris","value"), Input("verossimilhancas","value")]
)
def update_final_graph(a,b,c,d,m,x,x_bernoulli,n,conhecido,prioris,verossimilhancas):
    if not validate_all_params(a,b,c,d,m,x,x_bernoulli,n,conhecido,prioris,verossimilhancas):
        return go.Figure(layout={"title": "Parâmetros Inválidos", "template": "plotly_white"})

    if verossimilhancas=="Bernoulli": return beta_bernoulli(a,b,x_bernoulli,n)
    elif verossimilhancas=="Geométrica": return beta_geometrica(a,b,x,n)
    elif verossimilhancas=="Binomial negativa": return beta_binomial_negativa(a,b,x,m,n)
    elif verossimilhancas=="Exponencial": return gama_exponencial(a,b,x,n)
    elif verossimilhancas=="Poisson": return gama_poisson(a,b,x,n)
    elif verossimilhancas=="Gama (b desconhecido)": return gama_gama(a,b,x,conhecido,n)
    elif verossimilhancas=="Normal (média e precisão desconhecidas)":
        return Normal_Gama_final(a,b,c,d,x,conhecido,n)
    else: # Normal (média desconhecida)
        return normal_normal(a,b,x,conhecido,n)

@app.callback(
    Output("texto_formula_div", "children"),
    [Input("botao", "n_clicks")],
    [
        State("verossimilhancas", "value"),
        State("input-a", "value"), State("input-b", "value"),
        State("input-c", "value"), State("input-d", "value"),
        State("input-m", "value"), State("input-x", "value"),
        State("input-x-bernoulli", "value"), State("input-tamanho", "value"),
        State("input-conhecido", "value")
    ]
)
def update_formulas(n_clicks, verossimilhancas, a, b, c, d, m, x, x_bernoulli, n, conhecido):
    if n_clicks is None or n_clicks == 0:
        return html.Div()
    
    if n_clicks % 2 != 0:
        if verossimilhancas=="Bernoulli":
            return dcc.Markdown(r'''
### Fórmulas matemáticas:
#### Priori:
$p \sim Beta(a,b)$
$f(p)=\frac{\Gamma(a+b) p^{a-1}(1-p)^{b-1}}{\Gamma(a)\Gamma(b)}\mathbb{I}_{(0,1)}(p)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Bernoulli(p)$
#### Verossimilhança: $L(p|\mathbf{X})=p^{n\bar{x}}(1-p)^{(n-1)\bar{x}}$
#### Posteriori:
$p|\mathbf{X}\sim Beta(a+n\bar{x}, b+n(1-\bar{x}))$
''', mathjax=True)
        elif verossimilhancas=="Geométrica":
            return dcc.Markdown(r'''
### Fórmulas matemáticas:
#### Priori:
$p \sim Beta(a,b)$
$f(p)=\frac{\Gamma(a+b) p^{a-1}(1-p)^{b-1}}{\Gamma(a)\Gamma(b)}\mathbb{I}_{(0,1)}(p)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $\text{Geométrica}(p)$
#### Verossimilhança: $L(p|\mathbf{X})=p^n(1-p)^{n(\bar{x}-1)}$
#### Posteriori:
$p|\mathbf{X}\sim Beta(a + n, b+n(\bar{x} - 1))$
''', mathjax=True)
        elif verossimilhancas=="Binomial negativa":
            return dcc.Markdown(r'''
### Fórmulas matemáticas:
#### Priori:
$p \sim Beta(a,b)$
$f(p)=\frac{\Gamma(a+b) p^{a-1}(1-p)^{b-1}}{\Gamma(a)\Gamma(b)}\mathbb{I}_{(0,1)}(p)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Binomial Negativa(r, p)$
#### Verossimilhança: $L(p|\mathbf{X})=\left( \prod_{i=1}^n  \binom{x_i-1}{r-1} \right) p^{nr}(1-p)^{n(\bar{x}-r)}$
#### Posteriori:
$p|\mathbf{X}\sim Beta(a + nr, b+n(\bar{x} - r))$
''', mathjax=True)
        elif verossimilhancas=="Exponencial":
            return dcc.Markdown(r'''
### Fórmulas matemáticas:
#### Priori:
$\lambda \sim Gama(a,b)$
$f(\lambda)=\frac{b^a \lambda^{a - 1} e^{-b \lambda}}{\Gamma(a)} \mathbb{I}_{(0, \infty)}(\lambda)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Exponencial(\lambda)$
#### Verossimilhança: $L(\lambda|\mathbf{X})=\lambda^ne^{-\lambda n\bar{x}}$
#### Posteriori:
$\lambda|\mathbf{X}\sim Gama(a+n, b+n\bar{x})$
''', mathjax=True)
        elif verossimilhancas=="Poisson":
            return dcc.Markdown(r'''
### Fórmulas matemáticas:
#### Priori:
$\lambda \sim Gama(a,b)$
$f(\lambda)=\frac{b^a \lambda^{a - 1} e^{-b \lambda}}{\Gamma(a)} \mathbb{I}_{(0, \infty)}(\lambda)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Poisson(\lambda)$
#### Verossimilhança: $L(\lambda|\mathbf{X})= \frac{e^{-\lambda}\lambda^{n\bar{x}}}{\prod_{i=1}^n x_i!}$
#### Posteriori:
$\lambda|\mathbf{X}\sim Gama(n\bar{x}+a, b+n)$
''', mathjax=True)
        elif verossimilhancas=="Gama (b desconhecido)":
            return dcc.Markdown(r'''
### Fórmulas matemáticas:
#### Priori:
$b \sim Gama(a_0,b_0)$
$f(b)=\frac{b_0^{a_0} b^{a_0 - 1} e^{-b_0 b}}{\Gamma(a_0)} \mathbb{I}_{(0, \infty)}(b)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Gama(a,b)$
#### Verossimilhança: $L(b|\mathbf{X})=\frac{\left( \prod_{i=1}^n x_i \right)^{a-1} b^{na}e^{-nb\bar{x}}}{\Gamma(a)}$
#### Posteriori:
$p|\mathbf{X}\sim Gama(a_0 + na, b_0 + n \bar{x})$
''', mathjax=True)
        elif verossimilhancas=="Normal (média desconhecida)":
            return dcc.Markdown(r'''
### Fórmulas matemáticas:
#### Priori:
$\mu \sim Normal(\mu_0,\sigma^2_0)$
$f(\mu) = \frac{1}{\sqrt{2\pi\sigma^2_0}} \exp\left( -\frac{(\mu - \mu_0)^2}{2\sigma^2_0} \right) \mathbb{I}_{(-\infty, \infty)}(\mu)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Normal \left(\mu,\sigma^2 \right)$
#### Verossimilhança: $L(\mu|\mathbf{X}) = \left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^n \exp\left(-\frac{1}{2\sigma^2}\left(n\mu^2-2\mu n\bar{x} + \sum_{i=1}^{n} x_i^2\right)\right)$
#### Posteriori:
$\mu|\mathbf{X}\sim Normal\left( \frac{n \sigma^2_0 \bar{x} + \sigma^2 \mu_0}{n \sigma^2_0 + \sigma^2} \,,\, \frac{\sigma^2 \sigma^2_0}{n \sigma^2_0 + \sigma^2}\right)$
''', mathjax=True)
        else:
            return dcc.Markdown(r'''
### Fórmulas matemáticas:
#### Priori:
$(\mu,\tau) \sim Normal-Gama(\mu_0,\lambda,a,b)$
$f(\mu,\tau)=\frac{b^a \sqrt{\lambda}\tau^{a-0.5}e^{-b\tau}} {\Gamma{(a)}\sqrt{2\pi}} exp\left( \frac{\lambda \tau \left( \mu-\mu_0 \right)^2}{2} \right)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Normal(\mu,\tau^{-1})$
#### Verossimilhança: $L(\mu,\tau|\mathbf{X})=\frac{1}{(\sqrt{2\pi})^n}\tau^{\frac{n}{2}}exp(\frac{-ns^2\tau}{2})exp \left( \frac{-n\tau \left( \mu-\bar{x} \right)^2}{2} \right)$
#### Posteriori:
$(\mu,\tau)|\mathbf{X} \sim Normal-Gama\left( \frac{\lambda\mu_0 + n\bar{x}}{\lambda+n}, \lambda+n, a+\frac{n}{2}, b+\frac{1}{2}\left( ns^2 + \frac{\lambda n \left( \bar{x} - \mu_0 \right)^2}{\lambda+n} \right) \right)$
''', mathjax=True)
    else:
        try:
            a=float(a); b=float(b); c=float(c); d=float(d); m=float(m); x=float(x)
            x_bernoulli=float(x_bernoulli); n=float(n); conhecido=float(conhecido)
        except (ValueError, TypeError):
            return dcc.Markdown("Aguardando todos os parâmetros...", mathjax=True)

        if verossimilhancas=="Bernoulli":
            return dcc.Markdown(fr'''
### Fórmulas matemáticas:
#### Priori:
$p \sim Beta({a},{b})$
$f(p)=\frac{{\Gamma({a+b}) p^{{{a-1}}}(1-p)^{{{b-1}}}}}{{\Gamma({a})\Gamma({b})}}\mathbb{{I}}_{{(0,1)}}(p)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Bernoulli(p)$
#### Verossimilhança: $L(p|\mathbf{{X}})=p^{{{n*x_bernoulli}}}(1-p)^{{{n*(1-x_bernoulli)}}}$
#### Posteriori:
$p|\mathbf{{X}}\sim Beta({a+n*x_bernoulli}, {b+n*(1-x_bernoulli)})$
''', mathjax=True)
        elif verossimilhancas=="Geométrica":
            return dcc.Markdown(fr'''
### Fórmulas matemáticas:
#### Priori:
$p \sim Beta({a},{b})$
$f(p)=\frac{{\Gamma({a+b}) p^{{{a-1}}}(1-p)^{{{b-1}}}}}{{\Gamma({a})\Gamma({b})}}\mathbb{{I}}_{{(0,1)}}(p)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $\text{{Geométrica}}(p)$
#### Verossimilhança: $L(p|\mathbf{{X}})=p^{n}(1-p)^{{{n*(x-1)}}}$
#### Posteriori:
$p|\mathbf{{X}} \sim Beta({a + n}, {b+n*(x - 1)})$
''', mathjax=True)
        elif verossimilhancas=="Binomial negativa":
            return dcc.Markdown(fr'''
### Fórmulas matemáticas:
#### Priori:
$p \sim Beta({a},{b})$
$f(p)=\frac{{\Gamma({a+b}) p^{{{a-1}}}(1-p)^{{{b-1}}}}}{{\Gamma({a})\Gamma({b})}}\mathbb{{I}}_{{(0,1)}}(p)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Binomial Negativa({m}, p)$
#### Verossimilhança: $L(p|\mathbf{{X}})=\left( \prod_{{i=1}}^{n}  \binom{{x_i-1}}{{{m-1}}} \right) p^{{{n*m}}}(1-p)^{{{n*(x-m)}}}$
#### Posteriori:
$p|\mathbf{{X}} \sim Beta({a + n*m}, {b+n*(x - m)})$
''', mathjax=True)
        elif verossimilhancas=="Exponencial":
            return dcc.Markdown(fr'''
### Fórmulas matemáticas:
#### Priori:
$\lambda \sim Gama({a},{b})$
$f(\lambda)=\frac{{{b**a} \lambda^{{{a - 1}}} e^{{- {b} \lambda}}}}{{\Gamma({a})}} \mathbb{{I}}_{{(0, \infty)}}(\lambda)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Exponencial(\lambda)$
#### Verossimilhança: $L(\lambda|\mathbf{{X}})=\lambda^{n}e^{{- {n*x} \lambda }}$
#### Posteriori:
$\lambda|\mathbf{{X}}\sim Gama({a+n}, {b+n*x})$
''', mathjax=True)
        elif verossimilhancas=="Poisson":
            return dcc.Markdown(fr'''
### Fórmulas matemáticas:
#### Priori:
$\lambda \sim Gama({a},{b})$
$f(\lambda)=\frac{{{b**a} \lambda^{{{a - 1}}} e^{{- {b} \lambda}}}}{{\Gamma({a})}} \mathbb{{I}}_{{(0, \infty)}}(\lambda)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Poisson(\lambda)$
#### Verossimilhança: $L(\lambda|\mathbf{{X}})= \frac{{e^{{-\lambda}}\lambda^{{{n*x}}}}}{{\prod_{{i=1}}^{n} x_i!}}$
#### Posteriori:
$\lambda|\mathbf{{X}} \sim Gama({n*x+a}, {b+n})$
''', mathjax=True)
        elif verossimilhancas=="Gama (b desconhecido)":
            return dcc.Markdown(fr'''
### Fórmulas matemáticas:
#### Priori:
$b \sim Gama({a},{b})$
$f(b)=\frac{{{b**a} b^{{{a - 1}}} e^{{- {b} b}}}}{{\Gamma({a})}} \mathbb{{I}}_{{(0, \infty)}}(b)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Gama({conhecido},b)$
#### Verossimilhança: $L(b|\mathbf{{X}})=\frac{{\left( \prod_{{i=1}}^{n} x_i \right)^{{{conhecido-1}}} b^{{{n*conhecido}}}e^{{- {n*x} b}}}}{{\Gamma({conhecido})}}$
#### Posteriori:
$p|\mathbf{{X}}\sim Gama({a + n*conhecido}, {b + n*x})$
''', mathjax=True)
        elif verossimilhancas=="Normal (média desconhecida)":
            return dcc.Markdown(fr'''
### Fórmulas matemáticas:
#### Priori:
$\mu \sim Normal({a},{b})$
$f(\mu) = \frac{{1}}{{\sqrt{{2\cdot{b}\pi}}}} \exp\left( -\frac{{(\mu - {a})^2}}{{2\cdot{b}}} \right) \mathbb{{I}}_{{(-\infty, \infty)}}(\mu)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Normal \left(\mu,{conhecido} \right)$
#### Verossimilhança: $L(\mu|\mathbf{{X}}) = \left(\frac{{1}}{{\sqrt{{2\cdot{conhecido}\pi}}}}\right)^{n} \exp\left(-\frac{{1}}{{2\cdot{conhecido}}}\left({n} \mu^2- {2*n*x} \mu  + \sum_{{i=1}}^{n} x_i^2\right)\right)$
#### Posteriori:
$\mu|\mathbf{{X}}\sim Normal\left( {((n*b*x + conhecido*a)/(n*b + conhecido)):.3f} \,,\, {((conhecido*b)/(n*b + conhecido)):.3f}\right)$
''', mathjax=True)
        else:
            return dcc.Markdown(fr'''
### Fórmulas matemáticas:
#### Priori:
$(\mu,\tau) \sim Normal-Gama({a},{b},{c},{d})$
$f(\mu,\tau)=\frac{{{d**c} \sqrt{{{b}}}\tau^{{{c-0.5}}}e^{{- {d} \tau}}}} {{\Gamma{{({c})}}\sqrt{{2\pi}}}} exp\left( \frac{{{b} \tau \left( \mu-{a} \right)^2}}{{2}} \right)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Normal(\mu,\tau^{{-1}})$
#### Verossimilhança: $L(\mu,\tau|\mathbf{{X}})=\frac{{1}}{{(\sqrt{{2\pi}})^{n}}}\tau^{{{n/2}}}exp(\frac{{- {n*conhecido} \tau}}{{2}})exp \left( \frac{{- {n} \tau \left( \mu-{x} \right)^2}}{{2}} \right)$
#### Posteriori:
$(\mu,\tau)|\mathbf{{X}} \sim Normal-Gama\left( {((b*a+n*x)/(b+n)):.3f}, {b+n}, {c+n/2},{(d +0.5*(n*conhecido + (b*n*(x-a)**2)/(b+n))):.3f} \right)$
''', mathjax=True)

@app.callback(
    Output("botao", "children"),
    [Input("botao", "n_clicks")]
)
def update_button_text(n_clicks):
    if n_clicks is None or n_clicks % 2 == 0:
        return "Ver Fórmulas Gerais"
    return "Aplicar valores nos campos"

if __name__ == '__main__':
    app.run_server(debug=True)
