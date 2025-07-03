# -*- coding: utf-8 -*-
"""
Dashboard de Análise Bayesiana com Dash e Plotly.

Este aplicativo permite a visualização interativa de distribuições de probabilidade,
o Teorema de Bayes e a análise de pares conjugados (Priori-Verossimilhança-Posteriori).
O código foi reestruturado para maior clareza e manutenibilidade.
"""

# ==============================================================================
# SEÇÃO 1: IMPORTAÇÕES E CONFIGURAÇÃO INICIAL
# ==============================================================================

import numpy as np
import plotly.graph_objects as go
from scipy.stats import beta, gamma, norm
import dash
from dash import Dash, html, dcc, Input, Output, State, exceptions
import math

# Inicialização do aplicativo Dash
app = Dash(__name__)
server = app.server

# ==============================================================================
# SEÇÃO 2: CONSTANTES E ESTILOS
# ==============================================================================

# Lista de distribuições a priori disponíveis
LISTA_PRIORIS = ["Beta", "Gama", "Normal", "Normal-Gama"]

# Paletas de cores para os gráficos 3D
COLORSCALE_BLUE = [[0.0, "lightblue"], [0.5, "blue"], [1.0, "darkblue"]]
COLORSCALE_RED = [[0.0, "lightcoral"], [0.5, "red"], [1.0, "darkred"]]
COLORSCALE_GREEN = [[0.0, "lightgreen"], [0.5, "green"], [1.0, "darkgreen"]]

# Dicionários de estilo para componentes do layout
COLORS = {'background': '#F3F6FA', 'text': '#333333', 'light_grey': '#E9ECEF'}
HEADER_STYLE = {
    "backgroundColor": "white", "padding": "20px", "textAlign": "center",
    "borderBottom": f"1px solid {COLORS['light_grey']}", "display": "flex",
    "alignItems": "center", "justifyContent": "space-between",
    "boxShadow": "0 2px 4px rgba(0,0,0,0.05)"
}
LOGO_STYLE = {"height": "60px"}
NAV_BUTTONS_STYLE = {"display": "flex", "gap": "15px"}
MAIN_CONTENT_STYLE = {
    "padding": "30px", "backgroundColor": COLORS['background'],
    "fontFamily": "Roboto, sans-serif"
}
CARD_STYLE = {
    "backgroundColor": "white", "padding": "20px", "borderRadius": "10px",
    "boxShadow": "0 2px 4px rgba(0,0,0,0.05)", "marginBottom": "20px"
}


# ==============================================================================
# SEÇÃO 3: FUNÇÕES AUXILIARES DE PLOTAGEM E CÁLCULO
# ==============================================================================

# ------------------------------------------------------------------------------
# Funções para Plotagem de Distribuições Base
# ------------------------------------------------------------------------------

def plot_beta_distribution(a, b, v="Priori"):
    """Gera um gráfico da distribuição Beta."""
    if a<=0 or b<=0:
        return go.Figure(layout={"title": "Parâmetros a e b devem ser > 0", "template": "plotly_white"})
    theta = np.linspace(0, 1, 1000)
    beta_pdf_vals = beta.pdf(theta, a, b) if not (a==1 and b==1) else np.ones_like(theta)

    fig = go.Figure(go.Scatter(
        x=theta, y=beta_pdf_vals, mode='lines', name=f'Beta({a}, {b})',
        line=dict(color='royalblue', width=2)
    ))
    fig.update_layout(
        title=f'{v} Beta({a}, {b})', xaxis_title='Suporte do parâmetro',
        yaxis_title='Densidade', template='plotly_white', showlegend=True
    )
    return fig

def plot_gamma_distribution(a, b, v="Priori"):
    """Gera um gráfico da distribuição Gama."""
    if a<=0 or b<=0:
        return go.Figure(layout={"title": "Parâmetros a e b devem ser > 0", "template": "plotly_white"})
    mode = (a-1)/b if a>1 else 0
    std_dev = np.sqrt(a/b**2)
    x_min = max(0, mode-3*std_dev)
    x_max = mode+3*std_dev
    x = np.linspace(x_min+1e-9, x_max, 1000)
    gamma_pdf_vals = gamma.pdf(x, a, scale=1/b)

    fig = go.Figure(go.Scatter(
        x=x, y=gamma_pdf_vals, mode='lines', name=f'Gama({a}, {b})',
        line=dict(color='royalblue', width=2)
    ))
    fig.update_layout(
        title=f'{v} Gama({a}, {b})', xaxis_title='Suporte do parâmetro',
        yaxis_title='Densidade', template='plotly_white', showlegend=True
    )
    return fig

def plot_normal_distribution(mu, sigma2, v="Priori"):
    """Gera um gráfico da distribuição Normal."""
    if sigma2<=0:
        return go.Figure(layout={"title": "Variância deve ser > 0", "template": "plotly_white"})
    sigma = np.sqrt(sigma2)
    x = np.linspace(mu-4*sigma, mu+4*sigma, 1000)
    normal_pdf_vals = norm.pdf(x, mu, sigma)

    fig = go.Figure(go.Scatter(
        x=x, y=normal_pdf_vals, mode='lines', name=f'Normal({mu}, {sigma2})',
        line=dict(color='royalblue', width=2)
    ))
    fig.update_layout(
        title=f'{v} Normal({mu}, {sigma2})', xaxis_title='Suporte do parâmetro',
        yaxis_title='Densidade', template='plotly_white', showlegend=True
    )
    return fig

def plot_normal_gama_distribution(mu, lambda_, alpha, beta_val, cor=COLORSCALE_BLUE, v="Priori"):
    """Gera um gráfico de superfície 3D da distribuição Normal-Gama."""
    if alpha<=1 or lambda_<=0 or beta_val<=0:
        return go.Figure(layout={"title": f"Parâmetros Inválidos para {v} Normal-Gama", "template": "plotly_white"})

    x_vals, tau_vals = xvals_tauvals(mu, lambda_, alpha, beta_val)
    X, T = np.meshgrid(x_vals, tau_vals)
    Z = normal_gama_pdf(X, T, mu, lambda_, alpha, beta_val)

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=T, colorscale=cor)])
    fig.update_layout(
        title=f"{v} Normal-Gama({round(mu,3)}, {round(lambda_,3)}, {round(alpha,3)}, {round(beta_val,3)})",
        scene=dict(xaxis_title="μ", yaxis_title="τ", zaxis_title="Densidade")
    )
    return fig

# ------------------------------------------------------------------------------
# Funções para Plotagem da Verossimilhança (Aproximada)
# ------------------------------------------------------------------------------

def _plot_verossimilhanca(fig, nome_dist, titulo):
    """Função auxiliar para customizar o gráfico da verossimilhança."""
    fig.data[0].name = f'Verossim.: {nome_dist}'
    fig.data[0].line.color = 'red'
    fig.update_layout(title=f"Verossimilhança reescalada da {titulo}", yaxis_title="Verossimilhança", showlegend=True)
    return fig

def verossimilhanca_binomial_aproximada(x, m, n):
    figura = plot_beta_distribution(round(n*x+1, 3), round(n*(m-x)+1, 3))
    return _plot_verossimilhanca(figura, 'Binomial', 'Binomial')

def verossimilhanca_poisson_aproximada(x, n):
    figura = plot_gamma_distribution(round(n*x+1, 3), n)
    return _plot_verossimilhanca(figura, 'Poisson', 'Poisson')

def verossimilhanca_binomial_negativa_aproximada(x, r, n):
    figura = plot_beta_distribution(round(n*r+1, 3), round(n*(x-r)+1, 3))
    return _plot_verossimilhanca(figura, 'Binomial Negativa', 'Binomial negativa')

def verossimilhanca_gama_aproximada(x, a, n):
    figura = plot_gamma_distribution(round(n*a+1, 3), round(n*x, 3))
    return _plot_verossimilhanca(figura, 'Gama', 'Gama')

def verossimilhanca_exponencial_aproximada(x, n):
    figura = plot_gamma_distribution(n+1, round(n*x, 3))
    return _plot_verossimilhanca(figura, 'Exponencial', 'Exponencial')

def verossimilhanca_normal_aproximada(x, sigma2, n):
    figura = plot_normal_distribution(x, round(sigma2/n, 3))
    return _plot_verossimilhanca(figura, 'Normal', 'Normal')

def verossimilhanca_bernoulli_aproximada(x, n):
    figura = plot_beta_distribution(round(n*x+1, 3), round(n*(1-x)+1, 3))
    return _plot_verossimilhanca(figura, 'Bernoulli', 'Bernoulli')

def verossimilhanca_geometrica_aproximada(x, n):
    figura = plot_beta_distribution(n+1, round(n*(x-1)+1, 3))
    return _plot_verossimilhanca(figura, 'Geométrica', 'Geométrica')

def verossimilhanca_normal_gama_aproximada(x, s, n):
    alpha_vero = (n-1)/2
    beta_vero = n*s/2
    if alpha_vero<=0 or beta_vero<=0:
        return go.Figure(layout={"title": "Parâmetros da Verossimilhança Inválidos", "template": "plotly_white"})

    figura = plot_normal_gama_distribution(x, n, alpha_vero, beta_vero, COLORSCALE_RED, "Verossimilhança")
    figura.update_layout(
        title="Verossimilhança aproximada da Normal",
        scene=dict(xaxis_title="μ", yaxis_title="τ", zaxis_title="Verossimilhança")
    )
    return figura

# ------------------------------------------------------------------------------
# Funções para Plotagem da Posteriori
# ------------------------------------------------------------------------------

def _plot_posteriori(fig):
    """Função auxiliar para customizar o gráfico da posteriori."""
    fig.data[0].line.color = 'green'
    return fig

def posteriori_beta(a, b, v="Posteriori"):
    return _plot_posteriori(plot_beta_distribution(a, b, v))

def posteriori_gama(a, b, v="Posteriori"):
    return _plot_posteriori(plot_gamma_distribution(a, b, v))

def posteriori_normal(a, b, v="Posteriori"):
    return _plot_posteriori(plot_normal_distribution(a, b, v))

def posteriori_Normal_Gama(mu, lambda_, alpha, beta_val, x, s, n, v="Posteriori"):
    mu_post = (lambda_*mu+n*x)/(lambda_+n)
    lambda_post = lambda_+n
    alpha_post = alpha+n/2
    beta_post = beta_val+(n*s+(lambda_*n*(x-mu)**2)/(lambda_+n))/2

    if alpha_post<=1:
        return go.Figure(layout={"title": "Parâmetros da Posteriori Inválidos", "template": "plotly_white"})

    return plot_normal_gama_distribution(mu_post, lambda_post, alpha_post, beta_post, COLORSCALE_GREEN, v)

# ------------------------------------------------------------------------------
# Funções para Cálculo de PDF e Plotagem Conjunta
# ------------------------------------------------------------------------------

def beta_pdf_vals(a, b):
    """Calcula os valores da PDF da distribuição Beta para um eixo padrão."""
    if a<=0 or b<=0: return np.zeros(1000)
    theta = np.linspace(0, 1, 1000)
    return beta.pdf(theta, a, b) if not (a==1 and b==1) else np.ones_like(theta)

def gama_range(a, b):
    """Calcula um intervalo razoável para o eixo x de uma distribuição Gama."""
    mode = (a-1)/b if a>1 else 0
    std_dev = np.sqrt(a/b**2)
    x_min = max(0, mode-3*std_dev)
    x_max = mode+3*std_dev
    return x_min, x_max

def normal_range(mu, sigma2):
    """Calcula um intervalo razoável para o eixo x de uma distribuição Normal."""
    if sigma2<=0: return mu-4, mu+4
    sigma = np.sqrt(sigma2)
    return mu-4*sigma, mu+4*sigma

def _plot_conjugada_1d(eixo_x, y_priori, y_verossimilhanca, y_posteriori, nomes, titulo):
    """Função auxiliar para criar o gráfico 1D combinado."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eixo_x, y=y_priori, mode='lines', name=nomes['priori'], line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=eixo_x, y=y_verossimilhanca, mode='lines', name=nomes['verossimilhanca'], line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=eixo_x, y=y_posteriori, mode='lines', name=nomes['posteriori'], line=dict(color='green', width=2)))
    fig.update_layout(title=titulo, xaxis_title='Suporte do parâmetro', yaxis_title='Densidade', template='plotly_white')
    return fig

def beta_binomial(a, b, x, m, n):
    eixo_x = np.linspace(0, 1, 1000)
    y_priori = beta_pdf_vals(a, b)
    y_verossimilhanca = beta_pdf_vals(round(n*x+1, 3), round(n*(m-x)+1, 3))
    y_posteriori = beta_pdf_vals(round(n*x+a, 3), round(b+n*(m-x), 3))
    nomes = {
        'priori': f'Priori: Beta({a}, {b})',
        'verossimilhanca': 'Verossim.: Binomial',
        'posteriori': f'Posteriori: Beta({round(n*x+a,3)}, {round(b+n*(m-x),3)})'
    }
    return _plot_conjugada_1d(eixo_x, y_priori, y_verossimilhanca, y_posteriori, nomes, 'Distribuição Beta-Binomial')

def beta_bernoulli(a, b, x, n):
    eixo_x = np.linspace(0, 1, 1000)
    y_priori = beta_pdf_vals(a, b)
    y_verossimilhanca = beta_pdf_vals(round(n*x+1, 3), round(n*(1-x)+1, 3))
    y_posteriori = beta_pdf_vals(round(a+n*x, 3), round(b+n*(1-x), 3))
    nomes = {
        'priori': f'Priori: Beta({a}, {b})',
        'verossimilhanca': 'Verossim.: Bernoulli',
        'posteriori': f'Posteriori: Beta({round(a+n*x,3)}, {round(b+n*(1-x),3)})'
    }
    return _plot_conjugada_1d(eixo_x, y_priori, y_verossimilhanca, y_posteriori, nomes, 'Distribuição Beta-Bernoulli')

def beta_geometrica(a, b, x, n):
    eixo_x = np.linspace(0, 1, 1000)
    y_priori = beta_pdf_vals(a, b)
    y_verossimilhanca = beta_pdf_vals(n+1, round(n*(x-1)+1, 3))
    y_posteriori = beta_pdf_vals(a+n, round(b+n*(x-1), 3))
    nomes = {
        'priori': f'Priori: Beta({a}, {b})',
        'verossimilhanca': 'Verossim.: Geométrica',
        'posteriori': f'Posteriori: Beta({a+n}, {round(b+n*(x-1),3)})'
    }
    return _plot_conjugada_1d(eixo_x, y_priori, y_verossimilhanca, y_posteriori, nomes, 'Distribuição Beta-Geométrica')

def beta_binomial_negativa(a, b, x, r, n):
    eixo_x = np.linspace(0, 1, 1000)
    y_priori = beta_pdf_vals(a, b)
    y_verossimilhanca = beta_pdf_vals(round(n*r+1, 3), round(n*(x-r)+1, 3))
    y_posteriori = beta_pdf_vals(round(a+n*r, 3), round(b+n*(x-r), 3))
    nomes = {
        'priori': f'Priori: Beta({a}, {b})',
        'verossimilhanca': 'Verossim.: Binomial Negativa',
        'posteriori': f'Posteriori: Beta({round(a+n*r,3)}, {round(b+n*(x-r),3)})'
    }
    return _plot_conjugada_1d(eixo_x, y_priori, y_verossimilhanca, y_posteriori, nomes, 'Distribuição Beta-Binomial negativa')

def gama_exponencial(a, b, x, n):
    x_min_pri, x_max_pri = gama_range(a, b)
    x_min_ver, x_max_ver = gama_range(n+1, n*x)
    x_min_pos, x_max_pos = gama_range(a+n, b+n*x)
    x_min = min(x_min_pri, x_min_ver, x_min_pos)
    x_max = max(x_max_pri, x_max_ver, x_max_pos)
    eixo_x = np.linspace(x_min, x_max, 1000)
    y_priori = gamma.pdf(eixo_x, a, scale=1/b)
    y_verossimilhanca = gamma.pdf(eixo_x, n+1, scale=round(1/(n*x), 3))
    y_posteriori = gamma.pdf(eixo_x, a+n, scale=round(1/(b+n*x), 3))
    nomes = {
        'priori': f'Priori: Gama({a}, {b})',
        'verossimilhanca': 'Verossim.: Exponencial',
        'posteriori': f'Posteriori: Gama({a+n}, {round(b+n*x,3)})'
    }
    return _plot_conjugada_1d(eixo_x, y_priori, y_verossimilhanca, y_posteriori, nomes, 'Distribuição Gama-Exponencial')

def gama_poisson(a, b, x, n):
    x_min_pri, x_max_pri = gama_range(a, b)
    x_min_ver, x_max_ver = gama_range(n*x+1, n)
    x_min_pos, x_max_pos = gama_range(n*x+a, b+n)
    x_min = min(x_min_pri, x_min_ver, x_min_pos)
    x_max = max(x_max_pri, x_max_ver, x_max_pos)
    eixo_x = np.linspace(x_min, x_max, 1000)
    y_priori = gamma.pdf(eixo_x, a, scale=1/b)
    y_verossimilhanca = gamma.pdf(eixo_x, round(n*x+1, 3), scale=round(1/n, 3))
    y_posteriori = gamma.pdf(eixo_x, round(n*x+a, 3), scale=round(1/(b+n), 3))
    nomes = {
        'priori': f'Priori: Gama({a}, {b})',
        'verossimilhanca': 'Verossim.: Poisson',
        'posteriori': f'Posteriori: Gama({round(n*x+a,3)}, {b+n})'
    }
    return _plot_conjugada_1d(eixo_x, y_priori, y_verossimilhanca, y_posteriori, nomes, 'Distribuição Gama-Poisson')

def gama_gama(a, b, x, conhecido, n):
    x_min_pri, x_max_pri = gama_range(a, b)
    x_min_ver, x_max_ver = gama_range(n*conhecido+1, n*x)
    x_min_pos, x_max_pos = gama_range(a+n*conhecido, b+n*x)
    x_min = min(x_min_pri, x_min_ver, x_min_pos)
    x_max = max(x_max_pri, x_max_ver, x_max_pos)
    eixo_x = np.linspace(x_min, x_max, 1000)
    y_priori = gamma.pdf(eixo_x, a, scale=1/b)
    y_verossimilhanca = gamma.pdf(eixo_x, round(n*conhecido+1, 3), scale=round(1/(n*x), 3))
    y_posteriori = gamma.pdf(eixo_x, round(a+n*conhecido, 3), scale=round(1/(b+n*x), 3))
    nomes = {
        'priori': f'Priori: Gama({a}, {b})',
        'verossimilhanca': 'Verossim.: Gama',
        'posteriori': f'Posteriori: Gama({round(a+n*conhecido,3)}, {round(b+n*x,3)})'
    }
    return _plot_conjugada_1d(eixo_x, y_priori, y_verossimilhanca, y_posteriori, nomes, 'Distribuição Gama-Gama')

def normal_normal(mu, sigma2, x, conhecido, n):
    x_min_pri, x_max_pri = normal_range(mu, sigma2)
    x_min_ver, x_max_ver = normal_range(x, conhecido/n)
    mu_post = (n*sigma2*x+conhecido*mu)/(n*sigma2+conhecido)
    sigma2_post = sigma2*conhecido/(n*sigma2+conhecido)
    x_min_pos, x_max_pos = normal_range(mu_post, sigma2_post)
    x_min = min(x_min_pri, x_min_ver, x_min_pos)
    x_max = max(x_max_pri, x_max_ver, x_max_pos)
    eixo_x = np.linspace(x_min, x_max, 1000)
    y_priori = norm.pdf(eixo_x, mu, np.sqrt(sigma2))
    y_verossimilhanca = norm.pdf(eixo_x, x, round(np.sqrt(conhecido/n), 3))
    y_posteriori = norm.pdf(eixo_x, round(mu_post, 3), round(np.sqrt(sigma2_post), 3))
    nomes = {
        'priori': f'Priori: Normal({mu}, {sigma2})',
        'verossimilhanca': 'Verossim.: Normal',
        'posteriori': f'Posteriori: Normal({round(mu_post,3)}, {round(sigma2_post,3)})'
    }
    return _plot_conjugada_1d(eixo_x, y_priori, y_verossimilhanca, y_posteriori, nomes, 'Distribuição Normal-Normal')


def normal_gama_pdf(x, tau, mu, lambda_, alpha, beta_val):
    """Calcula a PDF da distribuição Normal-Gama."""
    tau = np.maximum(tau, 1e-9)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        term1 = (beta_val**alpha)
        term2 = np.sqrt(lambda_)
        term3 = tau**(alpha-0.5)
        term4 = np.exp(-beta_val*tau)
        term5 = np.exp(-lambda_*tau*(x-mu)**2/2)
        denominador = (math.gamma(alpha)*np.sqrt(2*np.pi))
        densidade = (term1*term2*term3*term4*term5)/denominador
    return np.nan_to_num(densidade)

def xvals_tauvals(mu, lambda_, alpha, beta_val):
    """Calcula os intervalos para os eixos x (mu) e y (tau) para o gráfico Normal-Gama."""
    if alpha<=1 or lambda_<=0 or beta_val<=0:
        return (np.linspace(mu-3, mu+3, 100), np.linspace(0.001, 3, 100))
    mode_tau = (alpha-0.5)/beta_val
    desvio_x = np.sqrt(beta_val/(lambda_*(alpha-1)))
    desvio_tau = np.sqrt(alpha/beta_val**2)
    x_vals = np.linspace(mu-3*desvio_x, mu+3*desvio_x, 100)
    tau_vals = np.linspace(max(mode_tau-3*desvio_tau, 0.001), mode_tau+3*desvio_tau, 100)
    return x_vals, tau_vals

def Normal_Gama_final(mu, lambda_, alpha, beta_val, x, s, n):
    """Gera o gráfico 3D combinado para o modelo Normal-Gama."""
    if any(p is None for p in [mu, lambda_, alpha, beta_val, x, s, n]):
        return go.Figure(layout={"title": "Aguardando todos os parâmetros", "template": "plotly_white"})
    if any(p<=0 for p in [lambda_, alpha, beta_val, n, s]) or alpha<=1 or (n-1)/2<=0:
        return go.Figure(layout={"title": "Parâmetros inválidos", "template": "plotly_white"})

    # Parâmetros da Posteriori
    mu_post = (lambda_*mu+n*x)/(lambda_+n)
    lambda_post = lambda_+n
    alpha_post = alpha+n/2
    beta_post = beta_val+(n*s+(lambda_*n*(x-mu)**2)/(lambda_+n))/2

    # Define os limites dos eixos
    x_priori, tau_priori = xvals_tauvals(mu, lambda_, alpha, beta_val)
    x_vero, tau_vero = xvals_tauvals(x, n, (n-1)/2, n*s/2)
    x_post, tau_post = xvals_tauvals(mu_post, lambda_post, alpha_post, beta_post)

    x_min = min(x_priori[0], x_vero[0], x_post[0])
    x_max = max(x_priori[-1], x_vero[-1], x_post[-1])
    tau_min = min(tau_priori[0], tau_vero[0], tau_post[0])
    tau_max = max(tau_priori[-1], tau_vero[-1], tau_post[-1])

    x_final = np.linspace(x_min, x_max, 70)
    tau_final = np.linspace(tau_min, tau_max, 70)
    X, T = np.meshgrid(x_final, tau_final)

    # Calcula as densidades
    Z_priori = normal_gama_pdf(X, T, mu, lambda_, alpha, beta_val)
    Z_verossimilhanca = normal_gama_pdf(X, T, x, n, (n-1)/2, n*s/2)
    Z_posteriori = normal_gama_pdf(X, T, mu_post, lambda_post, alpha_post, beta_post)

    # Cria o gráfico
    fig = go.Figure(go.Surface(z=Z_priori, x=X, y=T, colorscale=COLORSCALE_BLUE, name="Priori", showscale=False))
    fig.add_trace(go.Surface(z=Z_verossimilhanca, x=X, y=T, colorscale=COLORSCALE_RED, name="Verossimilhança", showscale=False, opacity=0.7))
    fig.add_trace(go.Surface(z=Z_posteriori, x=X, y=T, colorscale=COLORSCALE_GREEN, name="Posteriori", showscale=False, opacity=0.7))

    # Adiciona legenda customizada para superfícies 3D
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(color='blue'), name="Priori"))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(color='red'), name="Verossimilhança"))
    fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(color='green'), name="Posteriori"))

    fig.update_layout(
        title="Distribuição Normal-Gama", showlegend=True,
        scene=dict(xaxis_title="μ", yaxis_title="τ", zaxis_title="Densidade")
    )
    return fig


# ==============================================================================
# SEÇÃO 4: DEFINIÇÃO DO LAYOUT DA APLICAÇÃO
# ==============================================================================

# ------------------------------------------------------------------------------
# Layout da Página: Teorema de Bayes
# ------------------------------------------------------------------------------
def layout_teorema():
    return html.Div(className="container", children=[
        html.Div(style=CARD_STYLE, children=[html.H2("Teorema de Bayes — Quadrado de Bayes (2 hipóteses)", style={'textAlign': 'center'})]),
        html.Div(className="row", children=[
            html.Div(className="col", children=[
                html.Div(style=CARD_STYLE, children=[
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
                html.Div(style=CARD_STYLE, children=[dcc.Graph(id='bayes_graph')]),
                html.Div(id='posterior_text', style={**CARD_STYLE, 'textAlign': 'center', 'fontWeight': 'bold'}),
            ])
        ])
    ])

# ------------------------------------------------------------------------------
# Layout da Página: Análise de Conjugadas
# ------------------------------------------------------------------------------
def layout_conjugadas():
    return html.Div(children=[
        html.Div(style=CARD_STYLE, children=[
            html.H2('Análise de Distribuições Conjugadas', style={'textAlign': 'center'}),
            html.P('Selecione as distribuições da Priori e do Modelo para visualizar a Posteriori resultante.', style={'textAlign': 'center'})
        ]),
        html.Div(className="row", children=[
            # Coluna de Controles (Inputs)
            html.Div(className="col", children=[
                html.Div(style=CARD_STYLE, children=[
                    html.H4('Configuração da Priori'),
                    html.Label('Distribuição da Priori:'),
                    dcc.Dropdown(LISTA_PRIORIS, value="Beta", id="prioris"),
                    html.Div(id="priori-params-div", style={'marginTop': '15px'}),
                ]),
                html.Div(style=CARD_STYLE, children=[
                    html.H4('Configuração da Verossimilhança'),
                    html.Label('Modelo Estatístico:'),
                    dcc.Dropdown(id="verossimilhancas"),
                    html.Div(id="verossimilhanca-params-div", style={'marginTop': '15px'}),
                ]),
            ]),
            # Coluna de Gráficos Individuais
            html.Div(className="col", children=[
                html.Div(style=CARD_STYLE, children=[dcc.Graph(id='densidade_priori')]),
                html.Div(style=CARD_STYLE, id='aparencia_verossimilhanca', children=[dcc.Graph(id='densidade_verossimilhanca')]),
                html.Div(style=CARD_STYLE, children=[dcc.Graph(id='densidade_posteriori')]),
            ])
        ]),
        # Gráfico Combinado e Fórmulas
        html.Div(style=CARD_STYLE, children=[dcc.Graph(id="grafico_conjunto")]),
        html.Div(style=CARD_STYLE, children=[
            html.Button("Ver Fórmulas Gerais", id="botao", n_clicks=0, className="action-button"),
            html.Div(id="texto_formula_div")
        ])
    ])

# ------------------------------------------------------------------------------
# Layout Principal da Aplicação
# ------------------------------------------------------------------------------
header = html.Div(style=HEADER_STYLE, children=[
    html.Img(src="/assets/logopet.png", style=LOGO_STYLE),
    html.H1("Dashboard de Análise Bayesiana", style={'color': COLORS['text'], 'fontSize': '24px', 'margin': '0'}),
    html.Div(style=NAV_BUTTONS_STYLE, children=[
        html.Button("Análise de Conjugadas", id="btn-conjugadas", n_clicks=0, className="nav-button"),
        html.Button("Teorema de Bayes", id="btn-teorema", n_clicks=0, className="nav-button")
    ])
])

app.layout = html.Div([
    header,
    html.Div(id="content-div", style=MAIN_CONTENT_STYLE)
])


# ==============================================================================
# SEÇÃO 5: CALLBACKS DA APLICAÇÃO
# ==============================================================================

# ------------------------------------------------------------------------------
# Callback de Navegação Principal
# ------------------------------------------------------------------------------
@app.callback(
    Output("content-div", "children"),
    [Input("btn-conjugadas", "n_clicks"), Input("btn-teorema", "n_clicks")]
)
def display_page(n_conjugadas, n_teorema):
    """Renderiza a página selecionada pelo usuário."""
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else "btn-conjugadas"

    if button_id == "btn-teorema":
        return layout_teorema()
    return layout_conjugadas()

# ------------------------------------------------------------------------------
# Callbacks para a Página: Teorema de Bayes
# ------------------------------------------------------------------------------
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
    """Atualiza o Quadrado de Bayes e os valores da posteriori."""
    if PA is None or PEA is None or PEB is None:
        raise exceptions.PreventUpdate

    PB = 1-PA
    denom = PEA*PA+PEB*PB
    PAE = (PEA*PA)/denom if denom!=0 else 0
    PBE = (PEB*PB)/denom if denom!=0 else 0

    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=1, y0=0, y1=1, line=dict(color="black"))
    fig.add_shape(type="rect", x0=0, x1=PEA, y0=0, y1=PA, fillcolor='royalblue', opacity=0.7, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=PEB, y0=PA, y1=1, fillcolor="green", opacity=0.7, line_width=0)
    fig.update_layout(
        xaxis=dict(range=[0, 1], showticklabels=False), yaxis=dict(range=[0, 1], showticklabels=False),
        width=400, height=400, title="Quadrado de Bayes", showlegend=False, margin=dict(l=20, r=20, t=40, b=20)
    )
    text = f"Posterior A: {PAE:.2f} | Posterior B: {PBE:.2f}"
    return fig, text

# ------------------------------------------------------------------------------
# Callbacks para a Página: Análise de Conjugadas (Geração de UI)
# ------------------------------------------------------------------------------

@app.callback(
    [Output("verossimilhancas", "options"), Output("verossimilhancas", "value")],
    Input("prioris", "value")
)
def update_verossimilhanca_dropdown(priori):
    """Atualiza as opções de modelo (verossimilhança) com base na priori selecionada."""
    if priori == "Beta":
        options = ["Bernoulli", "Binomial", "Geométrica", "Binomial negativa"]
    elif priori == "Gama":
        options = ["Exponencial", "Poisson", "Gama (b desconhecido)"]
    elif priori == "Normal":
        options = ["Normal (média desconhecida)"]
    elif priori == "Normal-Gama":
        options = ["Normal (média e precisão desconhecidas)"]
    else:
        return [], None
    return options, options[0]

@app.callback(
    Output("priori-params-div", "children"),
    Input("prioris", "value")
)
def render_priori_params(priori):
    """Renderiza os campos de input para os parâmetros da priori."""
    if priori == "Beta":
        return html.Div([
            html.Label("Parâmetro de forma a (> 0):"),
            dcc.Input(id='input-a', type='number', value=2, min=0.001, className="styled-input"),
            html.Label("Parâmetro de forma b (> 0):"),
            dcc.Input(id='input-b', type='number', value=4, min=0.001, className="styled-input"),
            dcc.Input(id='input-c', style={'display': 'none'}, value=3),
            dcc.Input(id='input-d', style={'display': 'none'}, value=2),
        ])
    elif priori == "Gama":
        return html.Div([
            html.Label("Parâmetro de forma α (> 0):"),
            dcc.Input(id='input-a', type='number', value=2, min=0.001, className="styled-input"),
            html.Label("Parâmetro de taxa β (> 0):"),
            dcc.Input(id='input-b', type='number', value=4, min=0.001, className="styled-input"),
            dcc.Input(id='input-c', style={'display': 'none'}, value=3),
            dcc.Input(id='input-d', style={'display': 'none'}, value=2),
        ])
    elif priori == "Normal":
        return html.Div([
            html.Label("Média da priori μ (ℝ):"),
            dcc.Input(id='input-a', type='number', value=0, className="styled-input"),
            html.Label("Variância da priori σ² (> 0):"),
            dcc.Input(id='input-b', type='number', value=1, min=0.001, className="styled-input"),
            dcc.Input(id='input-c', style={'display': 'none'}, value=3),
            dcc.Input(id='input-d', style={'display': 'none'}, value=2),
        ])
    elif priori == "Normal-Gama":
        return html.Div([
            html.Label("Parâmetro μ da priori (ℝ):"),
            dcc.Input(id='input-a', type='number', value=0, className="styled-input"),
            html.Label("Parâmetro λ da priori (> 0):"),
            dcc.Input(id='input-b', type='number', value=1, min=0.001, className="styled-input"),
            html.Label("Parâmetro de forma α (> 1):"),
            dcc.Input(id='input-c', type='number', value=3, min=1.001, className="styled-input"),
            html.Label("Parâmetro de taxa β (> 0):"),
            dcc.Input(id='input-d', type='number', value=2, min=0.001, className="styled-input"),
        ])
    return []

@app.callback(
    Output("verossimilhanca-params-div", "children"),
    Input("verossimilhancas", "value")
)
def render_verossimilhanca_params(verossimilhanca):
    """Renderiza os campos de input para os parâmetros da verossimilhança."""
    if verossimilhanca is None:
        raise exceptions.PreventUpdate

    tamanho_min = 2 if verossimilhanca == "Normal (média e precisão desconhecidas)" else 1
    base_inputs = [
        html.Label(f"Tamanho amostral (n ≥ {tamanho_min}):"),
        dcc.Input(id='input-tamanho', type='number', min=tamanho_min, step=1, value=10, className="styled-input"),
    ]

    input_x = dcc.Input(id='input-x', type='number', value=1, className="styled-input")
    input_x_bernoulli = dcc.Input(id='input-x-bernoulli', type='number', step=0.01, min=0, max=1, value=0.5, className="styled-input")
    input_m = dcc.Input(id='input-m', type='number', min=1, step=1, value=10, className="styled-input")
    input_conhecido = dcc.Input(id='input-conhecido', type='number', value=1, min=0.001, className="styled-input")

    if verossimilhanca == "Bernoulli":
        return html.Div([
            html.Label("Média amostral (0 ≤ x̄ ≤ 1):"), input_x_bernoulli,
            *base_inputs,
            html.Div(input_x, style={'display': 'none'}),
            html.Div(input_m, style={'display': 'none'}),
            html.Div(input_conhecido, style={'display': 'none'}),
        ])

    specific_section, hidden_inputs = [], []
    media_label = "Média amostral (x̄):"

    if verossimilhanca == "Binomial":
        specific_section = [html.Label("Número de ensaios (m):"), input_m]
        hidden_inputs = [html.Div(input_conhecido, style={'display': 'none'})]
    elif verossimilhanca == "Binomial negativa":
        media_label = "Média amostral (x̄ > r):"
        specific_section = [html.Label("Número de sucessos (r):"), input_m]
        hidden_inputs = [html.Div(input_conhecido, style={'display': 'none'})]
    elif verossimilhanca == "Gama (b desconhecido)":
        media_label = "Média amostral (x̄ ≥ 0):"
        specific_section = [html.Label("Parâmetro 'a' conhecido:"), input_conhecido]
        hidden_inputs = [html.Div(input_m, style={'display': 'none'})]
    elif verossimilhanca == "Normal (média desconhecida)":
        media_label = "Média amostral (x̄ ∈ ℝ):"
        specific_section = [html.Label("Variância populacional (σ²) conhecida:"), input_conhecido]
        hidden_inputs = [html.Div(input_m, style={'display': 'none'})]
    elif verossimilhanca == "Normal (média e precisão desconhecidas)":
        media_label = "Média amostral (x̄ ∈ ℝ):"
        specific_section = [html.Label("Variância amostral (s² > 0):"), input_conhecido]
        hidden_inputs = [html.Div(input_m, style={'display': 'none'})]
    else:
        hidden_inputs = [html.Div(input_m, style={'display': 'none'}), html.Div(input_conhecido, style={'display': 'none'})]
        if verossimilhanca == "Geométrica":
            media_label = "Média amostral (x̄ ≥ 1):"
        elif verossimilhanca in ["Exponencial", "Poisson"]:
            media_label = "Média amostral (x̄ ≥ 0):"

    return html.Div([
        html.Label(media_label), input_x,
        *specific_section, *base_inputs,
        html.Div(input_x_bernoulli, style={'display': 'none'}),
        *hidden_inputs
    ])

# ------------------------------------------------------------------------------
# Callbacks para a Página: Análise de Conjugadas (Atualização de Gráficos)
# ------------------------------------------------------------------------------

@app.callback(
    Output('densidade_priori', 'figure'),
    [Input('input-a', 'value'), Input('input-b', 'value'), Input('input-c', 'value'), Input('input-d', 'value'), Input("prioris","value")]
)
def update_priori_graph(a, b, c, d, prioris):
    """Atualiza o gráfico da distribuição a priori."""
    if any(p is None for p in [a, b, prioris]):
        raise exceptions.PreventUpdate
    if prioris == "Beta":
        return plot_beta_distribution(a, b)
    elif prioris == "Gama":
        return plot_gamma_distribution(a, b)
    elif prioris == "Normal":
        return plot_normal_distribution(a, b)
    elif prioris == "Normal-Gama":
        if any(p is None for p in [c, d]):
            raise exceptions.PreventUpdate
        return plot_normal_gama_distribution(a, b, c, d)
    return go.Figure()

@app.callback(
    [Output('densidade_verossimilhanca', 'figure'), Output('aparencia_verossimilhanca', 'style')],
    [Input('input-m', 'value'), Input('input-x', 'value'), Input('input-x-bernoulli','value'), Input('input-tamanho','value'), Input('input-conhecido','value'), Input('verossimilhancas','value')]
)
def update_likelihood_graph(m, x, x_bernoulli, n, conhecido, verossimilhancas):
    """Atualiza o gráfico da função de verossimilhança."""
    if any(p is None for p in [m, x, x_bernoulli, n, conhecido, verossimilhancas]):
        raise exceptions.PreventUpdate

    style = {**CARD_STYLE}
    if verossimilhancas == "Bernoulli":
        return verossimilhanca_bernoulli_aproximada(x_bernoulli, n), style
    elif verossimilhancas == "Binomial":
        return verossimilhanca_binomial_aproximada(x, m, n), style
    elif verossimilhancas == "Geométrica":
        return verossimilhanca_geometrica_aproximada(x, n), style
    elif verossimilhancas == "Binomial negativa":
        return verossimilhanca_binomial_negativa_aproximada(x, m, n), style
    elif verossimilhancas == "Exponencial":
        return verossimilhanca_exponencial_aproximada(x, n), style
    elif verossimilhancas == "Poisson":
        return verossimilhanca_poisson_aproximada(x, n), style
    elif verossimilhancas == "Gama (b desconhecido)":
        return verossimilhanca_gama_aproximada(x, conhecido, n), style
    elif verossimilhancas == "Normal (média desconhecida)":
        return verossimilhanca_normal_aproximada(x, conhecido, n), style
    elif verossimilhancas == "Normal (média e precisão desconhecidas)":
        if n<2 or conhecido<=0:
            raise exceptions.PreventUpdate
        return verossimilhanca_normal_gama_aproximada(x, conhecido, n), style
    return go.Figure(), {'display': 'none'}

@app.callback(
    Output('densidade_posteriori', 'figure'),
    [Input('input-a', 'value'), Input('input-b', 'value'), Input('input-c', 'value'), Input('input-d', 'value'), Input('input-m', 'value'), Input('input-x', 'value'), Input('input-x-bernoulli','value'), Input('input-tamanho', 'value'), Input('input-conhecido', 'value'), Input("prioris","value"), Input("verossimilhancas","value")]
)
def update_posterior_graph(a,b,c,d,m,x,x_bernoulli,n,conhecido,prioris,verossimilhancas):
    """Atualiza o gráfico da distribuição a posteriori."""
    if any(p is None for p in [a,b,c,d,m,x,x_bernoulli,n,conhecido,prioris,verossimilhancas]):
        raise exceptions.PreventUpdate

    if verossimilhancas == "Bernoulli":
        return posteriori_beta(round(a+n*x_bernoulli, 3), round(b+n*(1-x_bernoulli), 3))
    elif verossimilhancas == "Binomial":
        return posteriori_beta(round(n*x+a, 3), round(b+n*(m-x), 3))
    elif verossimilhancas == "Geométrica":
        return posteriori_beta(a+n, round(b+n*(x-1), 3))
    elif verossimilhancas == "Binomial negativa":
        return posteriori_beta(round(a+n*m, 3), round(b+n*(x-m), 3))
    elif verossimilhancas == "Exponencial":
        return posteriori_gama(a+n, round(b+n*x, 3))
    elif verossimilhancas == "Poisson":
        return posteriori_gama(round(n*x+a, 3), b+n)
    elif verossimilhancas == "Gama (b desconhecido)":
        return posteriori_gama(round(a+n*conhecido, 3), round(b+n*x, 3))
    elif verossimilhancas == "Normal (média desconhecida)":
        mu_post = (n*b*x+a*conhecido)/(n*b+conhecido)
        sigma2_post = (conhecido*b)/(n*b+conhecido)
        return posteriori_normal(round(mu_post, 3), round(sigma2_post, 3))
    elif verossimilhancas == "Normal (média e precisão desconhecidas)":
        if c<=1 or n<2 or conhecido<=0:
            raise exceptions.PreventUpdate
        return posteriori_Normal_Gama(a, b, c, d, x, conhecido, n)
    return go.Figure()

@app.callback(
    Output('grafico_conjunto', 'figure'),
    [Input('input-a', 'value'), Input('input-b', 'value'), Input('input-c', 'value'), Input('input-d', 'value'), Input('input-m', 'value'), Input('input-x', 'value'), Input('input-x-bernoulli','value'), Input('input-tamanho', 'value'), Input('input-conhecido', 'value'), Input("prioris","value"), Input("verossimilhancas","value")]
)
def update_final_graph(a,b,c,d,m,x,x_bernoulli,n,conhecido,prioris,verossimilhancas):
    """Atualiza o gráfico combinado final com priori, verossimilhança e posteriori."""
    if any(p is None for p in [a,b,c,d,m,x,x_bernoulli,n,conhecido,prioris,verossimilhancas]):
        raise exceptions.PreventUpdate

    if verossimilhancas == "Bernoulli":
        return beta_bernoulli(a, b, x_bernoulli, n)
    elif verossimilhancas == "Binomial":
        return beta_binomial(a, b, x, m, n)
    elif verossimilhancas == "Geométrica":
        return beta_geometrica(a, b, x, n)
    elif verossimilhancas == "Binomial negativa":
        return beta_binomial_negativa(a, b, x, m, n)
    elif verossimilhancas == "Exponencial":
        return gama_exponencial(a, b, x, n)
    elif verossimilhancas == "Poisson":
        return gama_poisson(a, b, x, n)
    elif verossimilhancas == "Gama (b desconhecido)":
        return gama_gama(a, b, x, conhecido, n)
    elif verossimilhancas == "Normal (média desconhecida)":
        return normal_normal(a, b, x, conhecido, n)
    elif verossimilhancas == "Normal (média e precisão desconhecidas)":
        if c<=1 or n<2 or conhecido<=0:
            raise exceptions.PreventUpdate
        return Normal_Gama_final(a, b, c, d, x, conhecido, n)
    return go.Figure()

# ------------------------------------------------------------------------------
# Callbacks para a Página: Análise de Conjugadas (Fórmulas)
# ------------------------------------------------------------------------------

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
    """Exibe as fórmulas matemáticas (gerais ou aplicadas) ao clicar no botão."""
    if n_clicks is None or n_clicks==0:
        return html.Div()

    # Cliques ímpares: mostram fórmulas GERAIS
    if n_clicks%2!=0:
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
        elif verossimilhancas=="Binomial":
            return dcc.Markdown(r'''
### Fórmulas matemáticas:
#### Priori:
$p \sim Beta(a,b)$
$f(p)=\frac{\Gamma(a+b) p^{a-1}(1-p)^{b-1}}{\Gamma(a)\Gamma(b)}\mathbb{I}_{(0,1)}(p)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Binomial(m, p)$
#### Verossimilhança: $L(p|\mathbf{X})=\left(\prod_{i=1}^{n}\binom{m}{x_i} \right) p^{n\bar{x}} (1 - p)^{n(m-\bar{x})}$
#### Posteriori:
$p|\mathbf{X}\sim Beta(n\bar{x}+a, b+n(m-\bar{x}))$
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
$p|\mathbf{X}\sim Beta(a+n, b+n(\bar{x}-1))$
''', mathjax=True)
        elif verossimilhancas=="Binomial negativa":
            return dcc.Markdown(r'''
### Fórmulas matemáticas:
#### Priori:
$p \sim Beta(a,b)$
$f(p)=\frac{\Gamma(a+b) p^{a-1}(1-p)^{b-1}}{\Gamma(a)\Gamma(b)}\mathbb{I}_{(0,1)}(p)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Binomial Negativa(r, p)$
#### Verossimilhança: $L(p|\mathbf{X})=\left(\prod_{i=1}^n\binom{x_i-1}{r-1}\right) p^{nr}(1-p)^{n(\bar{x}-r)}$
#### Posteriori:
$p|\mathbf{X}\sim Beta(a+nr, b+n(\bar{x}-r))$
''', mathjax=True)
        elif verossimilhancas=="Exponencial":
            return dcc.Markdown(r'''
### Fórmulas matemáticas:
#### Priori:
$\lambda \sim Gama(a,b)$
$f(\lambda)=\frac{b^a \lambda^{a-1} e^{-b \lambda}}{\Gamma(a)} \mathbb{I}_{(0, \infty)}(\lambda)$
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
$f(\lambda)=\frac{b^a \lambda^{a-1} e^{-b \lambda}}{\Gamma(a)} \mathbb{I}_{(0, \infty)}(\lambda)$
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
$f(b)=\frac{b_0^{a_0} b^{a_0-1} e^{-b_0 b}}{\Gamma(a_0)} \mathbb{I}_{(0, \infty)}(b)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Gama(a,b)$
#### Verossimilhança: $L(b|\mathbf{X})=\frac{\left(\prod_{i=1}^n x_i\right)^{a-1} b^{na}e^{-nb\bar{x}}}{\Gamma(a)}$
#### Posteriori:
$p|\mathbf{X}\sim Gama(a_0+na, b_0+n\bar{x})$
''', mathjax=True)
        elif verossimilhancas=="Normal (média desconhecida)":
            return dcc.Markdown(r'''
### Fórmulas matemáticas:
#### Priori:
$\mu \sim Normal(\mu_0,\sigma^2_0)$
$f(\mu) = \frac{1}{\sqrt{2\pi\sigma^2_0}} \exp\left( -\frac{(\mu-\mu_0)^2}{2\sigma^2_0} \right) \mathbb{I}_{(-\infty, \infty)}(\mu)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Normal \left(\mu,\sigma^2 \right)$
#### Verossimilhança: $L(\mu|\mathbf{X}) = \left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)^n \exp\left(-\frac{1}{2\sigma^2}\left(n\mu^2-2\mu n\bar{x}+\sum_{i=1}^{n} x_i^2\right)\right)$
#### Posteriori:
$\mu|\mathbf{X}\sim Normal\left( \frac{n\sigma^2_0\bar{x}+\sigma^2\mu_0}{n\sigma^2_0+\sigma^2}, \frac{\sigma^2\sigma^2_0}{n\sigma^2_0+\sigma^2}\right)$
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
$(\mu,\tau)|\mathbf{X} \sim Normal-Gama\left( \frac{\lambda\mu_0+n\bar{x}}{\lambda+n}, \lambda+n, a+\frac{n}{2}, b+\frac{1}{2}\left( ns^2+\frac{\lambda n\left(\bar{x}-\mu_0\right)^2}{\lambda+n} \right) \right)$
''', mathjax=True)

    # Cliques pares: mostram fórmulas APLICADAS
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
        elif verossimilhancas=="Binomial":
            return dcc.Markdown(fr'''
### Fórmulas matemáticas:
#### Priori:
$p \sim Beta({a},{b})$
$f(p)=\frac{{\Gamma({a+b}) p^{{{a-1}}}(1-p)^{{{b-1}}}}}{{\Gamma({a})\Gamma({b})}}\mathbb{{I}}_{{(0,1)}}(p)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Binomial({m}, p)$
#### Verossimilhança: $L(p|\mathbf{{X}})=\left(\prod_{{i=1}}^{n}\binom{{{m}}}{{x_i}}\right) p^{{{n*x}}} (1-p)^{{{n*(m-x)}}}$
#### Posteriori:
$p|\mathbf{{X}}\sim Beta({n*x+a}, {b+n*(m-x)})$
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
$p|\mathbf{{X}} \sim Beta({a+n}, {b+n*(x-1)})$
''', mathjax=True)
        elif verossimilhancas=="Binomial negativa":
            return dcc.Markdown(fr'''
### Fórmulas matemáticas:
#### Priori:
$p \sim Beta({a},{b})$
$f(p)=\frac{{\Gamma({a+b}) p^{{{a-1}}}(1-p)^{{{b-1}}}}}{{\Gamma({a})\Gamma({b})}}\mathbb{{I}}_{{(0,1)}}(p)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Binomial Negativa({m}, p)$
#### Verossimilhança: $L(p|\mathbf{{X}})=\left(\prod_{{i=1}}^{n}\binom{{x_i-1}}{{{m-1}}}\right) p^{{{n*m}}}(1-p)^{{{n*(x-m)}}}$
#### Posteriori:
$p|\mathbf{{X}} \sim Beta({a+n*m}, {b+n*(x-m)})$
''', mathjax=True)
        elif verossimilhancas=="Exponencial":
            return dcc.Markdown(fr'''
### Fórmulas matemáticas:
#### Priori:
$\lambda \sim Gama({a},{b})$
$f(\lambda)=\frac{{{b**a} \lambda^{{{a-1}}} e^{{-{b}\lambda}}}}{{\Gamma({a})}} \mathbb{{I}}_{{(0, \infty)}}(\lambda)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Exponencial(\lambda)$
#### Verossimilhança: $L(\lambda|\mathbf{{X}})=\lambda^{n}e^{{-{n*x}\lambda }}$
#### Posteriori:
$\lambda|\mathbf{{X}}\sim Gama({a+n}, {b+n*x})$
''', mathjax=True)
        elif verossimilhancas=="Poisson":
            return dcc.Markdown(fr'''
### Fórmulas matemáticas:
#### Priori:
$\lambda \sim Gama({a},{b})$
$f(\lambda)=\frac{{{b**a} \lambda^{{{a-1}}} e^{{-{b}\lambda}}}}{{\Gamma({a})}} \mathbb{{I}}_{{(0, \infty)}}(\lambda)$
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
$f(b)=\frac{{{b**a} b^{{{a-1}}} e^{{-{b}b}}}}{{\Gamma({a})}} \mathbb{{I}}_{{(0, \infty)}}(b)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Gama({conhecido},b)$
#### Verossimilhança: $L(b|\mathbf{{X}})=\frac{{\left(\prod_{{i=1}}^{n} x_i\right)^{{{conhecido-1}}} b^{{{n*conhecido}}}e^{{-{n*x}b}}}}{{\Gamma({conhecido})}}$
#### Posteriori:
$p|\mathbf{{X}}\sim Gama({a+n*conhecido}, {b+n*x})$
''', mathjax=True)
        elif verossimilhancas=="Normal (média desconhecida)":
            return dcc.Markdown(fr'''
### Fórmulas matemáticas:
#### Priori:
$\mu \sim Normal({a},{b})$
$f(\mu) = \frac{{1}}{{\sqrt{{2\cdot{b}\pi}}}} \exp\left(-\frac{{(\mu-{a})^2}}{{2\cdot{b}}}\right)\mathbb{{I}}_{{(-\infty, \infty)}}(\mu)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Normal \left(\mu,{conhecido} \right)$
#### Verossimilhança: $L(\mu|\mathbf{{X}}) = \left(\frac{{1}}{{\sqrt{{2\cdot{conhecido}\pi}}}}\right)^{n} \exp\left(-\frac{{1}}{{2\cdot{conhecido}}}\left({n}\mu^2-{2*n*x}\mu+\sum_{{i=1}}^{n} x_i^2\right)\right)$
#### Posteriori:
$\mu|\mathbf{{X}}\sim Normal\left({((n*b*x+conhecido*a)/(n*b+conhecido)):.3f}, {((conhecido*b)/(n*b+conhecido)):.3f}\right)$
''', mathjax=True)
        else:
            return dcc.Markdown(fr'''
### Fórmulas matemáticas:
#### Priori:
$(\mu,\tau) \sim Normal-Gama({a},{b},{c},{d})$
$f(\mu,\tau)=\frac{{{d**c}\sqrt{{{b}}}\tau^{{{c-0.5}}}e^{{-{d}\tau}}}}{{\Gamma{{({c})}}\sqrt{{2\pi}}}} exp\left(\frac{{{b}\tau\left(\mu-{a}\right)^2}}{{2}}\right)$
#### Modelo estatístico:
$X_1 ... X_n$ condicionalmente independentes e identicamente distribuídos $Normal(\mu,\tau^{{-1}})$
#### Verossimilhança: $L(\mu,\tau|\mathbf{{X}})=\frac{{1}}{{(\sqrt{{2\pi}})^{n}}}\tau^{{{n/2}}}exp(\frac{{-{n*conhecido}\tau}}{{2}})exp\left(\frac{{-{n}\tau\left(\mu-{x}\right)^2}}{{2}}\right)$
#### Posteriori:
$(\mu,\tau)|\mathbf{{X}} \sim Normal-Gama\left({((b*a+n*x)/(b+n)):.3f}, {b+n}, {c+n/2},{(d+0.5*(n*conhecido+(b*n*(x-a)**2)/(b+n))):.3f}\right)$
''', mathjax=True)

@app.callback(
    Output("botao", "children"),
    [Input("botao", "n_clicks")]
)
def update_button_text(n_clicks):
    """Alterna o texto do botão de fórmulas."""
    if n_clicks is None or n_clicks%2==0:
        return "Ver Fórmulas Gerais"
    return "Aplicar valores nos campos"

# ==============================================================================
# SEÇÃO 6: EXECUÇÃO DO SERVIDOR
# ==============================================================================

if __name__ == '__main__':
    app.run_server(debug=True)
