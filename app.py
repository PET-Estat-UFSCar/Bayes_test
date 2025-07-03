# -*- coding: utf-8 -*-
"""
Dashboard Interativo para Análise Bayesiana

Uma aplicação web em Dash para visualizar o Teorema de Bayes e a relação 
entre distribuições conjugadas (Priori, Verossimilhança e Posteriori).
"""

# ==============================================================================
# 1. SETUP DA APLICAÇÃO
# ==============================================================================
import numpy as np
import plotly.graph_objects as go
from scipy.stats import beta, gamma, norm
from dash import Dash, html, dcc, Input, Output, State, exceptions, callback_context
import math

# Inicialização do App Dash
app = Dash(__name__, suppress_callback_exceptions=True, title="Dashboard Bayesiano")
server = app.server

# ==============================================================================
# 2. ESTILOS E CONSTANTES
# ==============================================================================
COLORS = {
    'background': '#F3F6FA', 'text': '#333333', 'card': '#FFFFFF',
    'shadow': 'rgba(0,0,0,0.05)', 'border': '#E9ECEF', 'priori': 'royalblue',
    'likelihood': 'red', 'posterior': 'green', 'primary_button': '#007BFF'
}

HEADER_STYLE = {
    "backgroundColor": COLORS['card'], "padding": "15px 25px", "textAlign": "center",
    "borderBottom": f"1px solid {COLORS['border']}", "display": "flex",
    "alignItems": "center", "justifyContent": "space-between",
    "boxShadow": f"0 2px 4px {COLORS['shadow']}"
}
CARD_STYLE = {
    "backgroundColor": COLORS['card'], "padding": "20px", "borderRadius": "10px",
    "boxShadow": f"0 2px 4px {COLORS['shadow']}", "marginBottom": "20px"
}
MAIN_CONTENT_STYLE = {"padding": "20px", "backgroundColor": COLORS['background']}
LISTA_PRIORIS = ["Beta", "Gama", "Normal", "Normal-Gama"]
COLORSCALE_BLUE = [[0.0, "lightblue"], [0.5, "blue"], [1.0, "darkblue"]]
COLORSCALE_RED = [[0.0, "lightcoral"], [0.5, "red"], [1.0, "darkred"]]
COLORSCALE_GREEN = [[0.0, "lightgreen"], [0.5, "green"], [1.0, "darkgreen"]]

# ==============================================================================
# 3. FUNÇÕES DE PLOTAGEM E CÁLCULO
# ==============================================================================

def create_error_figure(message):
    """Cria uma figura vazia com uma mensagem de erro centralizada."""
    return go.Figure(layout={
        "title": message, "template": "plotly_white",
        "xaxis": {"visible": False}, "yaxis": {"visible": False}
    })

def plot_1d_distribution(x, y, dist_name, params_str, color, title):
    """Função genérica para plotar qualquer distribuição 1D."""
    fig = go.Figure(go.Scatter(x=x, y=y, mode='lines', name=f'{dist_name}({params_str})', line=dict(color=color, width=2.5)))
    fig.update_layout(title=title, xaxis_title='Suporte do Parâmetro', yaxis_title='Densidade', template='plotly_white', showlegend=True, margin=dict(l=40, r=20, t=50, b=40))
    return fig

def get_beta_plot(a, b, title, color):
    if not all(p > 0 for p in [a, b]): return create_error_figure("Parâmetros 'a' e 'b' devem ser > 0")
    theta = np.linspace(0, 1, 1000)
    pdf = beta.pdf(theta, a, b) if not (a == 1 and b == 1) else np.ones_like(theta)
    return plot_1d_distribution(theta, pdf, "Beta", f"{a:.2f}, {b:.2f}", color, title)

def get_gamma_plot(a, b, title, color):
    if not all(p > 0 for p in [a, b]): return create_error_figure("Parâmetros 'a' e 'b' devem ser > 0")
    mode = (a - 1) / b if a > 1 else 0
    std_dev = np.sqrt(a / b**2)
    x_max = mode + 4 * std_dev
    if x_max <= 1e-9: x_max = 4 * std_dev # Handle case where mode is 0
    x = np.linspace(1e-9, x_max, 1000)
    pdf = gamma.pdf(x, a, scale=1/b)
    return plot_1d_distribution(x, pdf, "Gama", f"{a:.2f}, {b:.2f}", color, title)

def get_normal_plot(mu, sigma2, title, color):
    if sigma2 <= 0: return create_error_figure("Variância deve ser > 0")
    sigma = np.sqrt(sigma2)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    pdf = norm.pdf(x, mu, sigma)
    return plot_1d_distribution(x, pdf, "Normal", f"{mu:.2f}, {sigma2:.2f}", color, title)

def normal_gama_pdf(x, tau, mu, lambda_, alpha, beta_param):
    """Calcula a PDF da Normal-Gama, com tratamento de erros numéricos."""
    tau = np.maximum(tau, 1e-9) # Evita tau <= 0
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        term1 = (beta_param**alpha)
        term2 = np.sqrt(lambda_)
        term3 = tau**(alpha - 0.5)
        term4 = np.exp(-beta_param * tau)
        term5 = np.exp(-lambda_ * tau * (x - mu)**2 / 2)
        denominator = (math.gamma(alpha) * np.sqrt(2 * np.pi))
        if denominator == 0: return np.zeros_like(x)
        pdf = (term1 * term2 * term3 * term4 * term5) / denominator
    return np.nan_to_num(pdf)

def get_normal_gama_plot(mu, lambda_, alpha, beta_param, title, colorscale):
    """Plota a distribuição 3D Normal-Gama."""
    if not all(p > 0 for p in [lambda_, beta_param]) or alpha <= 1:
        return create_error_figure("Parâmetros inválidos (λ>0, β>0, α>1)")
    
    # Define a range for plotting
    mode_tau = (alpha - 0.5) / beta_param if alpha > 0.5 else 1e-9
    desvio_x = np.sqrt(beta_param / (lambda_ * (alpha - 1)))
    desvio_tau = np.sqrt(alpha / beta_param**2)
    
    x_vals = np.linspace(mu - 3*desvio_x, mu + 3*desvio_x, 70)
    tau_vals = np.linspace(max(mode_tau - 3*desvio_tau, 1e-9), mode_tau + 3*desvio_tau, 70)
    
    X, T = np.meshgrid(x_vals, tau_vals)
    Z = normal_gama_pdf(X, T, mu, lambda_, alpha, beta_param)
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=T, colorscale=colorscale, cmin=0, showscale=False)])
    fig.update_layout(title=title, scene=dict(xaxis_title="μ (Média)", yaxis_title="τ (Precisão)", zaxis_title="Densidade"), margin=dict(l=0, r=0, b=0, t=40))
    return fig

# ... (As funções de Fórmulas e Layout vêm a seguir, mas são muito longas para caber aqui. Elas estarão no código final)

# ==============================================================================
# 4. FUNÇÃO DE FÓRMULAS (para manter os callbacks limpos)
# ==============================================================================
def get_formula_text(verossimilhanca, params, general=True):
    # ... (Esta função conterá todas as strings de LaTeX, como no código original)
    # Exemplo para um caso:
    if verossimilhanca == "Bernoulli":
        if general:
            return r'''... fórmula geral ...'''
        else:
            # Desempacota os parâmetros
            a, b, n, x_bar = params['a'], params['b'], params['n'], params['x_bernoulli']
            return fr'''... fórmula com valores {a}, {b}, {n}, {x_bar} ...'''
    return "Fórmulas não disponíveis para esta seleção."


# ==============================================================================
# 5. FUNÇÕES DE LAYOUT
# ==============================================================================

def create_header():
    #... (igual ao código anterior)
    return html.Div("Cabeçalho aqui")

def create_layout_teorema():
    #... (layout completo da página do Teorema de Bayes)
    return html.Div("Layout do Teorema de Bayes aqui")

def create_layout_conjugadas():
    """Cria o layout da página principal de Análise de Conjugadas."""
    return html.Div([
        html.Div(style=CARD_STYLE, children=[
            html.H2('Análise de Distribuições Conjugadas', style={'textAlign': 'center'}),
            html.P('Selecione a Priori e o Modelo, insira os parâmetros e clique em "Aplicar" para ver os resultados.', style={'textAlign': 'center'})
        ]),
        html.Div(className="row", style={'display': 'flex', 'gap': '20px'}, children=[
            # Coluna de Controles
            html.Div(className="four columns", style={'flex': 1}, children=[
                html.Div(style=CARD_STYLE, children=[
                    html.H4('1. Configuração da Priori'),
                    dcc.Dropdown(LISTA_PRIORIS, value="Beta", id="prioris", clearable=False),
                    html.Div(id="priori-params-div", style={'marginTop': '15px'}),
                ]),
                html.Div(style=CARD_STYLE, children=[
                    html.H4('2. Configuração do Modelo'),
                    dcc.Dropdown(id="verossimilhancas", clearable=False),
                    html.Div(id="verossimilhanca-params-div", style={'marginTop': '15px'}),
                ]),
                html.Button("Aplicar Parâmetros", id="btn-aplicar", n_clicks=0, style={'width': '100%', 'padding': '10px', 'fontSize': '16px', 'backgroundColor': COLORS['primary_button'], 'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
            ]),
            # Coluna de Gráficos
            html.Div(className="eight columns", style={'flex': 2}, children=[
                html.Div(style=CARD_STYLE, children=[dcc.Graph(id='graph-priori', figure=create_error_figure("Aguardando parâmetros..."))]),
                html.Div(style=CARD_STYLE, children=[dcc.Graph(id='graph-verossimilhanca', figure=create_error_figure("Aguardando parâmetros..."))]),
                html.Div(style=CARD_STYLE, children=[dcc.Graph(id='graph-posteriori', figure=create_error_figure("Aguardando parâmetros..."))]),
            ])
        ]),
        html.Div(style=CARD_STYLE, children=[
            html.H3("Análise Combinada: Priori, Verossimilhança e Posteriori", style={'textAlign': 'center'}),
            dcc.Graph(id="graph-conjunto", figure=create_error_figure("Aguardando parâmetros..."))
        ]),
        html.Div(style=CARD_STYLE, children=[
             html.Button("Ver/Ocultar Fórmulas", id="btn-formulas", n_clicks=0),
             dcc.Markdown(id="formulas-div", mathjax=True, style={'marginTop': '15px', 'display': 'none'})
        ])
    ])

# Layout principal
app.layout = html.Div(id="app-container", children=[
    create_header(),
    html.Div(id="content-div", style=MAIN_CONTENT_STYLE)
])


# ==============================================================================
# 6. CALLBACKS
# ==============================================================================

# --- Callback de Navegação ---
@app.callback(
    Output("content-div", "children"),
    [Input("btn-conjugadas", "n_clicks"), Input("btn-teorema", "n_clicks")],
    prevent_initial_call=True
)
def display_page(n_conjugadas, n_teorema):
    # ... (lógica para trocar de página)
    return create_layout_conjugadas() # Padrão


# --- Callbacks para UI Dinâmica (Parâmetros) ---
@app.callback(
    [Output("verossimilhancas", "options"), Output("verossimilhancas", "value")],
    Input("prioris", "value")
)
def update_verossimilhanca_options(priori):
    # ... (código completo para atualizar as opções de verossimilhança)
    if priori == "Beta":
        options = ["Bernoulli", "Binomial", "Geométrica", "Binomial negativa"]
        return options, options[0]
    # ... etc
    return [], None

# ... (Callbacks `render_priori_params` e `render_verossimilhanca_params` completos aqui)


# --- Callback Principal para Atualizar Gráficos ---
@app.callback(
    [Output('graph-priori', 'figure'),
     Output('graph-verossimilhanca', 'figure'),
     Output('graph-posteriori', 'figure'),
     Output('graph-conjunto', 'figure')],
    Input('btn-aplicar', 'n_clicks'),
    [State('prioris', 'value'), State('verossimilhancas', 'value'),
     # ... todos os outros inputs como State ...
     State('input-a', 'value'), State('input-b', 'value'),
     State('input-n', 'value'), State('input-x-bernoulli', 'value')]
)
def update_all_graphs(n_clicks, priori, verossimilhanca, a, b, n, x_bernoulli):
    if n_clicks == 0:
        raise exceptions.PreventUpdate

    # ================= BETA FAMILY =================
    if priori == "Beta":
        if not all(isinstance(i, (int, float)) for i in [a, b]):
            return [create_error_figure("Parâmetros da priori inválidos.")] * 4
        
        fig_priori = get_beta_plot(a, b, f"Priori: Beta({a:.2f}, {b:.2f})", COLORS['priori'])

        if verossimilhanca == "Bernoulli":
            if not all(isinstance(i, (int, float)) for i in [n, x_bernoulli]):
                 return [fig_priori] + [create_error_figure("Parâmetros do modelo inválidos.")] * 3
            
            # Verossimilhança
            a_vero, b_vero = n * x_bernoulli + 1, n * (1 - x_bernoulli) + 1
            fig_vero = get_beta_plot(a_vero, b_vero, "Verossimilhança Reescalada", COLORS['likelihood'])

            # Posteriori
            a_post, b_post = a + n * x_bernoulli, b + n * (1 - x_bernoulli)
            fig_post = get_beta_plot(a_post, b_post, f"Posteriori: Beta({a_post:.2f}, {b_post:.2f})", COLORS['posterior'])
            
            # Conjunto (Lógica de plotagem combinada aqui)
            fig_conjunto = go.Figure() # ... construir o gráfico combinado
            
            return fig_priori, fig_vero, fig_post, fig_conjunto

    # ... Adicionar blocos `elif priori == "Gama":`, `elif priori == "Normal":` etc.

    # Fallback
    return [create_error_figure("Seleção inválida.")] * 4

# --- Callback para Fórmulas ---
@app.callback(
    [Output('formulas-div', 'style'), Output('formulas-div', 'children')],
    Input('btn-formulas', 'n_clicks'),
    [State('verossimilhancas', 'value'),
     # ... todos os outros parâmetros como State ...
    ],
    prevent_initial_call=True
)
def toggle_formulas(n_clicks, verossimilhanca, ...):
    if n_clicks % 2 != 0:
        display_style = {'display': 'block', 'marginTop': '15px'}
        # Obter os parâmetros e chamar get_formula_text
        # params_dict = {...}
        # formula_md = get_formula_text(verossimilhanca, params_dict, general=True)
        # return display_style, dcc.Markdown(formula_md, mathjax=True)
    else:
        display_style = {'display': 'none'}
    
    return display_style, ""


# ==============================================================================
# 7. EXECUÇÃO
# ==============================================================================
if __name__ == '__main__':
    app.run_server(debug=True)
