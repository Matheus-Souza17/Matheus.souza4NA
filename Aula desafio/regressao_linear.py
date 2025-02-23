# Função para calcular a média de uma lista de valores
def mean(values):
    return sum(values) / len(values)  # Soma todos os valores e divide pelo total de elementos

# Função para calcular a covariância entre duas variáveis (x e y)
def covariance(x, y, mean_x, mean_y):
    return sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))  # Soma os produtos das diferenças entre cada ponto e a média

# Função para calcular a variância de uma variável
def variance(values, mean_value):
    return sum((x - mean_value) ** 2 for x in values)  # Soma dos quadrados das diferenças entre os valores e a média

# Função principal para calcular os coeficientes da regressão linear (b0 e b1)
def linear_regression(x, y):
    mean_x, mean_y = mean(x), mean(y)  # Calcula a média de x e y
    b1 = covariance(x, y, mean_x, mean_y) / variance(x, mean_x)  # Calcula o coeficiente angular (b1)
    b0 = mean_y - b1 * mean_x  # Calcula o coeficiente linear (b0)
    return b0, b1  # Retorna os coeficientes da reta

# Função para fazer previsões usando a equação da reta: y = b0 + b1 * x
def predict(x, b0, b1):
    return [b0 + b1 * xi for xi in x]  # Para cada valor de x, calcula o valor previsto de y

# Dados de entrada para treinar a regressão
x = [1, 2, 3, 4, 5]  # Valores de entrada (variável independente)
y = [2, 3, 5, 6, 8]  # Valores de saída (variável dependente)

# Treina o modelo e obtém os coeficientes b0 e b1
b0, b1 = linear_regression(x, y)
print(f"Coeficientes: b0 = {b0}, b1 = {b1}")  # Exibe os coeficientes da regressão

# Novos valores de x para testar o modelo
x_test = [6, 7, 8]

# Faz previsões para os novos valores de x
predictions = predict(x_test, b0, b1)
print(f"Previsões para {x_test}: {predictions}")  # Exibe as previsões
