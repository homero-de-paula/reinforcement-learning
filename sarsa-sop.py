import numpy as np
import gzip

# Função para carregar e processar o arquivo .sop.gz
def load_sop_instance(filepath):
    with gzip.open(filepath, 'rt') as f:
        data = f.read().splitlines()
    
    # Processar os dados - baseado em uma estrutura comum de SOP
    num_tasks = int(data[0].split()[0])  # Número de tarefas
    cost_matrix = []
    
    # Lendo a matriz de custos
    for line in data[1:num_tasks+1]:
        cost_matrix.append([float(x) for x in line.split()])
    
    # Lendo as precedências
    precedences = []
    for line in data[num_tasks+1:]:
        precedences.append([int(x) for x in line.split()])
    
    return np.array(cost_matrix), precedences

# Algoritmo SARSA
def sarsa(num_episodes, alpha, gamma, epsilon, cost_matrix, precedences):
    num_tasks = cost_matrix.shape[0]
    q_table = np.zeros((num_tasks, num_tasks))  # Tabela Q
    
    for episode in range(num_episodes):
        state = 0  # Estado inicial
        done = False
        action = epsilon_greedy(q_table, state, epsilon, num_tasks)
        
        while not done:
            next_state = action
            next_action = epsilon_greedy(q_table, next_state, epsilon, num_tasks)
            reward = -cost_matrix[state][next_state]  # Recompensa negativa

            # Atualizar a tabela Q com SARSA
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * q_table[next_state, next_action] - q_table[state, action])

            state = next_state
            action = next_action

            # Checar se todas as tarefas foram completadas
            if len(precedences[state]) == 0:
                done = True

    return q_table

# Função para escolher uma ação usando epsilon-greedy
def epsilon_greedy(q_table, state, epsilon, num_tasks):
    if np.random.rand() < epsilon:
        return np.random.randint(0, num_tasks)  # Exploração
    else:
        return np.argmax(q_table[state])  # Exploração

# Função para rodar os experimentos e gerar gráficos
import matplotlib.pyplot as plt

def plot_performance(q_table):
    plt.figure(figsize=(10, 6))
    plt.imshow(q_table, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.title("Matriz Q após treinamento SARSA")
    plt.show()

# Carregar a instância e rodar SARSA
filepath = '/mnt/data/ry48p.4.sop.gz'
cost_matrix, precedences = load_sop_instance(filepath)

# Hiperparâmetros
alpha = 0.1  # Taxa de aprendizado
gamma = 0.9  # Fator de desconto
epsilon = 0.1  # Taxa de exploração
num_episodes = 1000

# Rodar o algoritmo SARSA
q_table = sarsa(num_episodes, alpha, gamma, epsilon, cost_matrix, precedences)

# Plotar o desempenho
plot_performance(q_table)
