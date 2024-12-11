import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import random


print("Milena Musiienko / KM-12 / Lab-1")

env = gym.make("FrozenLake-v1", is_slippery=True)
env = env.unwrapped

# Параметри
gamma = 0.5
theta = 1e-6  # Критерій зупинки
n_states = env.observation_space.n
n_actions = env.action_space.n

policy = np.ones((n_states, n_actions)) / n_actions


print("\n2a. Обчислити функцію ціни стану 𝑣𝜋1(𝑠) для рівноймовірної (випадкової) стратегії 𝜋1", 
      "при параметрі 𝛾 = 0.5 за допомогою ітераційного алгоритму Оцінювання стратегії")

# Оцінювання стратегії
def policy_evaluation(env, policy, gamma, theta):
    value_table = np.zeros(n_states)
    while True:
        delta = 0
        for s in range(n_states):
            v = 0
            for a in range(n_actions):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += policy[s, a] * prob * (reward + gamma * value_table[next_state])
            delta = max(delta, abs(v - value_table[s]))
            value_table[s] = v
        if delta < theta:
            break
    return value_table

policy_evaluation_value = policy_evaluation(env, policy, gamma, theta)
policy_evaluation_value = policy_evaluation_value.reshape((4, 4))
print("Policy Evaluation:\n", policy_evaluation_value)


print("\n2b. Обчислити функцію ціни стану 𝑣𝜋1(𝑠) для рівноймовірної (випадкової) стратегії 𝜋1", 
      "при параметрі 𝛾 = 0.5 за допомогою розв’язання системи рівнянь Белмана для функції",
      "ціни стану відносно невідомих значень 𝑥𝑖 = 𝑣𝜋1(𝑠𝑖)")

# Рівняння Белмана
def bellman(env, policy, gamma):
    A = np.zeros((n_states, n_states))
    b = np.zeros(n_states)
    for s in range(n_states):
        for a in range(n_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                A[s, next_state] -= gamma * prob * policy[s, a]
                b[s] += prob * reward * policy[s, a]
        A[s, s] += 1
    v = np.linalg.solve(A, b)
    return v

bellman_value = bellman(env, policy, gamma)
bellman_value = bellman_value.reshape((4, 4))
print("Bellman:\n", bellman_value)


print("\n3. Використовуючи знайдені значення функції ціни стану 𝑣𝜋1(𝑠) та рівняння Белмана",
      "для функції ціни дії-стану, оцінити функцію ціни дії-стану 𝑞𝜋1(𝑠𝑖,𝑎𝑗)")

# Функція ціни дії-стану
def action_value(env, value_table, gamma):
    action_value = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            q_sa = 0
            for prob, next_state, reward, done in env.P[s][a]:
                q_sa += prob * (reward + gamma * value_table[next_state])
            action_value[s, a] = q_sa
    return action_value

value_table_flat = policy_evaluation_value.flatten()
action_value = action_value(env, value_table_flat, gamma)
print("Action-Value Function:\n", action_value)


print("\n4. Створити функцію equiprobable, результатом якої є номер дії.",
      "Дія обирається випадковим чином з множини допустимих дій")

def equiprobable():
    actions = list(range(env.action_space.n))
    return random.choice(actions)

action = equiprobable()
print(f"Action number:", action)


print("\n5. Створити  функцію  get_episode,  яка  приймає  у  якості  аргументу екземпляр середовища,", 
      "а результатом функції є епізод, тобто список кортежів,  кожен  з  яких  зберігає  всі  характеристики",
      "кожного  кроку агента  (тобто  попередній  стан,  дію,  винагороду,  поточний  стан, значення параметрів terminated та truncated).",
      "Вибір  агентом  дії  у  кожному  стані  на  даному  етапі  реалізуйте  на основі  функції  equiprobable,",
      "або,  іншими  словами,  на  основі рівноймовірної (випадкової) стратегії 𝜋1.")

# Функція для створення епізоду
def get_episode(env):
    episode = [] 
    state = env.reset()  # Ініціалізувати середовище та отримати початковий стан
    while True:
        action = equiprobable()
        next_state, reward, terminated, truncated, info = env.step(action)
        episode.append((state, action, reward, next_state, terminated, truncated))
        if terminated or truncated:  # Якщо епізод завершено
            break
        state = next_state
    return episode

episode = get_episode(env)
print("\nEpisode:")
for step, (state, action, reward, next_state, terminated, truncated) in enumerate(episode):
    print(f"Step {step + 1}: State={state}, Action={action}, Reward={reward}, "
          f"Next state={next_state}, Terminated={terminated}, Truncated={truncated}")
    

print("\n6. Виконати 75 епізодів за допомогою функції get_episode. Виведіть на екран два графіки: винагорода та тривалість епізоду.")

n_episodes = 75
total_rewards = []
total_durations = []

for i in range(n_episodes):
    episode = get_episode(env)
    reward = sum(step[2] for step in episode)
    total_rewards.append(reward)
    total_durations.append(len(episode))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].plot(range(1, n_episodes + 1), total_rewards, color='r')
axes[0].set_title('Episode Rewards')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Reward')

axes[1].plot(range(1, n_episodes + 1), total_durations, color='b')
axes[1].set_title('Episode Durations')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Steps')

plt.savefig("task6.png", dpi=300, bbox_inches='tight')


print("\n7. Реалізувати метод Ітерації стратегії (Policy Iteration) для знаходження оптимальної стратегії за заданою початковою")

def policy_iteration(env, gamma, theta):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    value_table = np.zeros(n_states)

    # Оцінювання стратегії
    while True:
        delta = 0
        for s in range(n_states):
            q_values = []
            for a in range(n_actions):
                q_sa = 0
                for prob, next_state, reward, done in env.P[s][a]:
                    q_sa += prob * (reward + gamma * value_table[next_state])
                q_values.append(q_sa)
            max_value = max(q_values)
            delta = max(delta, abs(max_value - value_table[s]))
            value_table[s] = max_value
        if delta < theta:
            break

    # Покращення стратегії
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        q_values = []
        for a in range(n_actions):
            q_sa = 0
            for prob, next_state, reward, done in env.P[s][a]:
                q_sa += prob * (reward + gamma * value_table[next_state])
            q_values.append(q_sa)
        best_action = np.argmax(q_values)
        policy[s, best_action] = 1.0

    return value_table, policy


optimal_value, optimal_policy = policy_iteration(env, gamma, theta)
optimal_value = optimal_value.reshape((4, 4))

print("Optimal Policy:")
print(optimal_policy)
print("\nOptimal Value Function:")
print(optimal_value)

plt.figure(figsize=(6, 6))
sns.heatmap(optimal_value, annot=True, fmt=".2f")
plt.title("Optimal Value Function")
plt.savefig("task7.png", dpi=300, bbox_inches='tight')


print("\n8. Оцінити  оптимальну  стратегію  𝜋∗  та  функцію  ціни  стану  𝑣∗(𝑠)  для заданого  середовища  за",
      "допомогою  методу  Ітерації  стратегії, використовуючи  в  якості  початкових  параметрів  стратегію  𝜋1  та",
      "функцію  ціни  𝑣𝜋1(𝑠).  Порівняйте  отриману  функцію  ціни  𝑣∗(𝑠)  з функцією 𝑣𝜋1(𝑠) з завдання N2.")

difference = optimal_value - policy_evaluation_value

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.heatmap(optimal_value, annot=True, fmt=".2f", ax=axes[0])
axes[0].set_title("Optimal v*(s)")
sns.heatmap(policy_evaluation_value, annot=True, fmt=".2f", ax=axes[1])
axes[1].set_title("Iterative v_{π1}(s)")
sns.heatmap(difference, annot=True, fmt=".2f", ax=axes[2])
axes[2].set_title("Difference (v*(s) - v_{π1}(s))")

plt.savefig("task8.png", dpi=300, bbox_inches='tight')


print("\n9. Виконати 12 епізодів за допомогою функції get_episode зі стратегією 𝜋∗.",
      "Виведіть на екран два графіки: винагорода та тривалість епізоду. Порівняйте результати з відповідними результатами завдання 6.")

n_episodes = 12
total_rewards_optimal = []
total_durations_optimal = []

def optimal_episodes(env, policy):
    episode = [] 
    state = env.reset()[0]  # Ініціалізувати середовище та отримати початковий стан
    while True:
        action = np.argmax(policy[state])  # Використання оптимальної стратегії
        next_state, reward, terminated, truncated, info = env.step(action)
        episode.append((state, action, reward, next_state, terminated, truncated))
        if terminated or truncated:
            break
        state = next_state
    return episode

for i in range(n_episodes):
    episode = optimal_episodes(env, optimal_policy)
    total_rewards_optimal.append(sum(step[2] for step in episode))
    total_durations_optimal.append(len(episode))


fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].plot(range(1, n_episodes + 1), total_rewards_optimal, color='g', label='Optimal Policy')
axes[0].set_title('Episode Rewards (Optimal Policy)')
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Reward')

axes[1].plot(range(1, n_episodes + 1), total_durations_optimal, color='c', label='Optimal Policy')
axes[1].set_title('Episode Durations (Optimal Policy)')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Steps')

plt.savefig("task9.png", dpi=300, bbox_inches='tight')


print("\n10. Використовуючи знайдені значення оптимальної функції ціни стану 𝑣∗(𝑠) та  рівняння  Белмана",
      "для  функції  ціни  дії-стану,  оцінити оптимальну функцію ціни дії-стану 𝑞∗(𝑠𝑖,𝑎𝑗). Порівняйте результати з результатами завдання N3")

def optimal_action_value(env, optimal_value_table, gamma):
    """Обчислення оптимальної функції ціни дії-стану q*(s, a)."""
    optimal_action = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in range(n_actions):
            q_sa = 0
            for prob, next_state, reward, done in env.P[s][a]:
                q_sa += prob * (reward + gamma * optimal_value_table[next_state])
            optimal_action[s, a] = q_sa
    return optimal_action


optimal_action = optimal_action_value(env, optimal_value.flatten(), gamma)
difference_q = optimal_action - action_value

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.heatmap(optimal_action, annot=True, fmt=".2f", ax=axes[0])
axes[0].set_title("Optimal Action-Value Function q*")
sns.heatmap(action_value, annot=True, fmt=".2f", ax=axes[1])
axes[1].set_title("Action-Value Function q_π1")
sns.heatmap(difference_q, annot=True, fmt=".2f", ax=axes[2])
axes[2].set_title("Difference (q* - q_π1)")

plt.savefig("task10.png", dpi=300, bbox_inches='tight')


print("\n11. Створити  функцію  eps_greedy_policy,  аргументами  якої  є  масив значень функції",
      "ціни дії-стану 𝑞(𝑠,𝑎) та параметр 𝜀, та результатом є номер  дії,  обраний  з  множини  номерів",
      "допустимих  дій  допомогою методу 𝜀-жадібної стратегії (𝜀-greedy policy).")

def eps_greedy_policy(q_values, epsilon):
    n_actions = len(q_values)
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    else:
        return np.argmax(q_values)

q_values_example = [0.4, 0.2, 0.5, 0.9]
epsilon_example = 0.1
action = eps_greedy_policy(q_values_example, epsilon_example)
print(f"Action: {action}")
