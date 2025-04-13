import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os

# ---------------- Traffic Environment ---------------- #
class TrafficIntersection:
    def __init__(self, inflow_prob=0.5):
        self.cars_ns = np.zeros(5, dtype=int)
        self.cars_ew = np.zeros(5, dtype=int)
        self.inflow_prob = inflow_prob
        self.total_cleared = 0

    def reset(self):
        self.cars_ns[:] = 0
        self.cars_ew[:] = 0
        self.total_cleared = 0
        return self.get_state()

    def step(self, action):
        cleared = 0

        if action == 0:
            cleared = self.cars_ns[-1]
            self.cars_ns[1:] = self.cars_ns[:-1]
            self.cars_ns[0] = np.random.choice([0, 1], p=[1 - self.inflow_prob, self.inflow_prob])
            self.cars_ew += np.random.choice([0, 1], size=5, p=[0.7, 0.3])
        else:
            cleared = self.cars_ew[-1]
            self.cars_ew[1:] = self.cars_ew[:-1]
            self.cars_ew[0] = np.random.choice([0, 1], p=[1 - self.inflow_prob, self.inflow_prob])
            self.cars_ns += np.random.choice([0, 1], size=5, p=[0.7, 0.3])

        self.cars_ns = np.clip(self.cars_ns, 0, 1)
        self.cars_ew = np.clip(self.cars_ew, 0, 1)

        self.total_cleared += cleared
        reward = cleared
        state = self.get_state()
        return state, reward, reward > 0

    def get_state(self):
        return {
            "NS": self.cars_ns.sum(),
            "EW": self.cars_ew.sum(),
            "NS_array": self.cars_ns.copy(),
            "EW_array": self.cars_ew.copy()
        }

# ---------------- Q-Learning Functions ---------------- #
def get_discrete_state(state):
    return (min(state["NS"], 5), min(state["EW"], 5))

def select_action(q_table, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice([0, 1])
    return np.argmax(q_table[state])

def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    old_value = q_table[state][action]
    future_max = np.max(q_table[next_state])
    new_value = old_value + alpha * (reward + gamma * future_max - old_value)
    q_table[state][action] = new_value

# ---------------- Training ---------------- #
def train_agents(n_intersections, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    q_tables = []
    all_rewards = []
    inflow_probs = np.linspace(0.3, 0.7, n_intersections)

    for i in range(n_intersections):
        q_table = {(ns, ew): [0, 0] for ns in range(6) for ew in range(6)}
        rewards_per_episode = []
        env = TrafficIntersection(inflow_prob=inflow_probs[i])

        for ep in range(episodes):
            state = get_discrete_state(env.reset())
            total_reward = 0
            for step in range(50):
                action = select_action(q_table, state, epsilon)
                next_state, reward, _ = env.step(action)
                next_state = get_discrete_state(next_state)
                update_q_table(q_table, state, action, reward, next_state, alpha, gamma)
                state = next_state
                total_reward += reward
            rewards_per_episode.append(total_reward)

        q_tables.append(q_table)
        all_rewards.append(rewards_per_episode)

    return q_tables, all_rewards

# ---------------- Demo Logic ---------------- #
def draw_intersection(state, action, step, idx, countdown, highlight_clear):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor("#2e2e2e")
    ax.add_patch(plt.Rectangle((0, 4.5), 10, 1, color="#444"))
    ax.add_patch(plt.Rectangle((4.5, 0), 1, 10, color="#444"))
    for i in range(0, 10, 1):
        ax.plot([i, i + 0.5], [5.0, 5.0], color="white", linewidth=1, linestyle="--")
        ax.plot([5.0, 5.0], [i, i + 0.5], color="white", linewidth=1, linestyle="--")
    light_ns_color = "green" if action == 0 else "red"
    light_ew_color = "green" if action == 1 else "red"
    ax.add_patch(plt.Circle((5, 9), 0.4, color=light_ns_color))
    ax.text(5, 8.3, f"{countdown}s", ha='center', va='center', fontsize=10, color='black', bbox=dict(facecolor='white', boxstyle='round,pad=0.2'))
    ax.add_patch(plt.Circle((9, 5), 0.4, color=light_ew_color))
    ax.text(8.2, 5, f"{countdown}s", ha='center', va='center', fontsize=10, color='black', bbox=dict(facecolor='white', boxstyle='round,pad=0.2'))

    for i in range(5):
        if state["NS_array"][i]:
            ax.add_patch(plt.Rectangle((4.6, 9 - i), 0.8, 0.5, color="red", alpha=0.9))
        if state["EW_array"][i]:
            ax.add_patch(plt.Rectangle((i, 4.6), 0.5, 0.8, color="blue", alpha=0.9))

    if highlight_clear:
        ax.text(5, 5.2, "+1", ha='center', va='center', fontsize=14, color='lime', fontweight='bold')
    ax.text(5, 1, "S ↓", ha='center', va='center', fontsize=9, color='white')
    ax.text(5, 9.3, "N ↑", ha='center', va='center', fontsize=9, color='white')
    ax.text(1.2, 5, "W ←", ha='center', va='center', fontsize=9, color='white')
    ax.text(8.8, 5, "E →", ha='center', va='center', fontsize=9, color='white')
    fig.text(0.5, 1.0, f"Intersection {idx+1}", ha='center', fontsize=12, fontweight='bold', color='black')
    fig.text(0.5, -0.05, "🟥 Red = Stop     🟩 Green = Go     ⏱ = Countdown     +1 = Vehicle Passed", ha='center', fontsize=9, color='black', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4'))
    return fig

def demo_agents(q_tables, steps=30, speed=0.4):
    inflow_probs = np.linspace(0.3, 0.7, len(q_tables))
    envs = [TrafficIntersection(inflow_prob=prob) for prob in inflow_probs]
    states = [get_discrete_state(env.reset()) for env in envs]
    placeholders = [st.empty() for _ in q_tables]
    intersection_states = [{"action": 0, "timer": 3} for _ in q_tables]

    for t in range(steps):
        for i, (env, q_table) in enumerate(zip(envs, q_tables)):
            if intersection_states[i]["timer"] == 0:
                intersection_states[i]["action"] = 1 - intersection_states[i]["action"]
                intersection_states[i]["timer"] = 3
            action = intersection_states[i]["action"]
            intersection_states[i]["timer"] -= 1
            full_state, _, cleared = env.step(action)
            fig = draw_intersection(full_state, action, t, i, intersection_states[i]["timer"] + 1, highlight_clear=cleared)
            placeholders[i].pyplot(fig)
            plt.close(fig)
            states[i] = get_discrete_state(full_state)
        time.sleep(speed)

    st.subheader("🚗 Total Vehicles Cleared per Intersection")
    for i, env in enumerate(envs):
        st.markdown(f"**Intersection {i+1}:** {env.total_cleared} vehicles")

# ---------------- Streamlit UI ---------------- #
st.title("🚦 Multi-Intersection Traffic Light Optimization (Q-learning)")

mode = st.selectbox("Choose Mode", ["Train Agent", "Demo Agent"])
n_intersections = st.slider("Number of Intersections", 1, 4, 2)
speed = st.slider("Demo Speed (sec/frame)", 0.1, 1.0, 0.4)

q_file = f"q_tables_{n_intersections}.pkl"

if mode == "Train Agent":
    st.info("Training Q-learning agents...")
    q_tables, rewards_list = train_agents(n_intersections)
    st.success("✅ Training complete!")
    with open(q_file, "wb") as f:
        pickle.dump(q_tables, f)
    for i, rewards in enumerate(rewards_list):
        st.subheader(f"Intersection {i+1}")
        st.line_chart(rewards)

elif mode == "Demo Agent":
    if os.path.exists(q_file):
        with open(q_file, "rb") as f:
            q_tables = pickle.load(f)
        st.success("Loaded trained Q-tables.")
        demo_agents(q_tables, steps=30, speed=speed)
    else:
        st.error("No Q-tables found. Please train the agent(s) first.")
