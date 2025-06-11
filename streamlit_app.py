import streamlit as st
import numpy as np
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt

# Load trained Q-table
q_table = np.load("q_table.npy", allow_pickle=True).item()

# Define bins and ideal ranges
bins = [
    np.linspace(20, 35, 6),   # Temp
    np.linspace(5.5, 9.5, 5), # pH
    np.linspace(0, 10, 6)     # DO
]

ideal_ranges = {
    "Temperature (°C)": (26, 30),
    "pH": (6.5, 8.0),
    "Dissolved Oxygen (mg/L)": (5.0, 10.0)
}

actions = {
    0: "✅ Do nothing — conditions are acceptable.",
    1: "💨 Aerate the pond — low dissolved oxygen detected.",
    2: "🧪 Add buffer chemicals — pH level is off.",
    3: "♻️ Drain and refill pond — severe imbalance detected."
}

def discretize(values):
    return tuple(np.digitize(v, b) for v, b in zip(values, bins))

def check_status(value, range_):
    return "✅ OK" if range_[0] <= value <= range_[1] else "❌ Out of range"

def score_quality(temp, ph, do):
    score = 0
    score += 1 if ideal_ranges["Temperature (°C)"][0] <= temp <= ideal_ranges["Temperature (°C)"][1] else 0
    score += 1 if ideal_ranges["pH"][0] <= ph <= ideal_ranges["pH"][1] else 0
    score += 1 if ideal_ranges["Dissolved Oxygen (mg/L)"][0] <= do <= ideal_ranges["Dissolved Oxygen (mg/L)"][1] else 0
    return score

def simulate_day(temp, ph, do):
    return (
        np.clip(temp + np.random.uniform(-0.3, 0.3), 20, 35),
        np.clip(ph + np.random.uniform(-0.2, 0.2), 5.5, 9.5),
        np.clip(do - np.random.uniform(0.1, 0.3), 0, 10)
    )

def save_log(temp, ph, do, action_label):
    log_file = "pond_logs.csv"
    log_entry = {
        "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Temperature (°C)": temp,
        "pH": ph,
        "DO (mg/L)": do,
        "Recommended Action": action_label
    }
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([log_entry])
    df.to_csv(log_file, index=False)

def plot_comparison(temp, ph, do):
    labels = ["Temperature (°C)", "pH", "Dissolved Oxygen (mg/L)"]
    values = [temp, ph, do]
    lower_bounds = [ideal_ranges[l][0] for l in labels]
    upper_bounds = [ideal_ranges[l][1] for l in labels] 
    ideal_widths = [upper - lower for lower, upper in zip(lower_bounds, upper_bounds)]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(labels, values, color="skyblue", label="Current Reading")
    ax.barh(labels, ideal_widths, left=lower_bounds, height=0.5, alpha=0.2, color="green", label="Ideal Range")
    ax.set_xlim(0, max(upper_bounds) + 2)
    ax.set_title("Current vs Ideal Ranges")
    ax.legend()
    st.pyplot(fig) 

# --- Streamlit App ---
st.set_page_config(page_title="Catfish Pond Advisor", layout="centered")
st.title("🐟 Catfish Pond Water Advisor")

st.markdown("Input your water conditions to get a recommendation:")

temp = st.slider("🌡️ Temperature (°C)", 20.0, 35.0, 28.0, 0.1)
ph = st.slider("🧪 pH Level", 5.5, 9.5, 7.0, 0.1)
do = st.slider("💨 Dissolved Oxygen (mg/L)", 0.0, 10.0, 5.0, 0.1)

st.subheader("🧾 Water Status")
st.write(f"🌡️ Temperature: **{temp}°C** — {check_status(temp, ideal_ranges['Temperature (°C)'])}")
st.write(f"🧪 pH: **{ph}** — {check_status(ph, ideal_ranges['pH'])}")
st.write(f"💨 DO: **{do} mg/L** — {check_status(do, ideal_ranges['Dissolved Oxygen (mg/L)'])}")

plot_comparison(temp, ph, do)

# Main logic
if st.button("📌 Get Recommendation"):
    state_disc = discretize([temp, ph, do])

    if state_disc in q_table:
        action = np.argmax(q_table[state_disc])
        action_label = actions[action]
        st.success(f"🤖 Recommended Action:\n\n**{action_label}**")
        st.info(f"💡 Pond Health Score: **{score_quality(temp, ph, do)} / 3**")

        # Save recommendation
        save_log(temp, ph, do, action_label)
    else:
        st.warning("🤖 This condition is unknown. Try slightly changing values.")

# Simulate next day
if st.button("🔄 Simulate Next Day"):
    temp, ph, do = simulate_day(temp, ph, do)
    st.warning("🔁 Simulated new values. Please adjust sliders manually to match.") 

# View log
if st.checkbox("📂 Show Recommendation Log"):
    if os.path.exists("pond_logs.csv"):
        df = pd.read_csv("pond_logs.csv")
        st.dataframe(df)
    else: 
        st.info("No recommendations saved yet.")

st.markdown("---")
st.caption("Developed with Q-learning · ScholarGPT · Final Year Project") 
