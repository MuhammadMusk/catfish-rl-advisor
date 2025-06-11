import numpy as np

# Load Q-table
q_table = np.load("q_table.npy", allow_pickle=True).item()

# Define bins to match training
bins = [
    np.linspace(20, 35, 6),   # Temp
    np.linspace(5.5, 9.5, 5), # pH
    np.linspace(0, 10, 6)     # DO
]

# Discretizer function
def discretize(values):
    return tuple(np.digitize(v, b) for v, b in zip(values, bins))

# Interpret action
actions = {
    0: "Do nothing — conditions are acceptable",
    1: "Aerate the pond — low dissolved oxygen detected",
    2: "Add chemical buffer — pH is outside safe range",
    3: "Drain and partially refill pond — severe imbalance"
}

# Main chatbot loop
def run_chatbot():
    print("\n🐟 Welcome to Catfish Pond Water Advisor!\n")

    try:
        temp = float(input("🌡️  Enter water temperature (°C): "))
        ph = float(input("🧪 Enter water pH level: "))
        do = float(input("💨 Enter dissolved oxygen level (mg/L): "))
    except ValueError:
        print("⚠️  Invalid input. Please enter numeric values.")
        return

    state = [temp, ph, do]
    state_disc = discretize(state)

    if state_disc not in q_table:
        print("\n🤖 Sorry, I haven't learned about this condition yet. Try adjusting slightly.")
        return

    best_action = np.argmax(q_table[state_disc])
    print("\n🤖 Based on your input:")
    print(f"- Temperature: {temp}°C")
    print(f"- pH Level: {ph}")
    print(f"- Dissolved Oxygen: {do} mg/L")

    print(f"\n✅ Recommended Action: {actions[best_action]}")

if __name__ == "__main__":
    run_chatbot() 
    