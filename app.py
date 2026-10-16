
import random
import math
from collections import Counter

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Dice Prediction Challenge",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

SIDES = 6
TARGET_SUMS = (7, 12)

BG = "#101827"
PANEL = "#1B263B"
TEXT = "#F1F5F9"
MUTED = "#AAB7CF"
GREEN = "#34D399"
ORANGE = "#FB923C"
BLUE = "#60A5FA"
PURPLE = "#C084FC"

# =========================================================
# STYLING
# =========================================================

st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: #F1F5F9;
}
[data-testid="stSidebar"] {
    background-color: #1B263B;
}
h1, h2, h3, p, label {
    color: #F1F5F9;
}
div[data-testid="stMetric"] {
    background: #1B263B;
    border: 1px solid #334155;
    padding: 16px;
    border-radius: 12px;
}
.stButton > button {
    border-radius: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# PROBABILITY ENGINE
# =========================================================

def build_theoretical_distribution():
    ways = Counter(
        a + b
        for a in range(1, SIDES + 1)
        for b in range(1, SIDES + 1)
    )

    probabilities = {
        total: ways[total] / (SIDES ** 2)
        for total in range(2, 13)
    }

    return ways, probabilities


def simulate_rolls(number_of_rolls):
    return [
        random.randint(1, 6) + random.randint(1, 6)
        for _ in range(number_of_rolls)
    ]


def bayesian_predictions(observed, ways):
    counts = Counter(observed)
    n = len(observed)

    return {
        total: (counts[total] + ways[total]) / (n + 36)
        for total in range(2, 13)
    }


def wilson_interval(successes, trials, z=1.96):
    if trials == 0:
        return 0.0, 1.0

    p = successes / trials
    denominator = 1 + z**2 / trials

    center = (
        p + z**2 / (2 * trials)
    ) / denominator

    margin = (
        z * math.sqrt(
            p * (1 - p) / trials
            + z**2 / (4 * trials**2)
        )
        / denominator
    )

    return max(0, center - margin), min(1, center + margin)


# =========================================================
# DASHBOARD CHARTS
# =========================================================

def style_axis(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT)

    for spine in ax.spines.values():
        spine.set_color("#42516A")

    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(axis="y", alpha=0.18, color=MUTED)
    ax.set_axisbelow(True)


def make_dashboard(observed, simulated, theory, prediction):
    counts = Counter(observed)
    sim_counts = Counter(simulated)
    sums = list(range(2, 13))

    observed_rates = [
        counts[s] / len(observed) for s in sums
    ]
    theoretical_rates = [theory[s] for s in sums]
    predicted_rates = [prediction[s] for s in sums]
    simulation_rates = [
        sim_counts[s] / len(simulated) for s in sums
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(15, 10),
        layout="constrained"
    )
    fig.patch.set_facecolor(BG)

    for ax in axes.flat:
        style_axis(ax)

    # Chart A: probability comparison
    ax = axes[0, 0]
    x = list(range(len(sums)))
    width = 0.25

    ax.bar(
        [i - width for i in x],
        theoretical_rates,
        width,
        color=BLUE,
        label="Theoretical"
    )
    ax.bar(
        x, observed_rates,
        width,
        color=GREEN,
        label="Observed"
    )
    ax.bar(
        [i + width for i in x],
        predicted_rates,
        width,
        color=PURPLE,
        label="Smoothed prediction"
    )

    ax.set_xticks(x, sums)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("A. Probability Comparison")
    ax.set_xlabel("Dice sum")
    ax.set_ylabel("Probability")
    ax.legend(
        facecolor=PANEL,
        labelcolor=TEXT,
        fontsize=8
    )

    # Chart B: simulated results
    ax = axes[0, 1]
    ax.bar(
        sums,
        simulation_rates,
        color=[
            ORANGE if s == 12 else
            GREEN if s == 7 else BLUE
            for s in sums
        ],
        label="Simulated"
    )
    ax.plot(
        sums,
        theoretical_rates,
        color=TEXT,
        linestyle="--",
        marker="o",
        label="Theoretical"
    )

    ax.set_xticks(sums)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(
        f"B. Monte Carlo Simulation ({len(simulated):,} rolls)"
    )
    ax.set_xlabel("Dice sum")
    ax.set_ylabel("Probability")
    ax.legend(
        facecolor=PANEL,
        labelcolor=TEXT,
        fontsize=8
    )

    # Chart C: running frequencies
    ax = axes[1, 0]
    running_counts = Counter()
    running_seven = []
    running_twelve = []

    for i, total in enumerate(simulated, start=1):
        running_counts[total] += 1
        running_seven.append(running_counts[7] / i)
        running_twelve.append(running_counts[12] / i)

    ax.plot(
        running_seven,
        color=GREEN,
        label="Sum 7"
    )
    ax.plot(
        running_twelve,
        color=ORANGE,
        label="Sum 12"
    )
    ax.axhline(
        theory[7],
        color=GREEN,
        linestyle="--",
        alpha=0.8
    )
    ax.axhline(
        theory[12],
        color=ORANGE,
        linestyle="--",
        alpha=0.8
    )

    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("C. Running Frequency")
    ax.set_xlabel("Simulated roll")
    ax.set_ylabel("Observed frequency")
    ax.legend(
        facecolor=PANEL,
        labelcolor=TEXT,
        fontsize=8
    )

    # Chart D: all 36 dice combinations
    ax = axes[1, 1]
    matrix = [
        [a + b for b in range(1, 7)]
        for a in range(1, 7)
    ]

    heatmap = ax.imshow(
        matrix,
        cmap="viridis",
        origin="lower",
        vmin=2,
        vmax=12
    )

    for row in range(6):
        for col in range(6):
            ax.text(
                col, row, str(matrix[row][col]),
                ha="center",
                va="center",
                color="white",
                fontweight="bold"
            )

    ax.set_xticks(range(6), range(1, 7))
    ax.set_yticks(range(6), range(1, 7))
    ax.set_xlabel("Die 2")
    ax.set_ylabel("Die 1")
    ax.set_title("D. Dice Combination Heatmap")

    colorbar = fig.colorbar(
        heatmap, ax=ax, fraction=0.046, pad=0.04
    )
    colorbar.ax.tick_params(colors=TEXT)
    colorbar.set_label("Dice sum", color=TEXT)

    fig.suptitle(
        "DICE ANALYTICS | PROBABILITY INTELLIGENCE",
        color=TEXT,
        fontsize=17,
        fontweight="bold"
    )

    return fig


# =========================================================
# MAIN STREAMLIT INTERFACE
# =========================================================

st.title("🎲 Dice Prediction Challenge")
st.markdown(
    "### Probability Analytics & Monte Carlo Simulation"
)
st.caption(
    "Explore dice outcomes, compare real rolls with theory, "
    "and investigate statistical uncertainty."
)

ways, theory = build_theoretical_distribution()

with st.sidebar:
    st.header("⚙️ Experiment Settings")

    number_of_rolls = st.slider(
        "Manual dice rolls",
        min_value=5,
        max_value=100,
        value=10,
        step=5
    )

    simulation_count = st.select_slider(
        "Simulation size",
        options=[1000, 5000, 10000, 25000, 50000],
        value=10000,
        format_func=lambda n: f"{n:,} rolls"
    )

    st.info(
        "Each new experiment starts fresh. "
        "Your roll data is held in this browser session."
    )

    if st.button("🔄 Start a New Experiment", use_container_width=True):
        st.session_state.pop("observed", None)
        st.session_state.pop("experiment_id", None)
        st.rerun()


# Initialize a fresh experiment
if "observed" not in st.session_state:
    st.session_state.observed = []
    st.session_state.experiment_id = 1

st.subheader("1. Enter Your Dice Results")
st.write(
    "For each roll, enter the result of die 1 and die 2. "
    "Both values must be between 1 and 6."
)

with st.form("dice_input_form"):
    dice_values = []

    for i in range(number_of_rolls):
        col1, col2 = st.columns(2)

        with col1:
            die_a = st.selectbox(
                f"Roll {i + 1} — Die 1",
                options=[1, 2, 3, 4, 5, 6],
                index=0,
                key=f"die_a_{i}"
            )

        with col2:
            die_b = st.selectbox(
                f"Roll {i + 1} — Die 2",
                options=[1, 2, 3, 4, 5, 6],
                index=0,
                key=f"die_b_{i}"
            )

        dice_values.append((die_a, die_b))

    submitted = st.form_submit_button(
        "🚀 Analyze Dice Rolls",
        use_container_width=True
    )

if submitted:
    st.session_state.observed = [
        a + b for a, b in dice_values
    ]
    st.session_state.experiment_id += 1

observed = st.session_state.observed

if not observed:
    st.info(
        "Choose your dice values above and click "
        "'Analyze Dice Rolls' to generate the dashboard."
    )
    st.stop()

# Fresh simulation for the current submitted experiment.
# Streamlit caches the completed output in session state,
# so unrelated widget interactions don't rerun the simulation.
if (
    "simulated" not in st.session_state
    or st.session_state.get("simulation_experiment")
    != st.session_state.experiment_id
):
    st.session_state.simulated = simulate_rolls(
        simulation_count
    )
    st.session_state.simulation_experiment = (
        st.session_state.experiment_id
    )

simulated = st.session_state.simulated
prediction = bayesian_predictions(observed, ways)
observed_counts = Counter(observed)

# =========================================================
# RESULTS
# =========================================================

st.divider()
st.subheader("2. 📊 Your Prediction Results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Sum 7 — Observed",
        f"{observed_counts[7]} times",
        f"{observed_counts[7] / len(observed):.1%} of rolls"
    )

with col2:
    st.metric(
        "Sum 12 — Observed",
        f"{observed_counts[12]} times",
        f"{observed_counts[12] / len(observed):.1%} of rolls"
    )

with col3:
    st.metric(
        "Sum 7 — Theoretical",
        f"{theory[7]:.2%}"
    )

with col4:
    st.metric(
        "Sum 12 — Theoretical",
        f"{theory[12]:.2%}"
    )

st.subheader("3. 🤖 Prediction Engine")

predicted_sum = max(prediction, key=prediction.get)

st.success(
    f"Most likely next sum: {predicted_sum} "
    f"— smoothed model probability: "
    f"{prediction[predicted_sum]:.2%}"
)

st.caption(
    "This model combines observed frequencies with the "
    "known theoretical distribution. For fair dice, the "
    "true probabilities do not change after previous rolls."
)

st.subheader("4. 📈 Statistical Confidence")

confidence_rows = []

for target in TARGET_SUMS:
    count = observed_counts[target]
    low, high = wilson_interval(count, len(observed))

    confidence_rows.append({
        "Target sum": target,
        "Observed count": count,
        "Experimental probability": (
            f"{count / len(observed):.2%}"
        ),
        "Theoretical probability": f"{theory[target]:.2%}",
        "95% confidence interval": (
            f"{low:.2%} – {high:.2%}"
        ),
        "Smoothed estimate": f"{prediction[target]:.2%}"
    })

st.dataframe(
    pd.DataFrame(confidence_rows),
    use_container_width=True,
    hide_index=True
)

st.subheader("5. 🎨 Interactive Analytics Dashboard")

figure = make_dashboard(
    observed, simulated, theory, prediction
)

# Display in the browser; never save a PNG or image file.
st.pyplot(figure, use_container_width=True)
plt.close(figure)

st.divider()
st.caption(
    "Dice Prediction Challenge • Educational statistics project • "
    "No dashboard images are saved."
)
