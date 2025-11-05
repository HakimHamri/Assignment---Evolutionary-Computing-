# app.py
import streamlit as st
import pandas as pd
import random
from io import StringIO

st.set_page_config(page_title="TV Schedule Genetic Algorithm", layout="wide")
st.title("TV Schedule Optimization (Genetic Algorithm)")

st.markdown(
    "Upload a CSV where the first column is the program name and the remaining columns are ratings "
    "for each time slot (in order). Example header: Program,6,7,8,... or Program,slot1,slot2,..."
)

uploaded_file = st.file_uploader("Upload program_ratings.csv", type=["csv"])

# GA hyperparameters defaults/ranges
CO_DEFAULT = 0.8
CO_MIN, CO_MAX = 0.0, 0.95
MUT_MIN, MUT_MAX = 0.01, 0.05
MUT_DEFAULT = 0.02  # chosen to be inside requested allowed range

GEN_DEFAULT = 100
POP_DEFAULT = 50
ELITISM_DEFAULT = 2

# Sidebar for global GA settings
with st.sidebar:
    st.header("GA Global Settings")
    generations = st.number_input("Generations", min_value=1, value=GEN_DEFAULT, step=1)
    population_size = st.number_input("Population Size", min_value=2, value=POP_DEFAULT, step=1)
    elitism_size = st.number_input("Elitism Size", min_value=0, value=ELITISM_DEFAULT, step=1)
    seed = st.number_input("Random seed (0 = none)", min_value=0, value=0, step=1)
    if seed != 0:
        random.seed(seed)

def parse_ratings_csv(file_like):
    # Returns (ratings_dict, programs_list, num_time_slots)
    df = pd.read_csv(file_like)
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least two columns: program and at least one time-slot rating column.")
    program_col = df.columns[0]
    program_names = df[program_col].astype(str).tolist()
    rating_cols = df.columns[1:]
    # convert ratings to floats
    ratings = {}
    for idx, prog in enumerate(program_names):
        row = df.iloc[idx, 1:].tolist()
        row_floats = [float(x) for x in row]
        ratings[prog] = row_floats
    num_slots = len(rating_cols)
    # Determine time slot labels: if header values are numeric (like 6,7,8...), use them; otherwise create offsets
    try:
        time_labels = [int(x) for x in rating_cols]
        # convert to displayable strings like "06:00"
        time_labels = [f"{h:02d}:00" for h in time_labels]
    except Exception:
        # fallback: use 6:00.. or generic indices 1..n
        time_labels = [f"Slot {i+1}" for i in range(num_slots)]
    return ratings, program_names, time_labels

# GA functions
def greedy_initial_schedule(ratings, programs, num_slots):
    # For each slot pick program with highest rating (allows repeats)
    schedule = []
    for t in range(num_slots):
        best_prog = None
        best_val = float("-inf")
        for prog in programs:
            val = ratings[prog][t]
            if val > best_val:
                best_val = val
                best_prog = prog
        schedule.append(best_prog)
    return schedule

def fitness_function(schedule, ratings):
    total = 0.0
    for t, prog in enumerate(schedule):
        total += ratings[prog][t]
    return total

def crossover(parent1, parent2):
    if len(parent1) < 2:
        return parent1[:], parent2[:]
    cp = random.randint(1, len(parent1)-1)
    child1 = parent1[:cp] + parent2[cp:]
    child2 = parent2[:cp] + parent1[cp:]
    return child1, child2

def mutate(schedule, programs):
    idx = random.randrange(len(schedule))
    schedule[idx] = random.choice(programs)
    return schedule

def genetic_algorithm(ratings, programs, num_slots,
                      generations=100, population_size=50,
                      crossover_rate=0.8, mutation_rate=0.02, elitism_size=2):
    # initialize population: start with greedy best, rest random
    population = []
    seed_individual = greedy_initial_schedule(ratings, programs, num_slots)
    population.append(seed_individual)
    for _ in range(population_size - 1):
        individual = [random.choice(programs) for _ in range(num_slots)]
        population.append(individual)

    for gen in range(generations):
        # evaluate and sort by fitness descending
        population.sort(key=lambda s: fitness_function(s, ratings), reverse=True)
        new_pop = population[:elitism_size]  # preserve elites
        # produce children
        while len(new_pop) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]
            if random.random() < mutation_rate:
                child1 = mutate(child1, programs)
            if random.random() < mutation_rate:
                child2 = mutate(child2, programs)
            new_pop.append(child1)
            if len(new_pop) < population_size:
                new_pop.append(child2)
        population = new_pop
    # final best
    population.sort(key=lambda s: fitness_function(s, ratings), reverse=True)
    return population[0]

if uploaded_file is not None:
    try:
        ratings, programs, time_labels = parse_ratings_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error parsing CSV: {e}")
        st.stop()

    num_slots = len(time_labels)
    st.success(f"Loaded {len(programs)} programs and {num_slots} time slots.")

    st.markdown("## Trial Parameters (set different parameters for each of the 3 trials)")

    # Input widgets for 3 trials
    trial_params = []
    cols = st.columns(3)
    for i in range(3):
        with cols[i]:
            st.subheader(f"Trial {i+1}")
            co = st.slider(f"CO_R (Crossover Rate) - Trial {i+1}", min_value=CO_MIN, max_value=CO_MAX, value=CO_DEFAULT, step=0.01, key=f"co_{i}")
            mut = st.slider(f"MUT_R (Mutation Rate) - Trial {i+1}", min_value=MUT_MIN, max_value=MUT_MAX, value=MUT_DEFAULT, step=0.01, key=f"mut_{i}")
            trial_params.append((co, mut))

    run_button = st.button("Run 3 Trials")

    if run_button:
        st.markdown("## Results")
        results = []
        for i, (co, mut) in enumerate(trial_params, start=1):
            st.markdown(f"### Trial {i} — CO_R={co:.2f}, MUT_R={mut:.2f}")
            # run GA
            best = genetic_algorithm(
                ratings=ratings,
                programs=programs,
                num_slots=num_slots,
                generations=generations,
                population_size=population_size,
                crossover_rate=co,
                mutation_rate=mut,
                elitism_size=elitism_size
            )
            total = fitness_function(best, ratings)
            # Build table
            df = pd.DataFrame({
                "Time Slot": time_labels,
                "Program": best,
                "Rating": [ratings[p][t] for t, p in enumerate(best)]
            })
            st.write(f"Total Ratings: {total:.2f}")
            st.table(df)
else:
    st.info("Please upload the program_ratings.csv to continue.")
