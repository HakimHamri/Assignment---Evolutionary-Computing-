import streamlit as st
import pandas as pd
import random
from typing import Dict, List

# Configuration
ALL_TIME_SLOTS = list(range(6, 24))  # 6:00 .. 23:00 inclusive (18 slots)

# ---------- CSV reading ----------
def read_csv_to_dict(file) -> Dict[str, List[float]]:
    """
    Reads a CSV where the first column is program name and subsequent columns are ratings.
    If header of rating columns contains hour numbers (6..23) those are used to align; otherwise
    ratings are assumed to map left-to-right starting at 6:00. Missing ratings are treated as 0.0.
    Returns mapping: program -> list of length len(ALL_TIME_SLOTS).
    """
    df = pd.read_csv(file)
    if df.shape[1] < 2:
        raise ValueError("CSV must have program name column and at least one rating column.")

    prog_col = df.columns[0]
    rating_cols = list(df.columns[1:])

    # Try to interpret rating columns as hour labels (ints) e.g. "6", "06:00", "6:00", "7"
    col_hour_map = {}
    for col in rating_cols:
        try:
            # try bare int
            h = int(col)
            col_hour_map[h] = col
        except Exception:
            # try parsing like "06:00" or "6:00" -> take hour part
            try:
                hour_part = str(col).split(":")[0]
                h = int(hour_part)
                col_hour_map[h] = col
            except Exception:
                # not parseable; ignore for now
                pass

    program_ratings = {}
    n_slots = len(ALL_TIME_SLOTS)
    for _, row in df.iterrows():
        program = row[prog_col]
        # initialize full-length rating list
        ratings = [0.0] * n_slots
        if col_hour_map:
            # place available columns into proper slots
            for h, colname in col_hour_map.items():
                if h in ALL_TIME_SLOTS:
                    idx = ALL_TIME_SLOTS.index(h)
                    try:
                        val = float(row[colname])
                    except Exception:
                        val = 0.0
                    ratings[idx] = val
        else:
            # no hour-labeled columns: assume left-to-right mapping starting at 6:00
            for i, colname in enumerate(rating_cols):
                if i >= n_slots:
                    break
                try:
                    val = float(row[colname])
                except Exception:
                    val = 0.0
                ratings[i] = val
        program_ratings[str(program)] = ratings
    return program_ratings

# ---------- GA helpers ----------
def fitness_function(schedule: List[str], ratings: Dict[str, List[float]]) -> float:
    total = 0.0
    n = len(schedule)
    for i, prog in enumerate(schedule):
        vals = ratings.get(prog)
        if vals and i < len(vals):
            total += vals[i]
        else:
            # missing program or missing slot -> 0
            total += 0.0
    return total

def greedy_initial_schedule(ratings: Dict[str, List[float]]) -> List[str]:
    """
    For each slot pick the program with the highest rating for that slot (break ties randomly).
    If no programs exist, fill with "NOOP".
    """
    n_slots = len(ALL_TIME_SLOTS)
    programs = list(ratings.keys())
    if not programs:
        return ["NOOP"] * n_slots

    schedule = []
    for slot_idx in range(n_slots):
        best_val = float('-inf')
        best_progs = []
        for p in programs:
            vals = ratings.get(p, [])
            val = vals[slot_idx] if slot_idx < len(vals) else 0.0
            if val > best_val:
                best_val = val
                best_progs = [p]
            elif val == best_val:
                best_progs.append(p)
        schedule.append(random.choice(best_progs) if best_progs else random.choice(programs))
    return schedule

def initialize_population(seed_schedule: List[str], population_size: int, all_programs: List[str]) -> List[List[str]]:
    n = len(seed_schedule)
    population = []
    # include seed
    population.append(seed_schedule.copy())
    # add slight variations of seed
    for _ in range(max(0, population_size // 5 - 1)):
        s = seed_schedule.copy()
        # perform a few random swaps
        for _ in range(3):
            i, j = random.randrange(n), random.randrange(n)
            s[i], s[j] = s[j], s[i]
        population.append(s)
    # fill the rest with fully random schedules (allow repetitions)
    while len(population) < population_size:
        population.append([random.choice(all_programs) for _ in range(n)])
    return population

def crossover(parent1: List[str], parent2: List[str]) -> (List[str], List[str]):
    n = len(parent1)
    if n < 2:
        return parent1.copy(), parent2.copy()
    cp = random.randint(1, n - 1)
    child1 = parent1[:cp] + parent2[cp:]
    child2 = parent2[:cp] + parent1[cp:]
    return child1, child2

def mutate(schedule: List[str], all_programs: List[str], mutation_strength: int = 1) -> List[str]:
    n = len(schedule)
    s = schedule.copy()
    for _ in range(mutation_strength):
        i = random.randrange(n)
        s[i] = random.choice(all_programs)
    return s

def tournament_selection(population: List[List[str]], ratings: Dict[str, List[float]], k: int = 3) -> List[str]:
    contenders = random.sample(population, k)
    contenders.sort(key=lambda s: fitness_function(s, ratings), reverse=True)
    return contenders[0]

def genetic_algorithm(seed_schedule: List[str],
                      generations: int,
                      population_size: int,
                      crossover_rate: float,
                      mutation_rate: float,
                      elitism_size: int,
                      all_programs: List[str],
                      ratings: Dict[str, List[float]]) -> List[str]:
    # Initialize population
    population = initialize_population(seed_schedule, population_size, all_programs)

    for _gen in range(generations):
        # sort population by fitness (desc)
        population.sort(key=lambda s: fitness_function(s, ratings), reverse=True)
        new_pop = []
        # elitism
        new_pop.extend([p.copy() for p in population[:elitism_size]])

        # generate children
        while len(new_pop) < population_size:
            parent1 = tournament_selection(population, ratings, k=3)
            parent2 = tournament_selection(population, ratings, k=3)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            if random.random() < mutation_rate:
                child1 = mutate(child1, all_programs)
            if random.random() < mutation_rate:
                child2 = mutate(child2, all_programs)
            new_pop.append(child1)
            if len(new_pop) < population_size:
                new_pop.append(child2)
        population = new_pop

    # return best individual
    population.sort(key=lambda s: fitness_function(s, ratings), reverse=True)
    return population[0]

# ---------- Streamlit UI ----------
st.title("Optimal Schedule Generator with 3 Trials")
st.write("Upload a CSV with program ratings (first column program name, remaining columns ratings).")
uploaded_file = st.file_uploader("Upload CSV file with program ratings", type="csv")

if uploaded_file is None:
    st.info("Awaiting CSV upload to run the optimizer.")
    st.stop()

# Read CSV
try:
    ratings = read_csv_to_dict(uploaded_file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

all_programs = list(ratings.keys())
if not all_programs:
    st.error("No programs found in uploaded CSV.")
    st.stop()

# Sidebar GA params for three trials
st.sidebar.header("Trial 1 Parameters")
tr1_co = st.sidebar.slider("Trial 1 CO_R", 0.0, 1.0, 0.8, 0.01, key="tr1_co")
tr1_mut = st.sidebar.slider("Trial 1 MUT_R", 0.0, 1.0, 0.02, 0.01, key="tr1_mut")

st.sidebar.header("Trial 2 Parameters")
tr2_co = st.sidebar.slider("Trial 2 CO_R", 0.0, 1.0, 0.75, 0.01, key="tr2_co")
tr2_mut = st.sidebar.slider("Trial 2 MUT_R", 0.0, 1.0, 0.03, 0.01, key="tr2_mut")

st.sidebar.header("Trial 3 Parameters")
tr3_co = st.sidebar.slider("Trial 3 CO_R", 0.0, 1.0, 0.85, 0.01, key="tr3_co")
tr3_mut = st.sidebar.slider("Trial 3 MUT_R", 0.0, 1.0, 0.04, 0.01, key="tr3_mut")

generations = st.sidebar.number_input("Generations", min_value=10, max_value=5000, value=200, step=10)
population_size = st.sidebar.number_input("Population Size", min_value=10, max_value=1000, value=100, step=10)
elitism = st.sidebar.number_input("Elitism Size", min_value=0, max_value=10, value=2)

# Seed schedule (greedy)
seed_schedule = greedy_initial_schedule(ratings)

if st.button("Run Trials"):
    with st.spinner("Running trials..."):
        results = []
        # Trial 1
        best1 = genetic_algorithm(
            seed_schedule=seed_schedule,
            generations=int(generations),
            population_size=int(population_size),
            crossover_rate=float(tr1_co),
            mutation_rate=float(tr1_mut),
            elitism_size=int(elitism),
            all_programs=all_programs,
            ratings=ratings
        )
        total1 = fitness_function(best1, ratings)
        per_slot_ratings1 = [ratings.get(p, [0.0]*len(ALL_TIME_SLOTS))[i] for i, p in enumerate(best1)]
        results.append(("Trial 1", tr1_co, tr1_mut, best1, per_slot_ratings1, total1))

        # Trial 2
        best2 = genetic_algorithm(
            seed_schedule=seed_schedule,
            generations=int(generations),
            population_size=int(population_size),
            crossover_rate=float(tr2_co),
            mutation_rate=float(tr2_mut),
            elitism_size=int(elitism),
            all_programs=all_programs,
            ratings=ratings
        )
        total2 = fitness_function(best2, ratings)
        per_slot_ratings2 = [ratings.get(p, [0.0]*len(ALL_TIME_SLOTS))[i] for i, p in enumerate(best2)]
        results.append(("Trial 2", tr2_co, tr2_mut, best2, per_slot_ratings2, total2))

        # Trial 3
        best3 = genetic_algorithm(
            seed_schedule=seed_schedule,
            generations=int(generations),
            population_size=int(population_size),
            crossover_rate=float(tr3_co),
            mutation_rate=float(tr3_mut),
            elitism_size=int(elitism),
            all_programs=all_programs,
            ratings=ratings
        )
        total3 = fitness_function(best3, ratings)
        per_slot_ratings3 = [ratings.get(p, [0.0]*len(ALL_TIME_SLOTS))[i] for i, p in enumerate(best3)]
        results.append(("Trial 3", tr3_co, tr3_mut, best3, per_slot_ratings3, total3))

    # Display results
    for name, co, mut, schedule, per_slot_ratings, total in results:
        st.subheader(f"{name} (CO_R={co:.2f}, MUT_R={mut:.2f})")
        rows = []
        for i, prog in enumerate(schedule):
            slot = ALL_TIME_SLOTS[i]
            rows.append({"Time Slot": f"{slot:02d}:00", "Program": prog, "Rating": per_slot_ratings[i]})
        df = pd.DataFrame(rows)
        st.dataframe(df)
        st.write(f"Total Rating: {total:.2f}")
        # line chart
        chart_idx = [f"{t:02d}:00" for t in ALL_TIME_SLOTS]
        chart_series = pd.Series(per_slot_ratings, index=chart_idx)
        st.line_chart(chart_series)
