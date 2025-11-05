import streamlit as st
import pandas as pd
import random

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file):
    df = pd.read_csv(file)
    program_ratings = {}
    for row in df.itertuples(index=False):
        program = row[0]  # First column should be the program name
        ratings = list(row[1:])  # Remaining columns are ratings
        program_ratings[program] = ratings
    return program_ratings

# Genetic algorithm functions
def fitness_function(schedule, ratings):
    total = 0.0
    for time_slot, program in enumerate(schedule):
        if program in ratings:
            vals = ratings[program]
            total += vals[time_slot] if time_slot < len(vals) else 0.0
    return total

def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]
    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)
    return all_schedules

def finding_best_schedule(all_schedules, ratings=None):
    best_schedule = []
    max_ratings = float('-inf')
    for schedule in all_schedules:
        total_ratings = fitness_function(schedule, ratings or {})
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule
    return best_schedule

def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

def mutate(schedule, all_programs):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

def genetic_algorithm(initial_schedule, generations, population_size, crossover_rate, mutation_rate, elitism_size, all_programs, ratings):
    population = [initial_schedule]
    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)
    for _generation in range(generations):
        new_population = []
        population.sort(key=lambda s: fitness_function(s, ratings), reverse=True)
        new_population.extend(population[:elitism_size])
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            if random.random() < mutation_rate:
                child1 = mutate(child1, all_programs)
            if random.random() < mutation_rate:
                child2 = mutate(child2, all_programs)
            new_population.extend([child1, child2])
        population = new_population
    return population[0]

# Streamlit application starts here
st.title("Optimal Schedule Generator with 3 Trials")
st.write("Upload a CSV with program ratings and run three GA trials with different CO_R / MUT_R settings.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file with program ratings", type="csv")

# Global placeholders
ratings = {}
all_programs = []
all_time_slots = list(range(6, 24))  # 6:00 to 23:00
initial_best_schedule = []

# 1) Load data if provided
if uploaded_file is not None:
    ratings = read_csv_to_dict(uploaded_file)
    all_programs = list(ratings.keys())
    # If there are more time slots than ratings length, we pad later if needed
    # Build an initial best schedule using brute-force (permutation search)
    all_possible_schedules = initialize_pop(all_programs, all_time_slots)
    initial_best_schedule = finding_best_schedule(all_possible_schedules, ratings)
else:
    st.info("Awaiting CSV upload to run the optimizer.")
    st.stop()

# 2) GA parameter interface for three trials
st.sidebar.header("Trial 1 Parameters")
tr1_co = st.sidebar.number_input("Trial 1 CO_R", min_value=0.0, max_value=0.95, value=0.8, step=0.01, key="tr1_co")
tr1_mut = st.sidebar.number_input("Trial 1 MUT_R", min_value=0.01, max_value=0.05, value=0.02, step=0.01, key="tr1_mut")

st.sidebar.header("Trial 2 Parameters")
tr2_co = st.sidebar.number_input("Trial 2 CO_R", min_value=0.0, max_value=0.95, value=0.75, step=0.01, key="tr2_co")
tr2_mut = st.sidebar.number_input("Trial 2 MUT_R", min_value=0.01, max_value=0.05, value=0.03, step=0.01, key="tr2_mut")

st.sidebar.header("Trial 3 Parameters")
tr3_co = st.sidebar.number_input("Trial 3 CO_R", min_value=0.0, max_value=0.95, value=0.85, step=0.01, key="tr3_co")
tr3_mut = st.sidebar.number_input("Trial 3 MUT_R", min_value=0.01, max_value=0.05, value=0.04, step=0.01, key="tr3_mut")

# Run trials button
if st.button("Run Trials"):
    with st.spinner("Running trials..."):
        results = []
        # Ensure we have the initial best schedule and program list
        if not initial_best_schedule:
            all_possible_schedules = initialize_pop(all_programs, all_time_slots)
            initial_best_schedule = finding_best_schedule(all_possible_schedules, ratings)

        # Trial 1
        rem_t_slots = len(all_time_slots) - len(initial_best_schedule)
        gene1 = genetic_algorithm(
            initial_schedule=initial_best_schedule,
            generations=100,
            population_size=50,
            crossover_rate=tr1_co,
            mutation_rate=tr1_mut,
            elitism_size=2,
            all_programs=all_programs,
            ratings=ratings
        )
        final_schedule1 = initial_best_schedule + gene1[:rem_t_slots]
        per_slot_ratings1 = []
        for i, prog in enumerate(final_schedule1):
            if prog == "NOOP":
                per_slot_ratings1.append(0.0)
            else:
                vals = ratings.get(prog, [])
                per_slot_ratings1.append(vals[i] if i < len(vals) else 0.0)
        total1 = sum(per_slot_ratings1)
        results.append({
            "trial": 1,
            "CO_R": tr1_co,
            "MUT_R": tr1_mut,
            "schedule": final_schedule1,
            "per_slot_ratings": per_slot_ratings1,
            "total": total1
        })

        # Trial 2
        gene2 = genetic_algorithm(
            initial_schedule=initial_best_schedule,
            generations=100,
            population_size=50,
            crossover_rate=tr2_co,
            mutation_rate=tr2_mut,
            elitism_size=2,
            all_programs=all_programs,
            ratings=ratings
        )
        final_schedule2 = initial_best_schedule + gene2[:rem_t_slots]
        per_slot_ratings2 = []
        for i, prog in enumerate(final_schedule2):
            if prog == "NOOP":
                per_slot_ratings2.append(0.0)
            else:
                vals = ratings.get(prog, [])
                per_slot_ratings2.append(vals[i] if i < len(vals) else 0.0)
        total2 = sum(per_slot_ratings2)
        results.append({
            "trial": 2,
            "CO_R": tr2_co,
            "MUT_R": tr2_mut,
            "schedule": final_schedule2,
            "per_slot_ratings": per_slot_ratings2,
            "total": total2
        })

        # Trial 3
        gene3 = genetic_algorithm(
            initial_schedule=initial_best_schedule,
            generations=100,
            population_size=50,
            crossover_rate=tr3_co,
            mutation_rate=tr3_mut,
            elitism_size=2,
            all_programs=all_programs,
            ratings=ratings
        )
        final_schedule3 = initial_best_schedule + gene3[:rem_t_slots]
        per_slot_ratings3 = []
        for i, prog in enumerate(final_schedule3):
            if prog == "NOOP":
                per_slot_ratings3.append(0.0)
            else:
                vals = ratings.get(prog, [])
                per_slot_ratings3.append(vals[i] if i < len(vals) else 0.0)
        total3 = sum(per_slot_ratings3)
        results.append({
            "trial": 3,
            "CO_R": tr3_co,
            "MUT_R": tr3_mut,
            "schedule": final_schedule3,
            "per_slot_ratings": per_slot_ratings3,
            "total": total3
        })

    # Display results
    st.subheader("Trial Results")
    for r in results:
        trial_num = r["trial"]
        st.write(f"Trial {trial_num}: CO_R={r['CO_R']:.3f}, MUT_R={r['MUT_R']:.3f}")
        # Schedule table
        rows = []
        for i, prog in enumerate(r["schedule"]):
            t_slot = all_time_slots[i]
            rows.append({"Time Slot": f"{t_slot:02d}:00", "Program": prog})
        df = pd.DataFrame(rows)
        st.dataframe(df)
        st.write(f"Total Rating: {r['total']:.2f}")
        # Optional per-slot ratings chart
        chart_series = pd.Series(r["per_slot_ratings"], index=[f"{t:02d}:00" for t in all_time_slots[:len(r["schedule"])]])
        if not chart_series.empty:
            st.line_chart(chart_series)  # Only line chart remains
