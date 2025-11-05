import streamlit as st
import pandas as pd
import random

# ======================== CSV READER ==============================
def read_csv_to_dict(file):
    df = pd.read_csv(file)
    program_ratings = {}
    for row in df.itertuples(index=False):
        program = row[0]  # First column: program name
        ratings = list(row[1:])  # Remaining columns: ratings
        program_ratings[program] = ratings
    return program_ratings

# ======================== GENETIC ALGORITHM ======================
def fitness_function(schedule, ratings):
    total = 0.0
    for time_slot, program in enumerate(schedule):
        if program in ratings:
            vals = ratings[program]
            total += vals[time_slot] if time_slot < len(vals) else 0.0
    return total

def crossover(schedule1, schedule2):
    point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:point] + schedule2[point:]
    child2 = schedule2[:point] + schedule1[point:]
    return child1, child2

def mutate(schedule, all_programs):
    point = random.randint(0, len(schedule) - 1)
    schedule[point] = random.choice(all_programs)
    return schedule

def initialize_population(all_programs, pop_size):
    population = []
    for _ in range(pop_size):
        schedule = random.sample(all_programs, len(all_programs))
        population.append(schedule)
    return population

def genetic_algorithm(generations, pop_size, co_rate, mut_rate, elitism, all_programs, ratings):
    population = initialize_population(all_programs, pop_size)
    for _ in range(generations):
        # Sort by fitness
        population.sort(key=lambda s: fitness_function(s, ratings), reverse=True)
        new_population = population[:elitism]
        while len(new_population) < pop_size:
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < co_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            if random.random() < mut_rate:
                child1 = mutate(child1, all_programs)
            if random.random() < mut_rate:
                child2 = mutate(child2, all_programs)
            new_population.extend([child1, child2])
        population = new_population[:pop_size]
    return max(population, key=lambda s: fitness_function(s, ratings))

# ======================== STREAMLIT UI ===========================
st.title("Optimal Schedule Generator (3 GA Trials)")
st.write("Upload a CSV of program ratings to optimize schedules using different crossover and mutation rates.")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

# Load data
ratings = read_csv_to_dict(uploaded_file)
all_programs = list(ratings.keys())
all_time_slots = list(range(6, 6 + len(all_programs)))

# ======================== SIDEBAR SLIDERS ========================
st.sidebar.header("Trial 1 Parameters")
tr1_co = st.sidebar.slider("Trial 1 Crossover Rate (CO_R)", 0.0, 1.0, 0.8, 0.01)
tr1_mut = st.sidebar.slider("Trial 1 Mutation Rate (MUT_R)", 0.0, 1.0, 0.02, 0.01)

st.sidebar.header("Trial 2 Parameters")
tr2_co = st.sidebar.slider("Trial 2 Crossover Rate (CO_R)", 0.0, 1.0, 0.75, 0.01)
tr2_mut = st.sidebar.slider("Trial 2 Mutation Rate (MUT_R)", 0.0, 1.0, 0.03, 0.01)

st.sidebar.header("Trial 3 Parameters")
tr3_co = st.sidebar.slider("Trial 3 Crossover Rate (CO_R)", 0.0, 1.0, 0.85, 0.01)
tr3_mut = st.sidebar.slider("Trial 3 Mutation Rate (MUT_R)", 0.0, 1.0, 0.04, 0.01)

# ======================== RUN TRIALS =============================
if st.button("Run 3 Trials"):
    with st.spinner("Running genetic algorithm trials..."):
        trials = [
            {"name": "Trial 1", "co": tr1_co, "mut": tr1_mut},
            {"name": "Trial 2", "co": tr2_co, "mut": tr2_mut},
            {"name": "Trial 3", "co": tr3_co, "mut": tr3_mut},
        ]

        results = []
        for t in trials:
            best_schedule = genetic_algorithm(
                generations=100,
                pop_size=50,
                co_rate=t["co"],
                mut_rate=t["mut"],
                elitism=2,
                all_programs=all_programs,
                ratings=ratings,
            )
            total = fitness_function(best_schedule, ratings)
            results.append({
                "Trial": t["name"],
                "Crossover": t["co"],
                "Mutation": t["mut"],
                "Total Rating": total,
                "Schedule": best_schedule
            })

    # ======================== RESULTS DISPLAY =====================
    st.subheader("Summary Comparison Table")
    summary_df = pd.DataFrame(
        [{"Trial": r["Trial"], "CO_R": r["Crossover"], "MUT_R": r["Mutation"], "Total Rating": r["Total Rating"]} for r in results]
    )
    st.dataframe(summary_df)

    # Show best schedules individually
    for r in results:
        st.markdown(f"### {r['Trial']}: CO_R = {r['Crossover']:.2f}, MUT_R = {r['Mutation']:.2f}")
        df = pd.DataFrame({
            "Time Slot": [f"{t:02d}:00" for t in all_time_slots],
            "Program": r["Schedule"]
        })
        st.dataframe(df)
        st.bar_chart(pd.Series(
            [ratings[p][i] if p in ratings and i < len(ratings[p]) else 0 for i, p in enumerate(r["Schedule"])],
            index=[f"{t:02d}:00" for t in all_time_slots]
        ))
        st.write(f"**Total Rating:** {r['Total Rating']:.2f}")
