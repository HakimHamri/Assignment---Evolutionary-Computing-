import streamlit as st
import pandas as pd
import csv
import random
import io

# ============================== CSV READER ==============================
def read_csv_to_dict(file):
    program_ratings = {}
    if file is None:
        return program_ratings
    file.seek(0)
    reader = csv.reader(io.StringIO(file.read().decode('utf-8')))
    header = next(reader)
    for row in reader:
        program = row[0]
        ratings = [float(x) for x in row[1:]]
        program_ratings[program] = ratings
    return program_ratings


# ========================== GENETIC ALGORITHM ==========================
def fitness_function(schedule, ratings):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot % len(ratings[program])]
    return total_rating


def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]
    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)
    return all_schedules


def finding_best_schedule(all_schedules, ratings):
    best_schedule = []
    max_ratings = 0
    for schedule in all_schedules:
        total_ratings = fitness_function(schedule, ratings)
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
    schedule[mutation_point] = random.choice(all_programs)
    return schedule


def genetic_algorithm(
    initial_schedule, ratings, all_programs, generations, population_size, crossover_rate, mutation_rate, elitism_size
):
    population = [initial_schedule]
    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
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

    return max(population, key=lambda s: fitness_function(s, ratings))


# ============================ STREAMLIT INTERFACE ============================
st.title("📺 TV Program Scheduling using Genetic Algorithm")

uploaded_file = st.file_uploader("Upload CSV File with Program Ratings", type=["csv"])

if uploaded_file:
    ratings = read_csv_to_dict(uploaded_file)
    all_programs = list(ratings.keys())
    all_time_slots = list(range(6, 24))

    st.success("✅ CSV successfully loaded!")

    GEN = 100
    POP = 50
    EL_S = 2

    # Sliders for three trials
    st.header("⚙️ Genetic Algorithm Parameters (Three Trials)")
    trials = []
    for i in range(1, 4):
        st.subheader(f"Trial {i}")
        co_r = st.slider(f"Crossover Rate (Trial {i})", 0.0, 0.95, 0.8, 0.01, key=f"co_{i}")
        mut_r = st.slider(f"Mutation Rate (Trial {i})", 0.01, 0.05, 0.02, 0.01, key=f"mut_{i}")
        trials.append((co_r, mut_r))

    if st.button("Run Genetic Algorithm"):
        # Initialize population
        all_possible_schedules = initialize_pop(all_programs, all_time_slots)
        initial_best_schedule = finding_best_schedule(all_possible_schedules, ratings)

        st.header("🧠 Experiment Results")
        for i, (co_r, mut_r) in enumerate(trials, start=1):
            with st.expander(f"Trial {i} Results"):
                best_schedule = genetic_algorithm(
                    initial_best_schedule,
                    ratings,
                    all_programs,
                    GEN,
                    POP,
                    co_r,
                    mut_r,
                    EL_S,
                )

                # === FIX: Ensure table lengths match ===
                time_slots_formatted = [f"{t:02d}:00" for t in all_time_slots]
                programs_for_slots = best_schedule

                if len(programs_for_slots) < len(time_slots_formatted):
                    programs_for_slots += [
                        random.choice(all_programs)
                        for _ in range(len(time_slots_formatted) - len(programs_for_slots))
                    ]
                elif len(programs_for_slots) > len(time_slots_formatted):
                    programs_for_slots = programs_for_slots[: len(time_slots_formatted)]

                schedule_data = {
                    "Time Slot": time_slots_formatted,
                    "Program": programs_for_slots,
                }

                df_schedule = pd.DataFrame(schedule_data)

                st.markdown(f"**Parameters Used:**  CO_R = {co_r}, MUT_R = {mut_r}")
                st.table(df_schedule)
                st.markdown(f"**Total Ratings:** {fitness_function(best_schedule, ratings)}")

else:
    st.info("Please upload a CSV file to begin.")
