import streamlit as st
import csv
import random
import time
import io

# ---------- CSV reading ----------
def read_csv_to_dict(file):
    reader = csv.reader((line.decode('utf-8') if isinstance(line, bytes) else line) for line in file)
    header = next(reader)
    timeslots = header[1:]
    program_ratings = {}
    programs = []
    for row in reader:
        if not row:
            continue
        program = row[0]
        ratings = [float(x) for x in row[1:]]
        program_ratings[program] = ratings
        programs.append(program)
    return programs, timeslots, program_ratings

# ---------- Fitness ----------
def fitness_function(schedule, ratings):
    total_rating = 0.0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

# ---------- Population initialization ----------
def initialize_population(programs, pop_size):
    population = []
    seen = set()
    for _ in range(pop_size):
        perm = tuple(random.sample(programs, len(programs)))
        attempts = 0
        while perm in seen and attempts < 10:
            perm = tuple(random.sample(programs, len(programs)))
            attempts += 1
        seen.add(perm)
        population.append(list(perm))
    return population

# ---------- Selection (tournament) ----------
def tournament_selection(population, ratings, k=3):
    competitors = random.sample(population, k)
    best = max(competitors, key=lambda s: fitness_function(s, ratings))
    return best

# ---------- Ordered Crossover (OX) for permutations ----------
def ordered_crossover(parent1, parent2):
    size = len(parent1)
    if size < 2:
        return parent1[:], parent2[:]
    a, b = sorted(random.sample(range(size), 2))
    # child1
    child1 = [None] * size
    child1[a:b+1] = parent1[a:b+1]
    fill_pos = (b + 1) % size
    p2_idx = (b + 1) % size
    while None in child1:
        if parent2[p2_idx] not in child1:
            child1[fill_pos] = parent2[p2_idx]
            fill_pos = (fill_pos + 1) % size
        p2_idx = (p2_idx + 1) % size
    # child2
    child2 = [None] * size
    child2[a:b+1] = parent2[a:b+1]
    fill_pos = (b + 1) % size
    p1_idx = (b + 1) % size
    while None in child2:
        if parent1[p1_idx] not in child2:
            child2[fill_pos] = parent1[p1_idx]
            fill_pos = (fill_pos + 1) % size
        p1_idx = (p1_idx + 1) % size

    return child1, child2

# ---------- Mutation (swap mutation) ----------
def mutate_swap(schedule, mutation_rate):
    schedule = schedule[:]
    size = len(schedule)
    for i in range(size):
        if random.random() < mutation_rate:
            j = random.randrange(size)
            schedule[i], schedule[j] = schedule[j], schedule[i]
    return schedule

# ---------- Evolve population (uses CO_R as probability to crossover) ----------
def evolve_population(population, ratings, crossover_rate=0.8, mutation_rate=0.02, elite_size=1, tournament_k=3):
    pop_size = len(population)
    scored = [(fitness_function(s, ratings), s) for s in population]
    scored.sort(reverse=True, key=lambda x: x[0])
    new_pop = [s for (_, s) in scored[:elite_size]]  # elitism

    while len(new_pop) < pop_size:
        parent1 = tournament_selection(population, ratings, k=tournament_k)
        parent2 = tournament_selection(population, ratings, k=tournament_k)
        if parent1 == parent2:
            parent2 = tournament_selection(population, ratings, k=tournament_k)
        if random.random() < crossover_rate:
            child1, child2 = ordered_crossover(parent1, parent2)
        else:
            # no crossover: children are copies of parents
            child1, child2 = parent1[:], parent2[:]
        child1 = mutate_swap(child1, mutation_rate)
        child2 = mutate_swap(child2, mutation_rate)
        new_pop.append(child1)
        if len(new_pop) < pop_size:
            new_pop.append(child2)
    return new_pop

# ---------- GA run ----------
def run_ga(programs, ratings, pop_size=200, generations=500, elite_size=2, crossover_rate=0.8, mutation_rate=0.02, tournament_k=3, progress_callback=None):
    population = initialize_population(programs, pop_size)
    best_schedule = None
    best_score = float("-inf")
    for gen in range(generations):
        if progress_callback:
            progress_callback(gen, generations)
        population = evolve_population(population, ratings, crossover_rate=crossover_rate, mutation_rate=mutation_rate, elite_size=elite_size, tournament_k=tournament_k)
        current_best = max(population, key=lambda s: fitness_function(s, ratings))
        current_score = fitness_function(current_best, ratings)
        if current_score > best_score:
            best_score = current_score
            best_schedule = current_best[:]
    return best_schedule, best_score

# ---------- Streamlit UI ----------
st.title("Program Scheduling GA — 3 Trials with Different CO_R & MUT_R")

uploaded_file = st.file_uploader("Upload CSV (first column program, rest slot ratings)", type=["csv"])
if not uploaded_file:
    st.info("Please upload a CSV file first.")
    st.stop()

programs, timeslots, ratings = read_csv_to_dict(uploaded_file)
n_programs = len(programs)
n_slots = len(timeslots)

st.write(f"Detected {n_programs} programs and {n_slots} time slots.")
st.write("Time slots:", timeslots)

# GA general parameters
st.subheader("General GA parameters")
col1, col2, col3 = st.columns(3)
with col1:
    pop_size = st.number_input("Population size", min_value=10, max_value=2000, value=200, step=10)
with col2:
    generations = st.number_input("Generations", min_value=1, max_value=5000, value=500, step=10)
with col3:
    elite_size = st.number_input("Elitism (top N kept)", min_value=0, max_value=10, value=2)

tournament_k = st.number_input("Tournament size (selection)", min_value=2, max_value=max(2, n_programs), value=3)

st.markdown("---")
st.subheader("Trial-specific GA parameters")
st.write("Set CO_R (crossover rate) and MUT_R (mutation rate) for each of the 3 trials.")
st.write("CO_R range: 0.0 to 0.95 (default 0.8). MUT_R range: 0.01 to 0.05 (default 0.02).")

# Trial parameter inputs
trial_params = []
for i in range(1, 4):
    st.markdown(f"**Trial {i} parameters**")
    co_r = st.slider(f"Trial {i} CO_R (crossover rate)", min_value=0.0, max_value=0.95, value=0.8, step=0.01, key=f"co{i}")
    mut_r = st.slider(f"Trial {i} MUT_R (mutation rate)", min_value=0.01, max_value=0.05, value=0.02, step=0.005, key=f"mu{i}")
    trial_params.append((co_r, mut_r))

# Random seed checkbox
seed_checkbox = st.checkbox("Set random seed (for reproducibility)", value=False)
if seed_checkbox:
    random.seed(42)

if st.button("Run 3 Trials"):
    results = []
    overall_start = time.time()
    for idx, (co_r, mut_r) in enumerate(trial_params, start=1):
        st.write(f"Running Trial {idx} with CO_R={co_r:.3f}, MUT_R={mut_r:.3f} ...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(gen, gens, pb=progress_bar, stx=status_text):
            pb.progress(int((gen / max(1, gens - 1)) * 100))
            stx.text(f"Gen {gen+1}/{gens}")

        start = time.time()
        best_schedule, best_score = run_ga(programs, ratings,
                                           pop_size=int(pop_size),
                                           generations=int(generations),
                                           elite_size=int(elite_size),
                                           crossover_rate=float(co_r),
                                           mutation_rate=float(mut_r),
                                           tournament_k=int(tournament_k),
                                           progress_callback=progress_callback)
        progress_bar.progress(100)
        status_text.text("Done")
        elapsed = time.time() - start

        # If number of slots differs from program count, warn and adapt: use min(len(timeslots), len(programs))
        if len(best_schedule) != n_slots:
            st.warning(f"Number of programs ({len(best_schedule)}) != time slots ({n_slots}). Displaying only up to min count.")
        rows = []
        for slot_idx, program in enumerate(best_schedule):
            if slot_idx >= n_slots:
                slot_label = f"Slot {slot_idx}"
                rating = ratings[program][slot_idx] if slot_idx < len(ratings[program]) else None
            else:
                slot_label = timeslots[slot_idx]
                rating = ratings[program][slot_idx]
            rows.append((slot_idx + 1, slot_label, program, rating))

        # Show results for this trial
        st.subheader(f"Trial {idx} results")
        st.write(f"Parameters: CO_R = {co_r:.3f}, MUT_R = {mut_r:.3f}")
        st.write(f"Best total score: {best_score:.3f} (elapsed {elapsed:.2f}s)")
        st.write("Schedule (slot_index, slot_label, program, rating):")
        st.table(rows)

        # Save results for download later
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["slot_index", "slot_label", "program", "rating"])
        for r in rows:
            writer.writerow(r)
        results.append({
            "trial": idx,
            "co_r": co_r,
            "mut_r": mut_r,
            "score": best_score,
            "rows_csv": output.getvalue()
        })

    overall_elapsed = time.time() - overall_start
    st.success(f"All trials finished in {overall_elapsed:.2f}s")

    # Offer downloads per trial
    for res in results:
        st.download_button(f"Download Trial {res['trial']} schedule CSV (CO_R={res['co_r']:.3f}, MUT_R={res['mut_r']:.3f})",
                           data=res['rows_csv'],
                           file_name=f"trial_{res['trial']}_schedule.csv",
                           mime="text/csv")
