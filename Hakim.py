import streamlit as st
import csv
import random
import time

# ---------- CSV reading ----------
def read_csv_to_dict(file):
    """
    Expects CSV with header: TimeSlot,Slot1,Slot2,...
    Each subsequent row: ProgramName, rating_for_slot0, rating_for_slot1, ...
    Returns: programs (list), timeslots (list), ratings (dict program->list of floats)
    """
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
    # schedule is list of program names in order of time slots
    # ratings[program] is list of floats for those time slots
    total_rating = 0.0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

# ---------- Population initialization ----------
def initialize_population(programs, pop_size):
    # create pop_size random permutations (avoid duplicates where possible)
    population = []
    seen = set()
    for _ in range(pop_size):
        perm = tuple(random.sample(programs, len(programs)))
        # try to avoid exact duplicate population members
        attempts = 0
        while perm in seen and attempts < 10:
            perm = tuple(random.sample(programs, len(programs)))
            attempts += 1
        seen.add(perm)
        population.append(list(perm))
    return population

# ---------- Selection (tournament) ----------
def tournament_selection(population, ratings, k=3):
    # pick k random and return the best
    competitors = random.sample(population, k)
    best = max(competitors, key=lambda s: fitness_function(s, ratings))
    return best

# ---------- Ordered Crossover (OX) for permutations ----------
def ordered_crossover(parent1, parent2):
    size = len(parent1)
    if size < 2:
        return parent1[:], parent2[:]
    a, b = sorted(random.sample(range(size), 2))
    # child will have slice from parent1 and filled from parent2 in order
    child1 = [None]*size
    child1[a:b+1] = parent1[a:b+1]
    fill_pos = (b+1) % size
    p2_idx = (b+1) % size
    while None in child1:
        if parent2[p2_idx] not in child1:
            child1[fill_pos] = parent2[p2_idx]
            fill_pos = (fill_pos + 1) % size
        p2_idx = (p2_idx + 1) % size

    child2 = [None]*size
    child2[a:b+1] = parent2[a:b+1]
    fill_pos = (b+1) % size
    p1_idx = (b+1) % size
    while None in child2:
        if parent1[p1_idx] not in child2:
            child2[fill_pos] = parent1[p1_idx]
            fill_pos = (fill_pos + 1) % size
        p1_idx = (p1_idx + 1) % size

    return child1, child2

# ---------- Mutation (swap mutation) ----------
def mutate_swap(schedule, mutation_rate):
    schedule = schedule[:]  # copy
    size = len(schedule)
    # with probability mutation_rate, perform a swap (one or multiple)
    for i in range(size):
        if random.random() < mutation_rate:
            j = random.randrange(size)
            schedule[i], schedule[j] = schedule[j], schedule[i]
    return schedule

# ---------- Evolve population ----------
def evolve_population(population, ratings, elite_size=1, mutation_rate=0.05, tournament_k=3):
    pop_size = len(population)
    # evaluate fitness once
    scored = [(fitness_function(s, ratings), s) for s in population]
    scored.sort(reverse=True, key=lambda x: x[0])
    new_pop = [s for (_, s) in scored[:elite_size]]  # elitism

    # produce rest with selection + crossover + mutation
    while len(new_pop) < pop_size:
        parent1 = tournament_selection(population, ratings, k=tournament_k)
        parent2 = tournament_selection(population, ratings, k=tournament_k)
        # avoid identical parents
        if parent1 == parent2:
            parent2 = tournament_selection(population, ratings, k=tournament_k)
        child1, child2 = ordered_crossover(parent1, parent2)
        child1 = mutate_swap(child1, mutation_rate)
        child2 = mutate_swap(child2, mutation_rate)
        new_pop.append(child1)
        if len(new_pop) < pop_size:
            new_pop.append(child2)
    return new_pop

# ---------- GA run ----------
def run_ga(programs, ratings, pop_size=100, generations=200, elite_size=1, mutation_rate=0.05, tournament_k=3, progress_callback=None):
    population = initialize_population(programs, pop_size)
    best_schedule = None
    best_score = float("-inf")
    for gen in range(generations):
        # optionally report progress
        if progress_callback:
            progress_callback(gen, generations)
        population = evolve_population(population, ratings, elite_size=elite_size, mutation_rate=mutation_rate, tournament_k=tournament_k)
        # examine best in this population
        current_best = max(population, key=lambda s: fitness_function(s, ratings))
        current_score = fitness_function(current_best, ratings)
        if current_score > best_score:
            best_score = current_score
            best_schedule = current_best[:]
    return best_schedule, best_score

# ---------- Streamlit UI ----------
st.title("Program Scheduling GA")

uploaded_file = st.file_uploader("Upload CSV (first column program, rest slot ratings)", type=["csv"])
if uploaded_file:
    programs, timeslots, ratings = read_csv_to_dict(uploaded_file)
    st.write(f"Detected {len(programs)} programs and {len(timeslots)} time slots.")
    st.write("Time slots:", timeslots)
    st.write("Programs (sample):", programs[:10])

    # GA parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        pop_size = st.number_input("Population size", min_value=10, max_value=2000, value=200, step=10)
        generations = st.number_input("Generations", min_value=1, max_value=5000, value=500, step=10)
    with col2:
        elite_size = st.number_input("Elitism (top N kept)", min_value=0, max_value=10, value=2)
        mutation_rate = st.number_input("Mutation rate (0-1)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    with col3:
        tournament_k = st.number_input("Tournament size (selection)", min_value=2, max_value=len(programs), value=3)
        seed = st.checkbox("Set random seed for reproducibility")
    if seed:
        random.seed(42)

    if st.button("Run GA"):
        start = time.time()
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(gen, gens):
            progress_bar.progress(int((gen / max(1, gens - 1)) * 100))
            status_text.text(f"Gen {gen+1}/{gens}")

        best_schedule, best_score = run_ga(programs, ratings,
                                           pop_size=int(pop_size),
                                           generations=int(generations),
                                           elite_size=int(elite_size),
                                           mutation_rate=float(mutation_rate),
                                           tournament_k=int(tournament_k),
                                           progress_callback=progress_callback)
        progress_bar.progress(100)
        status_text.text("Done")
        elapsed = time.time() - start

        st.subheader("Best schedule found")
        st.write(f"Total score: {best_score:.3f}   (found in {elapsed:.1f}s)")
        # display schedule as a mapping time slot -> program -> rating
        rows = []
        for idx, program in enumerate(best_schedule):
            slot_label = timeslots[idx] if idx < len(timeslots) else f"Slot {idx}"
            rating = ratings[program][idx]
            rows.append((idx+1, slot_label, program, rating))
        # show as table
        st.write("Order (time slot index, time label, program, rating at that slot):")
        st.table(rows)

        # also show schedule as list
        st.write("Schedule order (programs):")
        st.write(best_schedule)

        # offer download as CSV
        import io, csv
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["slot_index", "slot_label", "program", "rating"])
        for r in rows:
            writer.writerow(r)
        st.download_button("Download schedule CSV", data=output.getvalue(), file_name="best_schedule.csv", mime="text/csv")
