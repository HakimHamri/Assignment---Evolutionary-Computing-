import streamlit as st
import pandas as pd
import random
import io

st.title("TV Program Scheduling with Genetic Algorithm")

st.markdown(
    "Upload a CSV where the first column is program name and the remaining columns are "
    "ratings for each time slot (in order). Example header: Program,06:00,07:00,..."
)

uploaded_file = st.file_uploader("Upload program_ratings.csv", type=["csv"])

st.sidebar.header("GA parameters")
generations = st.sidebar.number_input("Generations", min_value=1, value=200, step=10)
population_size = st.sidebar.number_input("Population size", min_value=2, value=100, step=1)
crossover_rate = st.sidebar.slider("Crossover rate", 0.0, 1.0, 0.8)
mutation_rate = st.sidebar.slider("Mutation rate", 0.0, 1.0, 0.2)
elitism = st.sidebar.number_input("Elitism (top N kept)", min_value=0, value=2, step=1)
repeats_allowed = st.sidebar.checkbox("Allow program repeats across slots", value=True)
tournament_k = st.sidebar.number_input("Tournament size (selection)", min_value=2, value=3, step=1)
random_seed = st.sidebar.number_input("Random seed (0 for random)", value=0, step=1)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if df.shape[1] < 2:
        st.error("CSV must have at least two columns: program and ratings for at least one time slot.")
        st.stop()

    # prepare ratings dict
    programs = df.iloc[:, 0].astype(str).tolist()
    ratings_lists = []
    for col in df.columns[1:]:
        # try convert columns to numeric
        try:
            _ = pd.to_numeric(df[col])
        except Exception:
            st.error(f"Column {col} cannot be converted to numeric ratings.")
            st.stop()
    for idx, row in df.iterrows():
        ratings_row = []
        for val in row[1:]:
            ratings_row.append(float(val))
        ratings_lists.append(ratings_row)
    ratings = {programs[i]: ratings_lists[i] for i in range(len(programs))}

    # time slot labels
    header_times = list(df.columns[1:])
    slots = len(header_times)

    st.write(f"Detected {len(programs)} programs and {slots} time slots.")
    st.write("Time slots:", header_times)

    if random_seed != 0:
        random.seed(random_seed)

    # Fitness function
    def fitness_function(schedule):
        total = 0.0
        for slot_idx, program in enumerate(schedule):
            prog_ratings = ratings.get(program)
            # If missing or rating index out of range -> heavy penalty
            if prog_ratings is None or slot_idx >= len(prog_ratings):
                total -= 1e6
            else:
                total += prog_ratings[slot_idx]
        return total

    # Initialization
    def init_population_repeats(pop_size):
        pop = []
        for _ in range(pop_size):
            schedule = [random.choice(programs) for _ in range(slots)]
            pop.append(schedule)
        return pop

    def init_population_unique(pop_size):
        if len(programs) < slots:
            st.error("Not enough distinct programs to fill all slots with unique assignment.")
            st.stop()
        pop = []
        for _ in range(pop_size):
            schedule = random.sample(programs, slots)
            pop.append(schedule)
        return pop

    # Selection: tournament
    def tournament_selection(population):
        contenders = random.sample(population, min(tournament_k, len(population)))
        contenders.sort(key=fitness_function, reverse=True)
        return contenders[0]

    # Crossover & mutation for repeats-allowed (simple one-point)
    def one_point_crossover(a, b):
        point = random.randint(1, len(a) - 1)
        child1 = a[:point] + b[point:]
        child2 = b[:point] + a[point:]
        return child1, child2

    def mutate_replace(schedule):
        idx = random.randrange(len(schedule))
        schedule[idx] = random.choice(programs)
        return schedule

    # Crossover & mutation for unique (order crossover, swap mutation)
    def order_crossover(parent1, parent2):
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        # copy slice from parent1
        child[a:b] = parent1[a:b]
        # fill remaining from parent2 in order
        p2_iter = [p for p in parent2 if p not in child[a:b]]
        pos = 0
        for i in range(size):
            if child[i] is None:
                child[i] = p2_iter[pos]
                pos += 1
        return child

    def swap_mutation(schedule):
        i, j = random.sample(range(len(schedule)), 2)
        schedule[i], schedule[j] = schedule[j], schedule[i]
        return schedule

    # GA
    def genetic_algorithm():
        if repeats_allowed:
            population = init_population_repeats(population_size)
        else:
            population = init_population_unique(population_size)

        progress_bar = st.progress(0)
        best_overall = None
        best_score = float("-inf")

        for gen in range(1, generations + 1):
            # sort and elitism
            population.sort(key=fitness_function, reverse=True)
            if fitness_function(population[0]) > best_score:
                best_score = fitness_function(population[0])
                best_overall = population[0].copy()

            new_pop = []
            # keep elites
            new_pop.extend(population[:elitism])

            while len(new_pop) < population_size:
                parent1 = tournament_selection(population)
                parent2 = tournament_selection(population)
                # crossover
                if random.random() < crossover_rate:
                    if repeats_allowed:
                        child1, child2 = one_point_crossover(parent1, parent2)
                    else:
                        child1 = order_crossover(parent1, parent2)
                        child2 = order_crossover(parent2, parent1)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # mutation
                if random.random() < mutation_rate:
                    if repeats_allowed:
                        child1 = mutate_replace(child1)
                    else:
                        child1 = swap_mutation(child1)
                if random.random() < mutation_rate:
                    if repeats_allowed:
                        child2 = mutate_replace(child2)
                    else:
                        child2 = swap_mutation(child2)

                new_pop.append(child1)
                if len(new_pop) < population_size:
                    new_pop.append(child2)

            population = new_pop

            # update progress and display best
            progress_bar.progress(gen / generations)
            if gen % max(1, generations // 10) == 0 or gen == generations:
                st.write(f"Generation {gen}/{generations} — best score so far: {best_score:.2f}")

        # final best
        population.sort(key=fitness_function, reverse=True)
        final_best = population[0]
        final_score = fitness_function(final_best)
        if final_score > best_score:
            best_overall, best_score = final_best, final_score

        return best_overall, best_score

    if st.button("Run GA"):
        with st.spinner("Running genetic algorithm..."):
            best_schedule, best_score = genetic_algorithm()

        st.success("GA finished")
        # display schedule
        schedule_df = pd.DataFrame({
            "Time Slot": header_times,
            "Program": best_schedule,
            "Rating": [ratings[p][i] for i, p in enumerate(best_schedule)]
        })
        st.write("Final optimal schedule:")
        st.dataframe(schedule_df)
        st.write(f"Total Ratings: {best_score:.2f}")

        # CSV download
        csv_buf = io.StringIO()
        schedule_df.to_csv(csv_buf, index=False)
        st.download_button("Download schedule CSV", csv_buf.getvalue().encode('utf-8'), file_name="best_schedule.csv", mime="text/csv")
else:
    st.info("Please upload a CSV file to begin.")
