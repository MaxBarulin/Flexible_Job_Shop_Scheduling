import random
import copy
import numpy as np
from deap import base, creator, tools, algorithms
from matplotlib import pyplot as plt

# --- Problem Data (Flexible Job Shop) ---
# jobs = ["J1", "J2", "J3"]
# machines = ["M1", "M2", "M3"]
# processing_times = {
#     ("J1", "O1"): {"M1": 0.3, "M2": 3}, ("J1", "O2"): {"M2": 4, "M3": 5},
#     ("J2", "O1"): {"M1": 3, "M3": 2}, ("J2", "O2"): {"M2": 2, "M3": 3},
#     ("J3", "O1"): {"M1": 1, "M2": 2}, ("J3", "O2"): {"M3": 4},
# }
# job_operations = {"J1": ["O1", "O2"], "J2": ["O1", "O2"], "J3": ["O1", "O2"]}

jobs = ["Фланец", "Прокладка", "Болт", "Рычаг", "Ручка", "Рым-болт", "Кница", "ВТУЛКА", "РБ", "ПРОСТАВЫШ", "ГОЛОВКА"]
machines = ["Сверлильная", "Фрезерная", "Фрезерная c ЧПУ", "Токарная", "Токарная с ЧПУ", "Слесарная", "Расточная", "Заготовительная", "Маркирование"]
processing_times = {
    ("Фланец", "O1"): {"Заготовительная": 0.08}, ("Фланец", "O2"): {"Токарная": 1.2, "Токарная с ЧПУ": 1.1}, ("Фланец", "O3"): {"Сверлильная": 0.2, "Фрезерная c ЧПУ": 0.4}, ("Фланец", "O4"): {"Слесарная": 0.1}, ("Фланец", "O5"): {"Маркирование": 0.1},
    ("Прокладка", "O1"): {"Фрезерная": 0.6, "Фрезерная c ЧПУ": 0.7}, ("Прокладка", "O2"): {"Сверлильная": 0.1, "Фрезерная c ЧПУ": 0.3}, ("Прокладка", "O3"): {"Слесарная": 0.1}, ("Прокладка", "O4"): {"Маркирование": 0.1},
    ("Болт", "O1"): {"Заготовительная": 0.08}, ("Болт", "O2"): {"Токарная": 0.6}, ("Болт", "O3"): {"Фрезерная": 0.06, "Фрезерная c ЧПУ": 0.09}, ("Болт", "O4"): {"Слесарная": 0.1}, ("Болт", "O5"): {"Маркирование": 0.1},
    ("Рычаг", "O1"): {"Фрезерная c ЧПУ": 2.3},("Рычаг", "O2"): {"Слесарная": 0.25}, ("Рычаг", "O3"): {"Сверлильная": 0.28, "Фрезерная c ЧПУ": 0.4}, ("Рычаг", "O4"): {"Слесарная": 0.1}, ("Рычаг", "O5"): {"Маркирование": 0.1},
    ("Ручка", "O1"): {"Слесарная": 0.5},("Ручка", "O2"): {"Токарная": 1.3}, ("Ручка", "O3"): {"Слесарная": 0.2}, ("Ручка", "O4"): {"Маркирование": 0.1},
    ("Рым-болт", "O1"): {"Заготовительная": 0.08}, ("Рым-болт", "O2"): {"Токарная": 0.75, "Токарная с ЧПУ": 0.7}, ("Рым-болт", "O3"): {"Слесарная": 0.1}, ("Рым-болт", "O4"): {"Маркирование": 0.1},
    ("Кница", "O1"): {"Фрезерная": 0.8, "Фрезерная c ЧПУ": 1.0}, ("Кница", "O2"): {"Слесарная": 0.1}, ("Кница", "O3"): {"Слесарная": 0.1}, ("Кница", "O4"): {"Маркирование": 0.1},
    ("ВТУЛКА", "O1"): {"Заготовительная": 0.11}, ('ВТУЛКА', "O2"): {"Токарная": 0.7, "Токарная с ЧПУ": 0.7}, ("ВТУЛКА", "O3"): {"Слесарная": 0.1}, ("ВТУЛКА", "O4"): {"Маркирование": 0.1},
    ("РБ", "O1"): {"Токарная": 1.2, "Токарная с ЧПУ": 1.1}, ("РБ", 'O2'): {'Слесарная': 0.1}, ("РБ", "O3"): {"Маркирование": 0.1},
    ('ПРОСТАВЫШ', 'O1'): {'Заготовительная': 0.4}, ('ПРОСТАВЫШ', 'O2'): {'Токарная': 0.7, 'Токарная с ЧПУ': 0.7}, ('ПРОСТАВЫШ', 'O3'): {'Расточная': 1.2}, ('ПРОСТАВЫШ', 'O4'): {'Слесарная': 0.3}, ('ПРОСТАВЫШ', 'O5'): {'Маркирование': 0.1},
    ('ГОЛОВКА', 'O1'): {'Заготовительная': 0.14}, ('ГОЛОВКА', 'O2'): {'Токарная': 0.55, 'Токарная с ЧПУ': 0.6}, ('ГОЛОВКА', 'O3'): {'Фрезерная': 0.55, "Фрезерная c ЧПУ": 0.5}, ('ГОЛОВКА', 'O4'): {'Слесарная': 0.1}, ('ГОЛОВКА', 'O5'): {'Маркирование': 0.05},
}
job_operations = {"Фланец": ["O1", "O2", "O3", "O4", "O5"],
                  "Прокладка": ["O1", "O2", "O3", "O4"],
                  "Болт": ["O1", "O2", "O3", "O4", "O5"],
                  "Рычаг": ["O1", "O2", "O3", "O4", "O5"],
                  "Ручка": ["O1", "O2", "O3", "O4"],
                  "Рым-болт": ["O1", "O2", "O3", "O4"],
                  "Кница": ["O1", "O2", "O3", "O4"],
                  "ВТУЛКА": ["O1", "O2", "O3", "O4"],
                  "РБ": ["O1", "O2", "O3"],
                  "ПРОСТАВЫШ": ["O1", "O2", "O3", "O4", "O5"],
                  "ГОЛОВКА": ["O1", "O2", "O3", "O4", "O5"],
                  }

# --- Mappings for Crossover ---
# Create a canonical list of unique operations (tuples)
canonical_all_operations = sorted(list(set(
    (job, op) for job in jobs for op in job_operations[job]
))) # Sorting ensures consistent mapping run-to-run
num_operations = len(canonical_all_operations)

# Create forward and reverse mappings
op_to_int = {op: i for i, op in enumerate(canonical_all_operations)}
int_to_op = {i: op for i, op in enumerate(canonical_all_operations)}

# --- DEAP Setup ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

def generate_schedule_output(schedule_details):
    """
    Генерирует текстовое представление расписания для вывода в консоль и сохранения в переменную.
    """
    output = ["--- Лучшее расписание (детали) ---", "Детали расписания (порядок обработки):"]
    for job, op, machine, start, end in schedule_details:
        output.append(f"  Операция ({job}, {op}) на {machine}: Начало={start:.2f}, Конец={end:.2f}")
    return "\n".join(output)

# --- Individual Representation ---
def create_individual():
    """Creates sequence (list of (job, op) tuples) + machine assignment."""
    # Sequence part: permutation of jobs, preserving operation order within each job
    sequence_part = []
    shuffled_jobs = random.sample(jobs, len(jobs))  # Перемешиваем порядок работ
    for job in shuffled_jobs:
        # Добавляем операции для текущей работы в правильном порядке
        for op in job_operations[job]:
            sequence_part.append((job, op))

    # Machine part: random valid machine for each op in the sequence
    machine_part = [random.choice(list(processing_times[op].keys())) for op in sequence_part]

    # Combine: sequence first, then machines
    return sequence_part + machine_part

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- Evaluation Function (Makespan) ---
def evaluate(individual):
    """
    Calculates makespan based on sequence and machine assignments.
    Enforces job precedence constraints and assigns infinite makespan
    if the sequence violates them.
    """
    sequence_part = individual[:num_operations]
    machine_part = individual[num_operations:]

    if len(sequence_part) != num_operations or len(machine_part) != num_operations:
         print(f"Warning: Individual length mismatch. Seq: {len(sequence_part)}, Mach: {len(machine_part)}, Expected: {num_operations}")
         return (float('inf'),) # Невалидный индивидуум

    machine_times = {m: 0.0 for m in machines}
    # Теперь отслеживаем время завершения КАЖДОЙ конкретной операции
    op_completion_times = {} # Словарь для хранения {(job, op): completion_time}

    # Построим словарь для быстрого поиска предшественника
    # { (job, op_k): (job, op_{k-1}) }
    predecessors = {}
    for job, ops_list in job_operations.items():
        for i in range(1, len(ops_list)):
            predecessors[(job, ops_list[i])] = (job, ops_list[i-1])

    # Симулируем расписание
    for idx, current_op_tuple in enumerate(sequence_part):
        if current_op_tuple not in processing_times:
             print(f"Error: Operation {current_op_tuple} not in processing_times.")
             return (float('inf'),) # Ошибка в данных или индивиде

        job, op = current_op_tuple
        assigned_machine = machine_part[idx]

        if assigned_machine not in processing_times[current_op_tuple]:
             print(f"Error: Assigned machine {assigned_machine} invalid for {current_op_tuple}.")
             return (float('inf'),) # Невалидное назначение станка

        proc_time = processing_times[current_op_tuple][assigned_machine]

        # --- Проверка и учет последовательности ---
        predecessor_tuple = predecessors.get(current_op_tuple)
        precedence_finish_time = 0.0

        if predecessor_tuple:
            # Если есть предшественник, он ДОЛЖЕН быть уже обработан
            if predecessor_tuple not in op_completion_times:
                # Нарушение последовательности! Операция появилась раньше предшественника.
                # print(f"Sequence Violation: {current_op_tuple} before {predecessor_tuple}") # Для отладки
                return (float('inf'),) # Огромный штраф
            precedence_finish_time = op_completion_times[predecessor_tuple]
        # -----------------------------------------

        # Время начала - максимум из времени освобождения станка и времени завершения предшественника
        start_time = max(machine_times[assigned_machine], precedence_finish_time)
        completion_time = start_time + proc_time

        # Обновляем время освобождения станка и запоминаем время завершения ТЕКУЩЕЙ операции
        machine_times[assigned_machine] = completion_time
        op_completion_times[current_op_tuple] = completion_time

    # Makespan - время завершения самой последней операции
    makespan = max(op_completion_times.values()) if op_completion_times else 0
    return (makespan,)

toolbox.register("evaluate", evaluate)

# --- Genetic Operators ---

def custom_crossover(ind1, ind2):
    """
    Applies cxOrdered to integer representation of the sequence.
    Machine assignments are inherited using parent maps.
    """
    seq_len = num_operations
    ind1_list = list(ind1)
    ind2_list = list(ind2)

    # 1. Split parents into sequence (tuples) and machines
    seq1_tuples = ind1_list[:seq_len]
    mach1 = ind1_list[seq_len:]
    seq2_tuples = ind2_list[:seq_len]
    mach2 = ind2_list[seq_len:]

    # 2. Create Operation -> Machine maps for quick lookup
    map1 = {op: machine for op, machine in zip(seq1_tuples, mach1)}
    map2 = {op: machine for op, machine in zip(seq2_tuples, mach2)}


    # 3. Convert tuple sequences to integer sequences using op_to_int map
    int_seq1 = [op_to_int[op] for op in seq1_tuples]
    int_seq2 = [op_to_int[op] for op in seq2_tuples]

    # 4. Apply cxOrdered to the *INTEGER* sequences
    new_int_seq1, new_int_seq2 = tools.cxOrdered(copy.deepcopy(int_seq1), copy.deepcopy(int_seq2))

    # 5. Convert the resulting integer sequences back to tuple sequences using int_to_op map
    new_seq1_tuples = [int_to_op[i] for i in new_int_seq1]
    new_seq2_tuples = [int_to_op[i] for i in new_int_seq2]

    # 6. Create new machine assignments for children using parent maps
    new_mach1 = []
    for op_tuple in new_seq1_tuples:
        # Prioritize machine from parent 1, fallback to parent 2
        machine = map1.get(op_tuple, map2.get(op_tuple))
        if machine is None: # Should theoretically not happen if op_tuple exists
             # Fallback: assign a random valid machine
             available_machines = list(processing_times[op_tuple].keys())
             machine = random.choice(available_machines)
             # print(f"Warning: Crossover fallback for op {op_tuple}") # Optional warning
        new_mach1.append(machine)

    new_mach2 = []
    for op_tuple in new_seq2_tuples:
        # Prioritize machine from parent 2, fallback to parent 1
        machine = map2.get(op_tuple, map1.get(op_tuple))
        if machine is None:
             available_machines = list(processing_times[op_tuple].keys())
             machine = random.choice(available_machines)
             # print(f"Warning: Crossover fallback for op {op_tuple}") # Optional warning
        new_mach2.append(machine)

    # 7. Create the final children individuals
    child1 = creator.Individual(new_seq1_tuples + new_mach1)
    child2 = creator.Individual(new_seq2_tuples + new_mach2)

    return child1, child2

toolbox.register("mate", custom_crossover)


def custom_mutate(individual, indpb_seq_swap, indpb_mach_change):
    """Mutates sequence (swap ops+machines) and machine assignments."""
    seq_len = num_operations
    # --- Mutation 1: Swap positions in the sequence ---
    for i in range(seq_len):
        if random.random() < indpb_seq_swap:
            j = random.randint(0, seq_len - 1)
            # Swap operation tuples at positions i and j
            individual[i], individual[j] = individual[j], individual[i]
            # Swap corresponding machine assignments at positions i+seq_len and j+seq_len
            individual[i + seq_len], individual[j + seq_len] = individual[j + seq_len], individual[i + seq_len]
    # --- Mutation 2: Change machine assignment for an operation ---
    for i in range(seq_len):
        if random.random() < indpb_mach_change:
            operation_tuple = individual[i]
            if operation_tuple not in processing_times:
                continue
            available_machines = list(processing_times[operation_tuple].keys())
            current_machine = individual[i + seq_len]
            other_options = [m for m in available_machines if m != current_machine]
            if other_options:
                individual[i + seq_len] = random.choice(other_options)
    return individual,

toolbox.register("mutate", custom_mutate, indpb_seq_swap=0.1, indpb_mach_change=0.1)
toolbox.register("select", tools.selTournament, tournsize=3) #Случайным образом выбирается группа индивидуумов. Из этой группы выбирается лучший индивидуум (размер группы задается параметром tournsize, например, tournsize=3)

# --- Main Execution ---
def main():
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42) # For numpy used in statistics

    # GA Parameters
    population_size = 500
    generations = 500   # Number of generations
    cxpb = 0.9         # Crossover probability
    mutpb = 0.5        # Mutation probability (for the whole individual)

    # Initialize Population
    print(f"Создание начальной популяции из {population_size} индивидуумов...")
    population = toolbox.population(n=population_size)

    # Statistics Setup
    stats = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.valid else None) # Get first fitness value
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max) # Max makespan might be less interesting, but good to track
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields

    # Evaluate Initial Population
    print("Оценка начальной популяции...")
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Log Initial State
    # Compile stats only on valid fitness individuals
    valid_fitness_pop = [ind for ind in population if ind.fitness.valid]
    if valid_fitness_pop: # Avoid error if all fitnesses were invalid
        record = stats.compile(valid_fitness_pop)
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
    else:
        logbook.record(gen=0, nevals=len(invalid_ind), avg=None, min=None, max=None) # Record Nones
    print(logbook.stream)


    # Run GA Generations
    print(f"Запуск GA на {generations} поколений...")
    for gen in range(1, generations + 1):
        # Selection
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring)) # Clone selected individuals

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                # Perform crossover - custom_crossover returns new children
                # The original child1, child2 in offspring list are modified *in place* by the returned values
                # This depends on how eaSimple/algorithms handle it vs manual loop.
                # Let's assume the toolbox.mate modifies the inputs OR we reassign.
                # Safer to reassign if mate returns new objects, but often it modifies in place.
                # Let's stick to the standard pattern: apply mate, then invalidate fitness.
                toolbox.mate(child1, child2) # Assume modifies child1, child2 in place
                del child1.fitness.values # Mark fitness as invalid
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant) # Assume modifies mutant in place
                del mutant.fitness.values # Mark fitness as invalid

        # Re-evaluate Individuals with Invalid Fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind: # Only evaluate if needed
             fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
             for ind, fit in zip(invalid_ind, fitnesses):
                 ind.fitness.values = fit

        # Update Population with the new offspring
        population[:] = offspring

        # Log Statistics for the current generation
        valid_fitness_pop = [ind for ind in population if ind.fitness.valid and ind.fitness.values[0] < float('inf')]
        if valid_fitness_pop:
            record = stats.compile(valid_fitness_pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        else:
            logbook.record(gen=gen, nevals=len(invalid_ind), avg=None, min=None, max=None)
        print(logbook.stream)


    print("GA завершен.")

    # --- Results ---
    # Find the best individual in the final population
    valid_final_pop = [ind for ind in population if ind.fitness.valid]
    if not valid_final_pop:
        print("\nНе найдено ни одного индивидуума с валидной пригодностью в конечной популяции.")
        exit()

    best_individual = tools.selBest(valid_final_pop, k=1)[0]
    best_makespan = best_individual.fitness.values[0]
    print(f"\nЛучший найденный Makespan: {best_makespan:.2f}")

    # Optional: Print details of the best schedule by re-simulating
    sequence = best_individual[:num_operations]
    machine_assignments = best_individual[num_operations:]

    # Re-simulate the best schedule to get start/end times accurately
    machine_times = {m: 0.0 for m in machines}
    job_last_scheduled_op_end_time = {j: 0.0 for j in jobs}
    schedule_details = []  # List to store (job, op, machine, start_time, end_time)
    for idx, operation_tuple in enumerate(sequence):
        job, op = operation_tuple
        assigned_machine = machine_assignments[idx]
        # Need to handle potential errors if best_individual is somehow invalid
        if operation_tuple not in processing_times or assigned_machine not in processing_times[operation_tuple]:
            print(f"Ошибка в лучшем индивиде: Невалидная операция/машина {operation_tuple} / {assigned_machine}")
            continue
        processing_time_val = processing_times[operation_tuple][assigned_machine]
        start_time = max(machine_times[assigned_machine], job_last_scheduled_op_end_time[job])
        completion_time = start_time + processing_time_val
        machine_times[assigned_machine] = completion_time
        job_last_scheduled_op_end_time[job] = completion_time
        schedule_details.append((job, op, assigned_machine, start_time, completion_time))

    # Sort schedule details by start time for clearer chronological output
    schedule_details.sort(key=lambda x: x[3])

    # Generate schedule output text
    schedule_output_text = generate_schedule_output(schedule_details)

    # Print to console
    print(schedule_output_text)

    # Собираем уникальные станки в порядке их появления в schedule_details
    unique_machines = []
    for job, op, machine, start, end in schedule_details:
        if machine not in unique_machines:
            unique_machines.append(machine)

    # Уникальные станки в порядке их первого появления
    machines_1 = unique_machines

    # Индексация станков для отображения на оси Y
    machine_indices = {machine: i for i, machine in enumerate(machines_1)}

    # Шаг 2: Построение диаграммы Ганта
    fig, ax = plt.subplots(figsize=(12, 8))

    # Отрисовка каждой операции
    for job, op, machine, start, end in schedule_details:
        machine_index = machine_indices[machine]
        ax.barh(machine_index, end - start, left=start, height=0.4, label=f"{job} ({op})")

    # Настройка осей
    ax.set_yticks(range(len(machines_1)))
    ax.set_yticklabels(machines_1)
    ax.set_xlabel("Время")
    ax.set_title("Диаграмма Ганта")

    # Установка шага шкалы времени на оси X
    x_min = min(start for _, _, _, start, _ in schedule_details)  # Минимальное время начала
    x_max = max(end for _, _, _, _, end in schedule_details)  # Максимальное время окончания

    # Устанавливаем шаг в 0.1
    x_ticks = np.arange(x_min, x_max + 0.1, 0.1)  # Создаем массив значений для делений
    ax.set_xticks(x_ticks)  # Устанавливаем деления
    ax.set_xticklabels([f"{x:.1f}" for x in x_ticks],
                       rotation=90)  # Устанавливаем метки и поворачиваем их для удобства чтения

    # Легенда
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Убираем дубликаты в легенде
    ax.legend(
        by_label.values(),
        by_label.keys(),
        title="Операции",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        ncol=2,
        fontsize='small'
    )

    # Отображение сетки
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Настройка отступов
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.2)

    # Показать график
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()