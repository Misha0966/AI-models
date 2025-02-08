using Random  # Подключаем библиотеку для случайных чисел
using Plots  # Подключаем библиотеку для графиков
using CSV  # Подключаем библиотеку для работы с CSV файлами
using DataFrames  # Подключаем библиотеку для работы с таблицами данных

# Параметры модели
grid_size = 100  # Размер сетки среды (100x100 клеток)
initial_population_size = 50  # Начальный размер популяции (50 организмов)
steps = 300  # Количество шагов, которые мы будем симулировать
food_amount = 100  # Количество пищи в каждой клетке
toxic_zone_count = 1  # Количество токсичных зон (1 токсичная зона)
reproduction_threshold = 200  # Порог энергии, при котором организм может размножаться
mutation_rate = 0.99  # Скорость мутаций (как часто происходят изменения в генах)

# Классы

# Микробная клетка — описание одного микроорганизма
mutable struct MicrobeCell
energy::Int  # Энергия клетки
speed::Int  # Скорость клетки
strategy::String  # Стратегия клетки (например, "поиск пищи" или "избегание опасности")
memory::Dict{String, Any}  # Память клетки для обучения (например, сколько пищи она съела)
end

# Организм — многоклеточный организм, состоящий из нескольких клеток
mutable struct MultiCellOrganism
cells::Vector{MicrobeCell}  # Список клеток, из которых состоит организм
x::Int  # Координата X организма на поле
y::Int  # Координата Y организма на поле
energy::Int  # Энергия организма
genome::Dict{String, Any}  # Геном организма, который определяет его поведение
memory::Dict{String, Any}  # Метапамять для адаптации организма
end

# Среда, в которой обитают организмы
struct Environment
grid::Matrix{Int}  # Матрица ресурсов (например, пищи) в каждой клетке
toxic_zones::Vector{Tuple{Int, Int}}  # Список координат токсичных зон
end

# Функции для инициализации:

# Функция случайного размещения (получение случайных координат)
function random_position()
return (rand(1:grid_size), rand(1:grid_size))  # Возвращаем случайные координаты в пределах размера сетки
end

# Функция для создания среды (например, заполняем сетку пищей и добавляем токсичные зоны)
function create_environment()
grid = fill(food_amount, grid_size, grid_size)  # Заполняем сетку клеток числом, обозначающим количество пищи
toxic_zones = [random_position() for _ in 1:toxic_zone_count]  # Создаём случайные токсичные зоны
for (x, y) in toxic_zones
grid[x, y] = -1  # Отмечаем токсичные зоны в сетке значением -1
end
return Environment(grid, toxic_zones)  # Возвращаем объект среды с сеткой и токсичными зонами
end

# Функция для инициализации популяции организмов
function initialize_population(size::Int)
population = Vector{MultiCellOrganism}()  # Создаём пустой список для популяции
for _ in 1:size  # Для каждого организма в популяции
x, y = random_position()  # Генерируем случайные координаты для организма
energy = rand(50:100)  # Генерируем случайное значение энергии для организма
cells = [MicrobeCell(rand(10:20), rand(1:5), "neutral", Dict()) for _ in 1:rand(3:7)]  # Генерируем клетки для организма
genome = Dict("mutation_rate" => mutation_rate, "aggression" => rand(0.1:0.1:1.0))  # Геном организма
memory = Dict("food_collected" => 0, "steps_survived" => 0)  # Изначальная память организма (сколько пищи собрал и сколько шагов прошёл)
push!(population, MultiCellOrganism(cells, x, y, energy, genome, memory))  # Добавляем нового организма в популяцию
end
return population  # Возвращаем популяцию
end

# Функции поведения:

# Функция поиска пищи для организма
function search_food!(organism::MultiCellOrganism, env::Environment)
if env.grid[organism.x, organism.y] > 0  # Если в текущей клетке есть пища
food_collected = min(env.grid[organism.x, organism.y], 10)  # Организм собирает до 10 единиц пищи
organism.energy += food_collected  # Организм увеличивает свою энергию
env.grid[organism.x, organism.y] -= food_collected  # Уменьшаем количество пищи в клетке
organism.memory["food_collected"] += food_collected  # Обновляем память организма о собранной пище
end
end

# Функция проверки, попал ли организм в токсичную зону
function check_toxic_zone!(organism::MultiCellOrganism, env::Environment)
if env.grid[organism.x, organism.y] == -1  # Если организм в токсичной зоне
organism.energy -= 20  # Организм теряет 20 единиц энергии
end
end

# Функция размножения организма
function reproduce!(organism::MultiCellOrganism, population::Vector{MultiCellOrganism})
if organism.energy >= reproduction_threshold  # Если энергии достаточно для размножения
x, y = random_position()  # Генерируем случайные координаты для потомства
cells = deepcopy(organism.cells)  # Копируем клетки организма
genome = deepcopy(organism.genome)  # Копируем геном организма
memory = Dict("food_collected" => 0, "steps_survived" => 0)  # Память для потомства
organism.energy -= reproduction_threshold  # Организм тратит энергию на размножение
offspring = MultiCellOrganism(cells, x, y, 100, genome, memory)  # Создаём нового потомка
push!(population, offspring)  # Добавляем потомка в популяцию
end
end

# Функция мутации генома организма
function mutate_genome!(organism::MultiCellOrganism)
organism.genome["mutation_rate"] += rand(-0.01:0.01:0.01)  # Меняем скорость мутации случайным образом
organism.genome["mutation_rate"] = max(0.01, min(0.1, organism.genome["mutation_rate"]))  # Ограничиваем скорость мутации в пределах от 0.01 до 0.1
end

# Функция для симуляции всех шагов
function simulate!(population::Vector{MultiCellOrganism}, env::Environment, steps::Int)
anim = Animation()  # Создаём объект для анимации
results = DataFrame(step=Int[], x=Int[], y=Int[], energy=Int[], food_collected=Int[], steps_survived=Int[])  # Таблица для хранения результатов

for step in 1:steps  # Для каждого шага симуляции
for organism in population  # Для каждого организма в популяции
dx, dy = rand(-1:1), rand(-1:1)  # Организм случайным образом двигается на одну клетку
organism.x = clamp(organism.x + dx, 1, grid_size)  # Обновляем координаты X организма, чтобы они оставались в пределах сетки
organism.y = clamp(organism.y + dy, 1, grid_size)  # Обновляем координаты Y организма, чтобы они оставались в пределах сетки

search_food!(organism, env)  # Организм ищет пищу
check_toxic_zone!(organism, env)  # Организм проверяет, не попал ли в токсичную зону

organism.memory["steps_survived"] += 1  # Увеличиваем количество шагов, которое организм выжил

reproduce!(organism, population)  # Организм пытается размножиться
mutate_genome!(organism)  # Организм мутирует

push!(results, (step, organism.x, organism.y, organism.energy, organism.memory["food_collected"], organism.memory["steps_survived"]))  # Записываем данные о текущем шаге в таблицу
end

scatter(
[o.x for o in population], [o.y for o in population],  # Рисуем координаты всех организмов
color=[o.energy > 150 ? :green : :blue for o in population],  # Организмы с энергией больше 150 будут зелёными
title="Step $step",  # Заголовок графика с номером шага
xlims=(1, grid_size), ylims=(1, grid_size)  # Ограничиваем оси графика размером сетки
        )
frame(anim)  # Добавляем текущий кадр в анимацию
end

gif(anim, "microbial_simulation.gif", fps=10)  # Создаём анимацию и сохраняем её в файл
CSV.write("microbial_results.csv", results)  # Сохраняем результаты в CSV файл
end

# Запуск симуляции
env = create_environment()  # Создаём среду
population = initialize_population(initial_population_size)  # Инициализируем популяцию
simulate!(population, env, steps)  # Запускаем симуляцию

println("Готово! Результаты в 'microbial_simulation.gif' и 'microbial_results.csv'.")  # Выводим сообщение о завершении симуляции