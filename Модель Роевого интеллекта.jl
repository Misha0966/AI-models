using Random  # Импортируем библиотеку для работы с случайными числами
using Plots  # Импортируем библиотеку для рисования графиков
using Printf  # Импортируем библиотеку для форматирования вывода текста

# Формула объема сферического сегмента.

# Функция для вычисления объема сегмента шара, где:
# R - радиус, H - высота сегмента
function spherical_segment_volume(R, H)

π * H^2 * (R - H / 3)  # Возвращаем расчет объема по формуле
end

# Инициализация частиц

# Функция для создания частиц в пространстве, где:
# num_particles - количество частиц
function initialize_particles(num_particles)

particles = [randn(2) * 200 for _ in 1:num_particles]  # Случайные координаты частиц
velocities = [randn(2) for _ in 1:num_particles]  # Случайные скорости для частиц
return particles, velocities  # Возвращаем начальные координаты и скорости
end

# Ограничение скорости частиц

# Функция для ограничения скорости частиц:
function clamp_velocity!(velocities, vmax)

for v in velocities # velocities - текущие скорости частиц, vmax - максимальная скорость
for i in 1:2  # Перебираем каждую координату (X и Y)
v[i] = clamp(v[i], -vmax, vmax)  # Ограничиваем скорость в пределах от -vmax до vmax
end
end
end

# Алгоритм PSO

# Основная функция алгоритма оптимизации частиц (PSO), где:
# num_particles - количество частиц, num_iterations - количество итераций, vmax - максимальная скорость

function pso_3d(num_particles, num_iterations, vmax = 10.0)
particles, velocities = initialize_particles(num_particles)  # Инициализация частиц и их скоростей
personal_best_positions = copy(particles)  # Копируем начальные позиции частиц как лучшие
personal_best_scores = [spherical_segment_volume(p[1], p[2]) for p in particles]  # Оценка начальных позиций по объему
global_best_position = personal_best_positions[argmin(personal_best_scores)]  # Находим глобальную лучшую позицию
global_best_score = minimum(personal_best_scores)  # Глобальный лучший результат

w_min = 0.4  # Минимальное значение инерции
w_max = 0.9  # Максимальное значение инерции
c1 = 1.5  # Коэффициент личной привязанности
c2 = 1.5  # Коэффициент глобальной привязанности
particle_paths = [Vector{Vector{Float64}}() for _ in 1:num_particles]  # История движения частиц

global_best_history = [] # Хранение истории глобального лучшего результата

for iter in 1:num_iterations

w = w_max - (w_max - w_min) * iter / num_iterations # Инициализация веса инерции в зависимости от итерации

for i in 1:num_particles
ai_prediction = randn(2) * 0.1  # Добавление случайного прогноза на основе ИИ

# Обновление скорости частиц

velocities[i] .= w .* velocities[i] +  # Обновление с учетом инерции
c1 * rand() .* (personal_best_positions[i] .- particles[i]) +  # Личная привязанность
c2 * rand() .* (global_best_position .- particles[i]) +  # Глобальная привязанность
ai_prediction  # Прогноз ИИ для улучшения движения

clamp_velocity!(velocities, vmax)  # Ограничение скорости
particles[i] .+= velocities[i]  # Обновление позиции частицы

# Обновление личных лучших результатов

current_score = spherical_segment_volume(particles[i][1], particles[i][2])  # Оценка новой позиции
if current_score < personal_best_scores[i]
personal_best_positions[i] = particles[i]  # Обновление личного лучшего положения
personal_best_scores[i] = current_score  # Обновление лучшего результата
end

push!(particle_paths[i], copy(particles[i])) # Сохранение текущей позиции частицы для последующей визуализации
end

# Обновление глобального лучшего результата

min_idx = argmin(personal_best_scores)  # Находим индекс наименьшего значения
if personal_best_scores[min_idx] < global_best_score  # Если найден лучший результат
global_best_position = personal_best_positions[min_idx]  # Обновляем глобальное лучшее положение
global_best_score = personal_best_scores[min_idx]  # Обновляем глобальный лучший результат
end

push!(global_best_history, global_best_score) # Сохранение текущего лучшего результата в историю

# Вывод промежуточных результатов в консоль

println("Итерация: $iter")
println("Глобальное лучшее положение: $global_best_position")
println("Глобальный лучший результат: $global_best_score")
println("-"^50)
end

# Вывод окончательных лучших результатов

println("Лучшее найденное решение: $global_best_position")
println("Лучший найденный результат: $global_best_score")

return particle_paths, global_best_position, global_best_history # Возвращаем историю для визуализации
end

# Запуск алгоритма PSO с 200 частицами и 1000 итерациями

particle_paths, best_position, global_best_history = pso_3d(200, 1000)

# Визуализация истории глобальных лучших результатов

plot(global_best_history, label="Глобальный лучший результат", xlabel="Итерация", ylabel="Результат", lw=2)
savefig("global_best_history.png")  # Сохраняем график в файл

# Создание анимации движения частиц, для 1000 итераций

anim = @animate for iter in 1:1000
scatter([p[iter][1] for p in particle_paths if length(p) >= iter],  # Координаты X
[p[iter][2] for p in particle_paths if length(p) >= iter],  # Координаты Y
color=:blue, label="")  # Настройки графика
end
gif(anim, "particles.gif", fps=15)  # Сохраняем анимацию в файл