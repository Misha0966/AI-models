using LinearAlgebra  # Подключаем модуль для работы с линейной алгеброй
using Flux  # Подключаем библиотеку для создания и обучения нейронных сетей
using Flux: Conv, relu  # Импортируем функции свертки (Conv) и активации (relu) из Flux
using DataLoaders  # Подключаем модуль для работы с загрузкой данных

# Функция для 3D роторного преобразования данных
function rotor_transform_3d(data::AbstractMatrix, angle::Real)
rotation_matrix = [cos(angle) -sin(angle) 0;  # Матрица вращения: элемент 1 строки
sin(angle) cos(angle) 0;  # Матрица вращения: элемент 2 строки
0 0 1]  # Матрица вращения: элемент 3 строки, вращение вокруг оси Z
return rotation_matrix * data  # Применяем вращение к входным данным
end

# Генерация данных в 3D
function generate_data_3d(samples::Int, angle::Real)
data = rand(3, samples)  # Создаем случайную матрицу данных размером 3xN (координаты x, y, z)
transformed_data = rotor_transform_3d(data, angle)  # Преобразуем данные, применяя роторное вращение
return transformed_data, data  # Возвращаем преобразованные данные и исходные данные как цель
end

# Фрактальный блок
function fractal_block(input_dim::Int, output_dim::Int)
return Chain(
Dense(input_dim, output_dim, relu),  # Первый слой: плотный, с функцией активации ReLU
Dense(output_dim, output_dim, relu),  # Второй слой: сохраняем размер выходных данных
Dense(output_dim, output_dim, relu),  # Третий слой: добавляем нелинейность
Dense(output_dim, output_dim, relu)  # Четвертый слой: дополнительная сложность
)
end

# Модификация обучения с использованием DataLoaders
function train_rotor_model(samples::Int, angle::Real, epochs::Int, learning_rate::Real, batch_size::Int)
x, y = generate_data_3d(samples, angle)  # Генерируем данные и цели для обучения

data = [(x[:, i:min(i + batch_size - 1, end)], y[:, i:min(i + batch_size - 1, end)]) 
for i in 1:batch_size:samples]  # Разделяем данные на батчи

model = Chain(  # Определяем архитектуру модели
Dense(3, 64, relu),  # Входной слой: из 3D в 64 признака
fractal_block(64, 128),  # Фрактальный блок: повышаем сложность
Dense(128, 64, relu),  # Слой уменьшения размерности до 64
Dense(64, 3)  # Выходной слой: возвращаем к 3D координатам
)

loss(x, y) = Flux.mse(model(x), y)  # Определяем функцию ошибки: среднеквадратичное отклонение

opt = Adam(learning_rate)  # Настраиваем оптимизатор Adam с заданной скоростью обучения

println("Начало обучения...")  # Выводим сообщение о начале обучения
for epoch in 1:epochs  # Основной цикл по эпохам
for (x_batch, y_batch) in data  # Итерируем по батчам данных
Flux.train!(loss, Flux.params(model), [(x_batch, y_batch)], opt)  # Выполняем один шаг обучения
end
if epoch % 1 == 0  # Каждую эпоху выводим ошибку
println("Эпоха $epoch: ошибка $(loss(x, y))")  # Показываем текущую ошибку
end
end
println("Обучение завершено!")  # Обучение завершено
return model  # Возвращаем обученную модель
end

samples = 10000  # Количество примеров для генерации данных
angle = π / 4  # Угол вращения в радианах (45 градусов)
epochs = 50  # Количество эпох обучения
learning_rate = 0.000001  # Скорость обучения для оптимизатора
batch_size = 32  # Размер батча для обучения

model = train_rotor_model(samples, angle, epochs, learning_rate, batch_size)  # Запускаем обучение модели

println("Проверка модели на новых данных:")  # Выводим сообщение перед проверкой
x_test, y_test = generate_data_3d(1, angle)  # Генерируем тестовые данные
predictions = model(x_test)  # Получаем предсказания модели на тестовых данных
println("Оригинальные данные:\n$y_test")  # Печатаем исходные данные
println("Предсказания модели:\n$predictions")  # Печатаем предсказания