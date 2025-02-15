using LinearAlgebra  # Импортируем библиотеку для работы с линейной алгеброй
using Random  # Импортируем библиотеку для генерации случайных чисел

# Простая модель нейросети с весами и смещениями
mutable struct SimpleModel
    weights::Matrix{Float64}  # Матрица весов (элементы, умножаемые на входные данные)
    bias::Vector{Float64}    # Вектор смещений (прибавляется к результату)
end

# Функция прямого прохода модели (применение модели к данным)
function (model::SimpleModel)(x::Matrix{Float64})
    return model.weights * x .+ model.bias  # Умножаем входные данные на веса и добавляем смещения
end

# Функция обновления параметров модели (градиентный спуск)
function update_model!(model::SimpleModel, data::Dict{Symbol, Matrix{Float64}}, learning_rate::Float64)
    # Прямой проход: делаем предсказания модели на основе входных данных
    predictions = model(data[:input])

    # Вычисляем ошибку между предсказаниями и целевыми значениями
    error = predictions - data[:target]

    # Обновляем параметры модели (веса и смещения) вручную с учётом ошибки
    model.weights -= learning_rate * error * data[:input]'  # Обновляем веса
    model.bias -= learning_rate * sum(error; dims=2)[:]  # Обновляем смещения
end

# Функция для 3D вращения данных (преобразование координат в 3D пространстве)
function rotor_transform_3d(data::AbstractMatrix, angle::Real)
    # Создаём матрицу вращения для вращения вокруг оси Z на заданный угол
    rotation_matrix = [cos(angle) -sin(angle) 0;  
                       sin(angle) cos(angle)  0;  
                       0          0           1]
    return rotation_matrix * data  # Применяем матрицу вращения к данным
end

# Функция для генерации случайных данных в 3D пространстве
function generate_data_3d(samples::Int, angle::Real)
    data = rand(3, samples)  # Генерируем случайные 3D точки
    transformed_data = rotor_transform_3d(data, angle)  # Применяем вращение к данным
    return transformed_data, data  # Возвращаем преобразованные и исходные данные
end

# Функция для тренировки модели с использованием сгенерированных данных
function train_rotor_model(samples::Int, angle::Real, epochs::Int, learning_rate::Real, batch_size::Int)
    # Генерация обучающих данных
    x, y = generate_data_3d(samples, angle)

    # Инициализация модели с случайными весами и смещениями
    input_dim, output_dim = size(x, 1), size(y, 1)
    model = SimpleModel(randn(output_dim, input_dim), randn(output_dim))  # Веса и смещения

    # Формируем батчи данных для эффективного обучения
    batches = [Dict(:input => x[:, i:min(i + batch_size - 1, end)],
                    :target => y[:, i:min(i + batch_size - 1, end)])
               for i in 1:batch_size:samples]

    println("Начало обучения...")  # Печатаем, что обучение начинается
    for epoch in 1:epochs  # Повторяем обучение для указанного числа эпох
        for batch in batches  # Для каждого батча данных
            update_model!(model, batch, learning_rate)  # Обновляем параметры модели
        end

        if epoch % 1 == 0  # Каждую эпоху выводим информацию о текущей ошибке
            predictions = model(x)  # Получаем предсказания модели для обучающих данных
            loss = sum((predictions - y).^2) / size(y, 2)  # Вычисляем ошибку (MSE)
            println("Эпоха $epoch: ошибка $loss")  # Печатаем ошибку на текущей эпохе
        end
    end

    println("Обучение завершено!")  # Печатаем сообщение о завершении обучения
    return model  # Возвращаем обученную модель
end

# Устанавливаем параметры обучения
samples = 10000  # Количество примеров для обучения
angle = π / 4    # Угол вращения (45 градусов)
epochs = 50      # Количество эпох обучения
learning_rate = 0.0001  # Скорость обучения (насколько сильно изменяются параметры)
batch_size = 32  # Размер батча (количество примеров в одном шаге)

# Обучаем модель
model = train_rotor_model(samples, angle, epochs, learning_rate, batch_size)

# Проверка модели на новых данных
println("Проверка модели на новых данных:")  
x_test, y_test = generate_data_3d(10, angle)  # Генерируем новые данные для теста
predictions = model(x_test)  # Получаем предсказания модели для новых данных
println("Оригинальные данные:\n$y_test")  # Печатаем оригинальные данные
println("Предсказания модели:\n$predictions")  # Печатаем предсказания модели