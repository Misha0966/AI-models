using Flux # Подключаем библиотеку Flux для работы с нейронными сетями
using Random # Подключаем библиотеку Random для генерации случайных чисел
using Statistics # Подключаем библиотеку Statistics для вычисления статистических характеристик
using StatsBase # Подключаем библиотеку StatsBase для дополнительных статистических функций


# Определяем структуру мета-модели, которая будет выбирать параметры для генеративной модели
struct MetaModel
    latent_dims::Vector{Int}    # Список возможных размеров скрытого пространства (размер входного шума для GAN)
    learning_rates::Vector{Float64} # Список возможных скоростей обучения (параметр, влияющий на скорость обучения моделей)
end

function suggest_hyperparameters(meta::MetaModel) # Функция для выбора случайных гиперпараметров из мета-модели
    latent_dim = rand(meta.latent_dims)   # Случайным образом выбираем размер скрытого пространства
    learning_rate = rand(meta.learning_rates) # Случайным образом выбираем скорость обучения
    return latent_dim, learning_rate     # Возвращаем выбранные параметры
end

function build_gan(latent_dim, learning_rate) # Функция для создания GAN (генеративно-состязательной сети) с заданными параметрами
# Определяем генератор — сеть, которая будет создавать поддельные данные

generator = Chain(
        Dense(latent_dim, 32, relu),  # Первый слой: преобразует шум в скрытое представление
        Dense(32, 2) # Второй слой: преобразует скрытое представление в данные размером 2
    )
# Определяем дискриминатор — сеть, которая будет отличать реальные данные от поддельных

    discriminator = Chain(
        Dense(2, 32, relu), # Первый слой: обрабатывает входные данные
        Dense(32, 1), # Второй слой: предсказывает, настоящие данные или нет
        sigmoid # Преобразует результат в вероятность (от 0 до 1)
    )

    opt_gen = Adam(learning_rate) # Создаем оптимизатор для генератора
    opt_disc = Adam(learning_rate) # Создаем оптимизатор для дискриминатора

    return generator, discriminator, opt_gen, opt_disc # Возвращаем созданные модели и оптимизаторы
end

function evaluate_gan(generator, discriminator, latent_dim) # Функция для оценки качества работы GAN

    batch_size = 100  # Размер партии данных (сколько данных анализируется за один раз)
    noise = randn(latent_dim, batch_size)  # Генерируем случайный шум для генератора
    fake_data = generator(noise)          # Генератор создает поддельные данные на основе шума
    real_data = randn(2, batch_size)      # Создаем реальные данные (случайные числа)
    disc_real = mean(discriminator(real_data))  # Дискриминатор оценивает реальные данные (среднее значение)
    disc_fake = mean(discriminator(fake_data))  # Дискриминатор оценивает поддельные данные (среднее значение)
    return abs(disc_real - 1.0) + abs(disc_fake)  # Возвращаем разницу (чем меньше, тем лучше модель)
end

function run_hyperparameter_optimization(meta_model, iterations) # Функция для поиска лучших гиперпараметров с помощью перебора
    best_score = Inf  # Устанавливаем начальное значение лучшей оценки (бесконечность)
    best_params = nothing  # Переменная для хранения лучших параметров

    for iteration in 1:iterations
        println("Итерация $iteration...")  # Выводим номер текущей итерации
        latent_dim, learning_rate = suggest_hyperparameters(meta_model) # Выбираем случайные гиперпараметры

        generator, discriminator, opt_gen, opt_disc = build_gan(latent_dim, learning_rate) # Создаем GAN с выбранными параметрами

        score = evaluate_gan(generator, discriminator, latent_dim) # Оцениваем качество GAN

        println("Гиперпараметры: latent_dim=$latent_dim, learning_rate=$learning_rate, оценка=$score")         # Выводим результаты текущей итерации

        # Сохраняем параметры, если результат оказался лучше предыдущих
        if score < best_score
            best_score = score
            best_params = (latent_dim, learning_rate)
        end
    end

    return best_params, best_score  # Возвращаем лучшие параметры и их оценку
end

meta_model = MetaModel([10, 20, 50], [0.001, 0.0005, 0.0001]) # Создаем мета-модель с возможными значениями гиперпараметров

best_params, best_score = run_hyperparameter_optimization(meta_model, 20) # Запускаем поиск лучших гиперпараметров на 20 итераций

println("Лучшая конфигурация: ", best_params, " с оценкой ", best_score) # Выводим лучшую конфигурацию параметров и их оценку
