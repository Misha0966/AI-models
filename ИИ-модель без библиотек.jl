# Гиперпараметры
learning_rate = 0.1  # Устанавливаем скорость обучения модели, которая контролирует, насколько сильно изменяются веса на каждом шаге
epochs = 1000  # Количество итераций (эпох), за которые модель будет обучаться
lambda = 1000000  # Коэффициент регуляризации, который помогает избежать переобучения, уменьшая влияние слишком больших весов

# Генерация данных
function generate_data(samples)
    X = hcat(ones(samples), rand(samples, 2))  # Генерируем случайные входные данные, добавляя колонку из единиц (для смещения)
    y = [x[2] + x[3] > 1.0 ? 1 : 0 for x in eachrow(X)]  # Генерируем метки классов: если сумма 2-го и 3-го столбца > 1, то метка 1, иначе 0
    return (X, y)  # Возвращаем данные и метки
end

# Сигмоида
sigmoid(z) = 1 / (1 + exp(-z))  # Функция активации, которая возвращает значение от 0 до 1, подходящее для вероятности

# Модифицированная функция стоимости с L2-регуляризацией
function cost_function(h, y, weights)
    m = length(y)  # Число обучающих примеров
    data_loss = (-y'log.(h) - (1 .- y)'log.(1 .- h)) / m  # Вычисление потерь (ошибки) на основе логистической регрессии
    reg_loss = (lambda/(2*m)) * sum(weights[2:end].^2)  # Регуляризация L2 для весов (кроме смещения)
    data_loss + reg_loss  # Суммируем потери и регуляризацию
end

# Расширенная функция обучения
function train(X, y)
    m, n = size(X)  # Размерность данных: m - количество примеров, n - количество признаков
    weights = zeros(n)  # Инициализируем веса (включая смещение) нулями
    loss_history = []  # История потерь для последующего анализа

    for epoch in 1:epochs  # Для каждой эпохи
        # Прямое распространение (вычисление предсказания)
        z = X * weights  # Линейная комбинация входных данных и весов
        h = sigmoid.(z)  # Применяем сигмоиду для получения вероятности

        # Градиент с регуляризацией
        gradient = (X' * (h - y) / m) + (lambda/m)*[0; weights[2:end]]  # Градиент с добавлением регуляризации для всех весов, кроме смещения

        # Обновление весов с использованием градиентного спуска
        weights -= learning_rate * gradient  # Обновляем веса с шагом, пропорциональным градиенту

        # Сохранение истории потерь для последующего анализа
        loss = cost_function(h, y, weights)  # Вычисляем текущие потери
        push!(loss_history, loss)  # Добавляем текущие потери в историю

        # Визуализация прогресса
        if epoch % 10 == 0  # Каждые 10 эпох выводим информацию
            println("Эпоха: $epoch \t Потери: $(round(loss, digits=4))")  # Печатаем номер эпохи и текущие потери
            progress = "|" * repeat("█", round(Int, 50*(epoch/epochs))) * repeat(" ", 50 - round(Int, 50*(epoch/epochs))) * "|"  # Прогрессбар
            println(progress)  # Выводим прогрессбар
        end
    end
    return weights, loss_history  # Возвращаем финальные веса и историю потерь
end

# Расширенная функция предсказания с порогом
function predict(X, weights; threshold=0.5)
    prob = sigmoid.(X * weights)  # Вычисляем вероятность для каждого примера
    [p > threshold ? 1 : 0 for p in prob]  # Если вероятность больше порога, то класс 1, иначе 0
end

# Новые метрики качества
function evaluate_model(y_true, y_pred)
    TP = sum((y_true .== 1) .& (y_pred .== 1))  # Истинно положительные (True Positives)
    TN = sum((y_true .== 0) .& (y_pred .== 0))  # Истинно отрицательные (True Negatives)
    FP = sum((y_true .== 0) .& (y_pred .== 1))  # Ложно положительные (False Positives)
    FN = sum((y_true .== 1) .& (y_pred .== 0))  # Ложно отрицательные (False Negatives)
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)  # Точность (доля правильных предсказаний)
    precision = TP / (TP + FP + eps())  # Прецизионность (доля правильно предсказанных положительных)
    recall = TP / (TP + FN + eps())  # Полнота (доля реально положительных, которые были предсказаны как положительные)
    f1 = 2 * (precision * recall) / (precision + recall + eps())  # F1-мера (среднее гармоническое precision и recall)
    
    (accuracy=accuracy, precision=precision, recall=recall, f1=f1)  # Возвращаем все метрики в виде словаря
end

# Функция для сохранения результатов в файл
function save_results(weights, history, metrics)
    # Сохраняем веса модели
    open("model_weights.txt", "w") do file
        println(file, "Weights:")  # Заголовок
        for w in weights  # Записываем каждый вес в файл
            println(file, w)
        end
    end
    
    # Сохраняем историю потерь
    open("loss_history.txt", "w") do file
        println(file, "Loss History:")  # Заголовок
        for loss in history  # Записываем каждое значение потерь
            println(file, loss)
        end
    end
    
    # Сохраняем метрики
    open("metrics.txt", "w") do file
        println(file, "Model Evaluation Metrics:")  # Заголовок
        println(file, "Accuracy: $(round(metrics.accuracy, digits=2))")  # Точность
        println(file, "Precision: $(round(metrics.precision, digits=2))")  # Прецизионность
        println(file, "Recall: $(round(metrics.recall, digits=2))")  # Полнота
        println(file, "F1-Score: $(round(metrics.f1, digits=2))")  # F1-мера
    end
end

# Пример использования
X, y = generate_data(1000000)  # Генерация 1 миллиона данных
weights, history = train(X, y)  # Обучение модели на данных
y_pred = predict(X, weights)  # Прогнозирование на тех же данных
metrics = evaluate_model(y, y_pred)  # Оценка качества модели

println("\nОценка модели:")  # Заголовок
println("Точность: $(round(metrics.accuracy, digits=2))")  # Выводим точность
println("Прецизионность: $(round(metrics.precision, digits=2))")  # Прецизионность
println("Полнота: $(round(metrics.recall, digits=2))")  # Полнота
println("F1-мера: $(round(metrics.f1, digits=2))")  # F1-мера

# Простая визуализация кривой обучения
println("\nКривая обучения:")  # Заголовок
for (i, loss) in enumerate(history)  # Перебор истории потерь
    if i % 100 == 0  # Каждые 100 эпох выводим график
        println("Эпоха $(lpad(i, 4)): " * repeat("◼", round(Int, 30*(loss/history[1]))))  # Печатаем график потерь
    end
end

# Сохранение результатов в файлы
save_results(weights, history, metrics)  # Сохраняем результаты в файлы