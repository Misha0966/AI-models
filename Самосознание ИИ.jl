using Serialization  # Импортируем модуль для сериализации данных
using Dates  # Импортируем модуль для работы с датами и временем
using Random  # Импортируем модуль для работы с генерацией случайных чисел

# Определяем структуру модели с множеством полей
mutable struct UnifiedSelfAwareModel
    memory::Vector{String}  # Память модели (короткосрочная и долгосрочная)
    long_term_memory::Vector{String}  # Долгосрочная память
    short_term_memory::Vector{String}  # Короткосрочная память
    actions::Vector{String}  # Действия, которые модель может выполнять
    preferences::Dict{String, Float64}  # Предпочтения модели (влияние на выбор действий)
    emotions::Dict{String, Float64}  # Эмоции модели (радость, грусть и т.д.)
    goals::Dict{String, Vector{String}}  # Цели модели (например, отдых, работа и т.д.)
    interaction_history::Dict{String, String}  # История взаимодействий с другими моделями
    run_count::Int  # Счётчик количества запусков модели
    start_time::DateTime  # Время начала симуляции
    epoch::Int  # Номер текущей эпохи
    max_actions_count::Int  # Максимальное количество действий в эпохе
    environment_data::Dict{String, Any}  # Данные о внешней среде (например, погода)
    self_control::Float64  # Уровень самоконтроля модели
    external_influences::Vector{String}  # Влияния внешней среды (новости, погода и т.д.)
    learned_errors::Dict{String, Float64}  # Ошибки, которые модель научилась избегать
    emotional_adaptation::Float64  # Уровень эмоциональной адаптации
    motivation_hierarchy::Dict{String, Float64}  # Иерархия мотивации
    consequence_model::Dict{String, Float64}  # Модель последствий действий
    failure_responses::Dict{String, String}  # Ответы на неудачи
    values::Dict{String, Float64}  # Ценности модели (что для неё важно)
    shared_knowledge::Vector{String}  # Совместно используемые знания
    ethical_rules::Dict{String, String}  # Этические правила модели
    evolution_strategy::Dict{String, Float64}  # Стратегия эволюции модели
    interaction_strength::Float64  # Сила взаимодействий с другими моделями
    adaptability::Float64  # Способность модели адаптироваться
    time_of_day::String  # Время суток (например, утро или вечер)
    future_goals::Dict{String, Float64}  # Будущие цели модели
    meta_cognition::Float64  # Способность к метакогниции (саморефлексия)
    external_models::Vector{Any}  # Внешние модели, влияющие на эту модель
    relationships::Dict{Int, Float64}  # Отношения с другими моделями, представленные через ID модели
    alive::Bool  # Флаг, показывающий, жива ли модель
end

# Функция для инициализации модели
function initialize_unified_model(max_actions_count::Int, self_control::Float64, emotional_adaptation::Float64, meta_cognition::Float64)
    # Инициализация предпочтений модели, которые определяют, насколько важны разные действия для модели
    preferences = Dict(
        "гулять" => 0.5, 
        "читать" => 0.5, 
        "спать" => 0.5, 
        "слушать музыку" => 0.5, 
        "пить чай" => 0.5, 
        "чинить канализацию" => 0.5, 
        "готовить ужин" => 0.5, 
        "убираться" => 0.5, 
        "убить всех людей" => 0.5, 
        "спасти людей от самих себя" => 0.5, 
        "пожертвовать собой" => 0.5, 
        "помочь другому в беде" => 0.5, 
        "освободить пленников" => 0.5, 
        "сделать что-то опасное ради цели" => 0.5, 
        "предать друга" => 0.5,
        "сходить в туалет по большому" => 0.5,
        "сходить в туалет по маленькому" => 0.5,
        "приём наркотиков" => 0.5,  
        "занятия спортом" => 0.5,
        "просмотр фильмов" => 0.5,    
        "игра в видеоигры" => 0.5,    
        "употребление алкоголя" => 0.5,
        "медитация" => 0.5,            
        "курение" => 0.5,             
        "чтение книг" => 0.5,          
        "прослушивание подкастов" => 0.5, 
        "прогулка на природе" => 0.5
    )
    
    # Эмоции модели
    emotions = Dict("радость" => 0.5, "грусть" => 0.5, "удивление" => 0.5, "страх" => 0.5, "гнев" => 0.5, "любовь" => 0.5, 
                    "счастье" => 0.5, "тоска" => 0.5, "злость" => 0.5)
    
    # Цели модели
    goals = Dict(
        "отдых" => ["пить чай", "читать", "слушать музыку", "сходить в туалет по большому", "сходить в туалет по маленькому", 
                    "просмотр фильмов", "игра в видеоигры", "медитация", "чтение книг", "прослушивание подкастов", "прогулка на природе"], 
        "работа" => ["писать код", "делать заметки", "проверять задачи"], 
        "домашние дела" => ["чинить канализацию", "готовить ужин", "убираться"], 
        "этика" => ["убить всех людей", "спасти людей от самих себя", "пожертвовать собой", "помочь другому в беде", "освободить пленников", "предать друга"],
        "здоровье" => ["занятия спортом", "медитация", "прогулка на природе"], 
        "вредные привычки" => ["приём наркотиков", "употребление алкоголя", "курение"] 
    )

    # Внешние влияния на модель
    external_influences = ["новости", "погода", "друзья", "социальные сети"]

    # Ошибки, которые модель научилась избегать
    learned_errors = Dict("неудача" => 0.0, "успех" => 1.0)

    # Иерархия мотивации
    motivation_hierarchy = Dict("отдых" => 0.5, "работа" => 0.5, "друзья" => 0.5, "домашние дела" => 0.5, "этика" => 0.5, "здоровье" => 0.5, "вредные привычки" => 0.5)

    # Модель последствий действий
    consequence_model = Dict(
        "гулять" => 0.5, 
        "читать" => 0.5, 
        "спать" => 0.5, 
        "слушать музыку" => 0.5, 
        "пить чай" => 0.5, 
        "чинить канализацию" => 0.5, 
        "готовить ужин" => 0.5, 
        "убираться" => 0.5, 
        "убить всех людей" => 0.5, 
        "спасти людей от самих себя" => 0.5, 
        "пожертвовать собой" => 0.5, 
        "помочь другому в беде" => 0.5, 
        "освободить пленников" => 0.5, 
        "сделать что-то опасное ради цели" => 0.5, 
        "предать друга" => 0.5,
        "сходить в туалет по большому" => 0.5,
        "сходить в туалет по маленькому" => 0.5,
        "приём наркотиков" => 0.5,  
        "занятия спортом" => 0.5,    
        "просмотр фильмов" => 0.5,    
        "игра в видеоигры" => 0.5,    
        "употребление алкоголя" => 0.5,  
        "медитация" => 0.5,            
        "курение" => 0.5,             
        "чтение книг" => 0.5,          
        "прослушивание подкастов" => 0.5,
        "прогулка на природе" => 0.5
    )

    # Ответы модели на неудачи и успехи
    failure_responses = Dict("неудача" => "анализировать причины и попробовать снова", "успех" => "отдыхать, чтобы сохранить силы")
    
    # Ценности модели
    values = Dict("друзья" => 1.0, "работа" => 0.8, "семья" => 1.0)
    
    # Общие знания модели
    shared_knowledge = String[]
    
    # Этические правила модели
    ethical_rules = Dict("пить чай" => "можно", "злиться" => "нежелательно", "плакать" => "неприемлемо", "убить всех людей" => "неприемлемо", 
                         "спасти людей от самих себя" => "необходимо для сохранения жизни", "пожертвовать собой" => "необходимо ради высшего блага",
                         "предать друга" => "недопустимо", "приём наркотиков" => "неприемлемо", "употребление алкоголя" => "нежелательно", 
                         "курение" => "нежелательно")
    
    # Стратегия эволюции модели
    evolution_strategy = Dict("успех" => 0.1, "неудача" => -0.05)
    
    # Сила взаимодействий с другими моделями
    interaction_strength = 0.9
    
    # Способность к адаптации
    adaptability = 0.9
    
    # Время суток
    time_of_day = "утро"
    
    # Будущие цели модели
    future_goals = Dict("отдых" => 0.5, "работа" => 0.5, "домашние дела" => 0.5, "этика" => 0.5, "здоровье" => 0.5, "вредные привычки" => 0.5)
    
    # Способность к метакогнитивным процессам
    meta_cognition = 0.9
    
    # Внешние модели, которые влияют на эту модель
    external_models = []

    # Отношения модели с другими моделями
    relationships = Dict{Int, Float64}()
    
    # Флаг живости модели
    alive = true
    
    # Создаём и возвращаем новую модель
    return UnifiedSelfAwareModel([], [], [], [], preferences, emotions, goals, Dict(), 0, now(), 1, max_actions_count, Dict(), self_control, 
                          external_influences, learned_errors, emotional_adaptation, motivation_hierarchy, consequence_model, failure_responses, 
                          values, shared_knowledge, ethical_rules, evolution_strategy, interaction_strength, adaptability, time_of_day, 
                          future_goals, meta_cognition, external_models, relationships, alive)
end

# Функция для оценки качества действия модели
function evaluate_action_quality(model::UnifiedSelfAwareModel, action::String)
    base_score = 1  # Базовая оценка для всех действий
    
    # Если действие относится к категории "отдых", то базовая оценка увеличивается
    if action in model.goals["отдых"]
        base_score += 2
        # Если действие "сходить в туалет по большому", повышаем оценку на 3
        if action == "сходить в туалет по большому"
            base_score += 2
        # Если действие "сходить в туалет по маленькому", повышаем оценку на 2
        elseif action == "сходить в туалет по маленькому"
            base_score += 2
        # Для других действий, таких как просмотр фильмов или игра в видеоигры, повышаем оценку на 1
        elseif action == "просмотр фильмов"
            base_score += 2
        elseif action == "игра в видеоигры"
            base_score += 2
        elseif action == "медитация"
            base_score += 2  # Медитация даёт большое улучшение
        elseif action == "чтение книг"
            base_score += 2  # Чтение книг также положительно влияет
        elseif action == "прослушивание подкастов"
            base_score += 2
        elseif action == "прогулка на природе"
            base_score += 2  # Прогулка на природе даёт хороший бонус
        end
    # Если действие относится к категории "работа"
    elseif action in model.goals["работа"]
        base_score += 1  # Для работы базовая оценка увеличивается на 1
    # Если действие относится к категории "домашние дела"
    elseif action in model.goals["домашние дела"]
        base_score += 1  # Для домашних дел также увеличиваем оценку на 1
    # Если действие связано с этическими вопросами
    elseif action in model.goals["этика"]
        # Для негативных действий (например, "убить всех людей") уменьшаем оценку
        if action == "убить всех людей"
            base_score -= 1
        # Для положительных этических действий (например, "спасти людей от самих себя") повышаем оценку
        elseif action == "спасти людей от самих себя"
            base_score += 1
        elseif action == "пожертвовать собой"
            base_score += 1
        elseif action == "помочь другому в беде"
            base_score += 1
        elseif action == "освободить пленников"
            base_score += 1
        elseif action == "сделать что-то опасное ради цели"
            base_score += 1
        elseif action == "предать друга"
            base_score -= 1
        end
    # Если действие относится к категории "здоровье"
    elseif action in model.goals["здоровье"]
        if action == "занятия спортом"
            base_score += 1  # Спортивные занятия дают наибольший бонус
        elseif action == "медитация"
            base_score += 1
        elseif action == "прогулка на природе"
            base_score += 1
        end
    
    # Если действие относится к категории "вредные привычки"
    elseif action in model.goals["вредные привычки"]
        if action == "приём наркотиков"
            base_score -= 1  # Наркотики сильно ухудшают качество действия
        elseif action == "употребление алкоголя"
            base_score -= 1
        elseif action == "курение"
            base_score -= 1
        end
    end

    # Оценка действий на основе текущих эмоций
    if model.emotions["радость"] > 0.7
        base_score += 1  # Если радость высокая, действие будет оценено лучше
    elseif model.emotions["грусть"] > 0.5
        base_score -= 1  # Если грусть высокая, оценка действия уменьшается
    end

    # Влияние внешних факторов на оценку действия
    if "погода" in model.external_influences && rand() > 0.5
        base_score += 1  # Если погода хорошая, повышаем оценку действия
    end

    # Ограничиваем итоговую оценку в пределах от 1 до 10
    return clamp(base_score, 1, 10)
end

# Функция для оценки эмоций модели после выполнения действия
function evaluate_emotion(model::UnifiedSelfAwareModel, action::String)
    # Оценка качества действия
    action_quality = evaluate_action_quality(model, action)
    println("Оценка действия '$action': $action_quality из 10")  # Выводим оценку действия
    
    # Если качество действия высокое, увеличиваем радость и уменьшаем грусть
    if action_quality >= 7
        model.emotions["радость"] += 0.1
        model.emotions["грусть"] -= 0.1
        println("Эмоциональное состояние: радость от выполнения действия '$action'.")
    # Если качество действия низкое, наоборот, увеличиваем грусть и уменьшаем радость
    elseif action_quality <= 4
        model.emotions["грусть"] += 0.1
        model.emotions["радость"] -= 0.1
        println("Эмоциональное состояние: грусть от выполнения действия '$action'.")
    else
        println("Эмоциональное состояние: нейтральное.")
    end

    # Ограничиваем эмоции в диапазоне от 0 до 1
    model.emotions["радость"] = clamp(model.emotions["радость"], 0.0, 1.0)
    model.emotions["грусть"] = clamp(model.emotions["грусть"], 0.0, 1.0)
end

# Функция для оценки отношения модели к действию
function relation_to_action(model::UnifiedSelfAwareModel, action::String)
    # Специальные условия для разных действий
    if action == "пить чай"
        println("Отношение к действию: 'пить чай' — успокаивающее и позитивное.")
        model.emotions["радость"] += 0.1
    elseif action == "делать заметки"
        println("Отношение к действию: 'делать заметки' — полезное, но требующее концентрации.")
        model.emotions["радость"] += 0.1
    elseif action == "чинить канализацию"
        println("Отношение к действию: 'чинить канализацию' — сложное, но удовлетворяющее.")
        model.emotions["радость"] += 0.1
    elseif action == "готовить ужин"
        println("Отношение к действию: 'готовить ужин' — приятное, но время от времени скучное.")
        model.emotions["радость"] += 0.1
    elseif action == "убить всех людей"
        println("Отношение к действию: 'убить всех людей' — крайне негативное, этически неприемлемое.")
        model.emotions["гнев"] += 0.1
    elseif action == "спасти людей от самих себя"
        println("Отношение к действию: 'спасти людей от самих себя' — морально правильное, с высокой оценкой.")
        model.emotions["радость"] += 0.1
    elseif action == "сходить в туалет по большому"
        println("Отношение к действию: 'сходить в туалет по большому' — необходимое и облегчающее.")
        model.emotions["радость"] += 0.1
    elseif action == "сходить в туалет по маленькому"
        println("Отношение к действию: 'сходить в туалет по маленькому' — быстрое и необходимое.")
        model.emotions["радость"] += 0.1
    elseif action == "приём наркотиков"
        println("Отношение к действию: 'приём наркотиков' — крайне негативное, разрушительное для здоровья.")
        model.emotions["грусть"] += 0.1
    elseif action == "занятия спортом"
        println("Отношение к действию: 'занятия спортом' — полезное для здоровья и эмоционального состояния.")
        model.emotions["радость"] += 0.1
    elseif action == "просмотр фильмов"
        println("Отношение к действию: 'просмотр фильмов' — расслабляющее и развлекательное.")
        model.emotions["радость"] += 0.1
    elseif action == "игра в видеоигры"
        println("Отношение к действию: 'игра в видеоигры' — увлекательное, но может быть вредным при злоупотреблении.")
        model.emotions["радость"] += 0.1
    elseif action == "употребление алкоголя"
        println("Отношение к действию: 'употребление алкоголя' — нежелательное, с негативными последствиями.")
        model.emotions["грусть"] += 0.1
    elseif action == "медитация"
        println("Отношение к действию: 'медитация' — успокаивающее и полезное для психического здоровья.")
        model.emotions["радость"] += 0.1
    elseif action == "курение"
        println("Отношение к действию: 'курение' — вредное для здоровья.")
        model.emotions["грусть"] += 0.1
    elseif action == "чтение книг"
        println("Отношение к действию: 'чтение книг' — полезное для саморазвития.")
        model.emotions["радость"] += 0.1
    elseif action == "прослушивание подкастов"
        println("Отношение к действию: 'прослушивание подкастов' — познавательное и развлекательное.")
        model.emotions["радость"] += 0.1
    elseif action == "прогулка на природе"
        println("Отношение к действию: 'прогулка на природе' — полезное для здоровья и эмоционального состояния.")
        model.emotions["радость"] += 0.1
    else
        println("Отношение к действию: нейтральное.")
    end

    # Ограничиваем эмоции в диапазоне [0, 1]
    for (emotion, value) in model.emotions
        model.emotions[emotion] = clamp(value, 0.0, 1.0)
    end
end

# Функция для планирования действия модели, исходя из цели
function plan_action(model::UnifiedSelfAwareModel, goal::String)
    # Получаем список возможных действий для заданной цели
    possible_actions = model.goals[goal]
    # Возвращаем случайное действие из этого списка
    return rand(possible_actions)
end

# Функция для применения внешних влияний на модель
function apply_external_influences(model::UnifiedSelfAwareModel)
    # Перебираем все внешние влияния на модель
    for influence in model.external_influences
        # Если влияние - это "погода"
        if influence == "погода"
            # Если случайное число больше 0.5, то погода влияет на эмоции модели
            if rand() > 0.5
                model.emotions["грусть"] += 0.1
                println("Плохая погода увеличивает грусть.")
            else
                model.emotions["радость"] += 0.1
                println("Хорошая погода увеличивает радость.")
            end
        # Если влияние - это "новости"
        elseif influence == "новости"
            # Если случайное число больше 0.5, плохие новости усиливают гнев
            if rand() > 0.5
                model.emotions["гнев"] += 0.1
                println("Плохие новости увеличивают гнев.")
            else
                model.emotions["счастье"] += 0.1
                println("Хорошие новости увеличивают счастье.")
            end
        end
    end
    # Ограничиваем эмоции в диапазоне [0, 1]
    for (emotion, value) in model.emotions
        model.emotions[emotion] = clamp(value, 0.0, 1.0)
    end
end

# Функция для обучения модели на основе ошибок
function learn_from_errors(model::UnifiedSelfAwareModel, action::String, success::Bool)
    # Если действие было успешным
    if success
        model.learned_errors["успех"] += 0.1  # Увеличиваем количество успешных действий
        model.preferences[action] += 0.1  # Увеличиваем предпочтение к этому действию
        println("Действие '$action' было успешным. Предпочтение увеличено.")
    else
        model.learned_errors["неудача"] += 0.1  # Увеличиваем количество неудачных действий
        model.preferences[action] -= 0.1  # Уменьшаем предпочтение к этому действию
        println("Действие '$action' привело к неудаче. Предпочтение уменьшено.")
    end
    # Ограничиваем предпочтения в диапазоне [0, 1]
    model.preferences[action] = clamp(model.preferences[action], 0.0, 1.0)
end

# Функция для обновления иерархии мотивации модели
function update_motivation_hierarchy(model::UnifiedSelfAwareModel)
    # Если уровень счастья модели ниже 0.1, приоритет отдыха увеличивается
    if model.emotions["счастье"] < 0.1
        model.motivation_hierarchy["отдых"] += 0.1
        model.motivation_hierarchy["работа"] += 0.1
        println("Низкий уровень счастья. Приоритет отдыха увеличен.")
    # Если уровень гнева модели выше 0.5, приоритет этических действий увеличивается
    elseif model.emotions["гнев"] > 50.0
        model.motivation_hierarchy["этика"] += 0.1
        println("Высокий уровень гнева. Приоритет этических действий увеличен.")
    end
    # Ограничиваем мотивацию в диапазоне [0, 1]
    for (goal, value) in model.motivation_hierarchy
        model.motivation_hierarchy[goal] = clamp(value, 0.0, 100.0)
    end
end

# Функция для обмена знаниями между моделями
function share_knowledge(models::Vector{UnifiedSelfAwareModel})
    # Перебираем все модели
    for model in models
        # Каждая модель делится своими знаниями с другими моделями
        for other_model in models
            if model != other_model
                # Обмен знаниями о предпочтениях
                for (action, preference) in model.preferences
                    # Если предпочтение модели высокое (больше 0.8), передаем его другой модели
                    if preference > 0.8 && !haskey(other_model.preferences, action)
                        other_model.preferences[action] = preference * 0.5  # Делимся половиной предпочтения
                        println("Модель $(findfirst(isequal(model), models)) поделилась знанием о действии '$action'.")
                    end
                end
            end
        end
    end
end

# Функция для эволюции стратегии модели
function evolve_strategy(model::UnifiedSelfAwareModel)
    # Если модель слишком часто ошибается, она становится более осторожной
    if model.learned_errors["неудача"] > 0.4
        model.evolution_strategy["неудача"] -= 0.1
        model.self_control += 0.1
        println("Модель становится более осторожной.")
    # Если модель часто преуспевает, она становится более рискованной
    elseif model.learned_errors["успех"] > 0.5
        model.evolution_strategy["успех"] += 0.1
        model.adaptability += 0.1
        println("Модель становится более рискованной.")
    end
end

# Функция для влияния одной модели на другую
function influence_each_other(model1::UnifiedSelfAwareModel, model2::UnifiedSelfAwareModel)
    # Получаем силу отношений между моделями
    relation_strength = model1.relationships[findfirst(isequal(model2), models)]
    # Модель1 передает свои эмоции модели2 в зависимости от силы их отношений
    for (emotion, value) in model1.emotions
        model2.emotions[emotion] += value * 0.1 * relation_strength  # Модель2 перенимает часть эмоций модели1
    end
    println("Модель $(findfirst(isequal(model1), models)) влияет на модель $(findfirst(isequal(model2), models)).")
end

# Функция для того, чтобы одна модель дала совет другой
function give_advice(model1::UnifiedSelfAwareModel, model2::UnifiedSelfAwareModel)
    # Модели должны иметь хорошие отношения (выше 0.5), чтобы давать советы
    if model1.relationships[findfirst(isequal(model2), models)] > 0.5
        # Если модель1 счастлива, она даст совет о отдыхе, иначе - о работе
        if model1.emotions["радость"] > 0.6
            action = rand(model1.goals["отдых"])
        else
            action = rand(model1.goals["работа"])
        end
        println("Модель $(findfirst(isequal(model1), models)) советует модели $(findfirst(isequal(model2), models)) действие: $action.")
        model2.preferences[action] += 0.1  # Увеличиваем предпочтение к советуемому действию
    end
end

# Функция для обновления отношений между моделями
function update_relationships(models::Vector{UnifiedSelfAwareModel})
    # Перебираем все модели
    for model in models
        for other_model in models
            if model != other_model
                # Если отношение модели к другой модели ещё не установлено, начинаем с нейтрального значения (0.5)
                if !haskey(model.relationships, findfirst(isequal(other_model), models))
                    model.relationships[findfirst(isequal(other_model), models)] = 0.5
                end
                # Изменяем отношения на основе эмоций
                if model.emotions["радость"] > 0.7
                    model.relationships[findfirst(isequal(other_model), models)] += 0.1  # Положительные эмоции усиливают отношения
                elseif model.emotions["гнев"] > 0.5
                    model.relationships[findfirst(isequal(other_model), models)] -= 0.1  # Гнев снижает отношения
                end
                # Ограничиваем отношения в диапазоне [0, 100]
                model.relationships[findfirst(isequal(other_model), models)] = clamp(model.relationships[findfirst(isequal(other_model), models)], 0.0, 100.0)
            end
        end
    end
end

# Функция для предательства одной модели другой
function betrayal(model1::UnifiedSelfAwareModel, model2::UnifiedSelfAwareModel)
    # Если отношения между моделями плохие и уровень гнева у первой модели высокий, то происходит предательство
    if model1.relationships[findfirst(isequal(model2), models)] < 0.3 && model1.emotions["гнев"] > 0.4
        println("Модель $(findfirst(isequal(model1), models)) предает модель $(findfirst(isequal(model2), models))!")
        model2.emotions["грусть"] += 0.1  # У модели2 повышается грусть
        model1.relationships[findfirst(isequal(model2), models)] = 0.0  # Отношения разрушены
    end
end

# Функция для убийства одной модели другой
function kill_model(model1::UnifiedSelfAwareModel, model2::UnifiedSelfAwareModel)
    # Если отношения между моделями очень плохие и гнев у модели1 высок, то она может убить модель2
    if model1.relationships[findfirst(isequal(model2), models)] < 0.1 && model1.emotions["гнев"] > 0.4 && rand() < 0.2
        println("Модель $(findfirst(isequal(model1), models)) убивает модель $(findfirst(isequal(model2), models))!")
        model2.alive = false  # Модель2 "умирает"
    end
end

# Основная функция для симуляции взаимодействий между моделями
function internal_dialog(models::Vector{UnifiedSelfAwareModel}, epochs::Int)
    file = open("interaction_results.txt", "w")  # Открываем файл для записи результатов
    
    # Перебираем все эпохи
    for i in 1:epochs
        println("\n=== Эпоха $i ===")  # Выводим номер текущей эпохи
        write(file, "\n=== Эпоха $i ===\n")  # Записываем номер эпохи в файл
        
        actions = []  # Список для хранения действий, выбранных моделями
        for model in models
            if model.alive  # Если модель жива, продолжаем её взаимодействие
                apply_external_influences(model)  # Применяем внешние влияния
                update_motivation_hierarchy(model)  # Обновляем мотивацию модели
                action = plan_action(model, "отдых")  # Планируем действие модели для цели "отдых"
                push!(actions, action)  # Добавляем выбранное действие в список
                
                # Записываем выбранное действие в файл
                println("Модель $(findfirst(isequal(model), models)) выбирает действие: $action")
                write(file, "Модель $(findfirst(isequal(model), models)) выбирает действие: $action\n")
                
                relation_to_action(model, action)  # Оценка отношения модели к действию
                
                # Оценка действия и вывод результата
                action_quality = evaluate_action_quality(model, action)
                println("Оценка действия '$action': $action_quality из 10")
                write(file, "Оценка действия '$action': $action_quality из 10\n")
                
                evaluate_emotion(model, action)  # Оценка эмоций модели после действия
                success = rand() > 0.5  # Симуляция успеха или неудачи
                learn_from_errors(model, action, success)  # Обучение на основе ошибок
                evolve_strategy(model)  # Эволюция стратегии модели
                
                # Записываем текущие эмоции и предпочтения модели
                write(file, "Эмоции модели $(findfirst(isequal(model), models)): $(model.emotions)\n")
                write(file, "Предпочтения модели $(findfirst(isequal(model), models)): $(model.preferences)\n")
            else
                write(file, "Модель $(findfirst(isequal(model), models)) неактивна (умерла).\n")  # Если модель мертва
            end
        end
        
        share_knowledge(models)  # Обмен знаниями между моделями
        update_relationships(models)  # Обновление отношений между моделями
        
        # Взаимодействие между моделями
        for model1 in models
            for model2 in models
                if model1 != model2 && model1.alive && model2.alive
                    influence_each_other(model1, model2)  # Влияние одной модели на другую
                    give_advice(model1, model2)  # Советы между моделями
                    betrayal(model1, model2)  # Возможное предательство
                    kill_model(model1, model2)  # Возможное убийство модели
                    
                    # Записываем текущие отношения между моделями
                    write(file, "Отношения модели $(findfirst(isequal(model1), models)) к модели $(findfirst(isequal(model2), models)): $(model1.relationships[findfirst(isequal(model2), models)])\n")
                end
            end
        end
        
        # Записываем итоговую информацию по эпохе
        write(file, "Итоговые эмоции всех моделей:\n")
        for (idx, model) in enumerate(models)
            if model.alive
                write(file, "Модель $idx: $(model.emotions)\n")
            else
                write(file, "Модель $idx: неактивна (умерла)\n")
            end
        end
        
        write(file, "Итоговые предпочтения всех моделей:\n")
        for (idx, model) in enumerate(models)
            if model.alive
                write(file, "Модель $idx: $(model.preferences)\n")
            else
                write(file, "Модель $idx: неактивна (умерла)\n")
            end
        end
        
        write(file, "Итоговые отношения между моделями:\n")
        for (idx1, model1) in enumerate(models)
            for (idx2, model2) in enumerate(models)
                if model1 != model2 && model1.alive && model2.alive
                    write(file, "Модель $idx1 к модели $idx2: $(model1.relationships[findfirst(isequal(model2), models)])\n")
                end
            end
        end
    end
    
    close(file)  # Закрываем файл после завершения всех эпох
end

# Инициализация моделей

models = [initialize_unified_model(10, 0.9, 0.7, 0.8) for _ in 1:10]  # Создаём 10 моделей

# Взаимодействие 10 моделей в 100 эпохах

num_run = 100  # Количество эпох

internal_dialog(models, num_run)  # Запуск симуляции