using Flux
model = Dense(10, 5, relu)  # Полносвязный слой
data = rand(10)
N = 100
[println(model(data)) for _ in 1:N]  # Однострочный цикл для прогона модели N раз
