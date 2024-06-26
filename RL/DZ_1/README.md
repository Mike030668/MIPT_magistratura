Алгоритм обучения с использованием итерации стохастической политики представляет собой метод обучения с подкреплением, который используется для обучения агента (такси) в среде с дискретным пространством действий и состояний. В данном случае, агент должен научиться эффективно перемещаться по игровому полю, совершая действия, чтобы подбирать пассажиров и доставлять их к целям.

1. **Состояние (State)**:
   Состояние агента определяется его текущим положением на игровом поле, а также расположением пассажиров и целей. Такое состояние можно описать тройкой значений: `(положение такси, пассажиры, цели)`.

2. **Изменение состояний в зависимости от действий агента**:
   - Перемещение на свободную клетку изменяет состояние агента на новое положение такси.
   - Подбор пассажира изменяет состояние агента, включая в себя новое положение такси и обновленный список пассажиров.
   - Достижение цели также изменяет состояние агента, включая в себя новое положение такси и обновленный список целей.
   - Выход за границы поля не изменяет состояние агента.

3. **Вознаграждения (Rewards)**:
   - Подбор пассажира и достижение цели приводят к положительному вознаграждению.
   - Выход за границы поля приводит к отрицательному вознаграждению.
   - Штрафы применяются за повторение действий и за приближение к стенам.

4. **Действия агента (Actions)**:
   Агент может совершать следующие действия:
   - Перемещение на одну клетку вверх.
   - Перемещение на одну клетку вниз.
   - Перемещение на одну клетку влево.
   - Перемещение на одну клетку вправо.
   - Подбор пассажира (если такси находится в клетке с пассажиром).
   - Достижение цели (если такси находится в клетке с целью).

Этот алгоритм позволяет агенту обучаться, принимая решения на основе текущего состояния окружающей среды и полученных вознаграждений, чтобы научиться эффективно перемещаться и доставлять пассажиров к целям.

_________________

## inference_v2
(пассажиры, цель) - смешаны в вознаграждении, поэтому много хаотичных движений между разными целями, но в итоге все-таки развозит по целям

<img src="images/inference_DZ_1.gif" alt="gif"  width="600"/> 

В текущей реализации кода отслеживание того, какой пассажир к какой цели должен быть усовершенствован для более точного моделирования среды и поведения агента. Вместо этого, сейчас алгоритм фокусируется на достижении целей и подборе пассажиров независимо друг от друга. Однако, это можно улучшить, чтобы учесть соответствие между пассажирами и целями.

*Примеры, как это можно учесть:*

1. **Сопоставление по ближайшему расстоянию**: При поиске ближайшей цели для каждого пассажира учитывать расстояние до доступных целей. Это позволит агенту более эффективно выбирать пассажиров и цели, оптимизируя время и расходуемые ресурсы.

2. **Использование алгоритмов сопоставления**: Разработка алгоритма сопоставления пассажиров и целей, который учитывает различные факторы, такие как расстояние, оценка времени доставки и даже предпочтения пассажиров. Это может потребовать более сложных методов, таких как алгоритмы оптимального сопоставления.

3. **Использование маршрутного планирования**: Использование методов маршрутного планирования для оптимального выбора пути такси, учитывая всех пассажиров и их цели. Это позволит агенту принимать решения на основе всей доступной информации о среде.

Улучшение алгоритма для учета соответствия пассажиров и целей может повысить эффективность и производительность агента в выполнении задачи доставки.

_____________________
