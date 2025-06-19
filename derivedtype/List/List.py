empty_list = [];
empty_list2= list();
print(type(empty_list2))
sensor_reading = [23.5, 24.1, 22.8, 25.0];

sensor_reading.append(26.200);
sensor_reading.extend([30.2,35.7]);
x = [23.5, 24.1, 22.8, 25.0] + [30.2,35.7];
print(f"here is X :{x}");

sensor_reading.pop();
print(f"{sensor_reading}");

if 23.4 in sensor_reading:
    print("ok");


copy_sensor_reading = sensor_reading.copy();
print(copy_sensor_reading);

numbers = [];
for i in range(1,6):
    numbers.append(i*i);
print(numbers);

squre = [i*i for i in range(1,6)];
print(squre);

words = ['hello', 'world', 'python'];
upperCase_words = [];
for word in words:
    upperCase_words.append(word.upper());

print(f"{upperCase_words}");

upper_words = [word.upper() for word in words];

print(upper_words);

evens = [];
for i in range(10):
    if i%2 ==0:
        evens.append(i);
print(evens);

comprehensivEvens = [i for i in range(10) if i%2==0];

print(comprehensivEvens);

char_list = list("hello");

print(char_list);
even_squres = [x*x for x in range(20) if x%2==0];
print(even_squres);
import random
print(random.choice(words));
random.shuffle(sensor_reading);
print(sensor_reading);


sensor_grid = [
    [23.5, 24.1, 22.8],
    [25.0, 26.2, 24.9],
    [23.1, 24.5, 25.8]
]
sensor_grid.append([23.5, 24.1, 22.8, 25.0]);
print(sensor_grid);
print(sensor_grid.index([25.0, 26.2, 24.9]));
print(x[:3]);