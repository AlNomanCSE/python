greeting = lambda message:print(message);
# greeting("I like tea");


fruits = ['apple', 'banana', 'orange']
for fruit in fruits:
    print(f"{fruit}")

for i in range(5):
    print(i)

for i in range(1, 6):
    print(i)
     
for i in range(2,10,5):
    print(i)
    

colors = ['red', 'green', 'blue']

for index,color in enumerate(colors):
    print(f"{index+1}. {color}");

names = ['Alice', 'Bob', 'Charlie'];
ages = [25, 30, 35];
cities = ['NYC', 'LA', 'Chicago'];

for index,(name,age,city) in enumerate(zip(names,ages,cities)):
    print(f"{index}.{name} is {age} and lives in {city}")

numbers1 = [1, 2, 3, 4, 5];
numbers2 = [1, 2, 3, 4, 5];
squqred = list(map(lambda x,y:x**y,numbers1,numbers2));
print(f"{squqred}");

strings = list(map(str,squqred));

print(strings);

numbers = [x for x in range(1,11)];
evens  = list(filter(lambda x:x%2==0,numbers));
print(evens);