a = list()
for i in range(100):
    a.append(i)
b = [i % 10 for i in a]
print(b)
