f = open("personality_datasert.csv", "r")

data = []
target = []

file = f.readlines()
temp =[]
for i in range(1, len(file)):
    temp.append(file[i].split(","))

for line in temp:
    
    line = [float(line[0]), int(line[1]), float(line[2]), float(line[3]), int(line[4]), float(line[5]), float(line[6]), int(line[7])]
    target.append(line[7])
    line.pop(7)
    data.append(line)