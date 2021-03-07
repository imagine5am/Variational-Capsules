with open('./logs/logs_7') as file:
    for line in file:
        if line.startswith('Train loss: ') or line.startswith('len('):
            print(line, sep='')