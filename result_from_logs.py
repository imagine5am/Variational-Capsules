with open('./logs/logs_7') as file:
    for line in file:
        if line.startswith('Train loss: '):
            print(line)