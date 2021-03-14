with open('./logs/logs_7') as file:
    for line in file:
        if line.startswith('Train loss: '):
            print(line, sep='')
            
        elif line.startswith('Epoch '):
            print(line, sep='')
        
        elif line.startswith('len('):
            pass