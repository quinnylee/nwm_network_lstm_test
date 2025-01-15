import glob

year = input("Give me the year: \n")
month = input('Give me the month:\n')

if int(month) < 10:
    month = str(0) + month
number_of_days = int(len(glob.glob('/media/volume/Imp_Data/FORCING/' + year + '/' + year + month + '*'))/24)

with open('times.txt', 'w') as file:
    for i in range(1, number_of_days + 1):
        if i < 10:
            file.write(year + '/' + year + month + str(0) + str(i) + '*' + '\n')
        else:
            file.write(year + '/' + year + month + str(i) + '*' + '\n')
        