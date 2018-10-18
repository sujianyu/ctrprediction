import datetime
import xlearn as xl
xl.hello()
str = "14102800"
print(str[6:8])
print(str[4:6])
year = int(str[0:2])
month = int(str[2:4])
day = int(str[4:6])
dt = datetime.date(year,month,day)
weekday = dt.weekday()
print(weekday)
