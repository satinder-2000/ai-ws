#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 09:56:23 2026

@author: singh
"""
import QuantLib as ql

print("-----DateGeneration-----\n")

start = ql.Date(7, 5, 2020)
end = ql.Date(15, 8, 2020)

rules = {
    "Backward": ql.DateGeneration.Backward,
    "Forward": ql.DateGeneration.Forward,
    "Zero": ql.DateGeneration.Zero,
    "ThirdWednesnesday": ql.DateGeneration.ThirdWednesday,
    "Twentieth": ql.DateGeneration.Twentieth,
    "TwentiethIMM": ql.DateGeneration.TwentiethIMM,
    "CDS": ql.DateGeneration.CDS
}

for name, rule in rules.items():
    schedule = ql.MakeSchedule(start, end, ql.Period('1m'), rule=rule)
    print(name, [dt for dt in schedule])
    

print("\n-----Date-----\n")
print("---Constructors---\n")

print("ql.Date(serialNumber)\n")
print("ql.Date(44000):",ql.Date(44000) )

print("ql.Date(day, month, year)\n")
print("ql.Date(18, 6, 2020):",ql.Date(18, 6, 2020))
print("\nql.Date(18, ql.June, 2020):",ql.Date(18, ql.June, 2020))

print("\nql.Date(dateString, formatString)")
print("ql.Date('18-06-2020', '%d-%m-%Y'):",ql.Date('18-06-2020', '%d-%m-%Y'))


print("\nMember functions")
today = ql.Date().todaysDate()
print('Original Date:', today)
print('ISO format:', today.ISO())
print('Weekday:', today.weekday())
print('Day of Month:', today.dayOfMonth())
print('Day of Year:', today.dayOfYear())
print('Month:', today.month())
print('Year:', today.year())
print('Serial Number:', today.serialNumber())

print("\nStatic functions")
print('Today :', ql.Date.todaysDate())
print('Min Date :', ql.Date.minDate())
print('Max Date :', ql.Date.maxDate())
print('Is Leap :', ql.Date.isLeap(2011))
print('End of Month :', ql.Date.endOfMonth(ql.Date(4, ql.August, 2009)))
print('Is Month End :', ql.Date.isEndOfMonth(ql.Date(29, ql.September, 2009)))
print('Is Month End :', ql.Date.isEndOfMonth(ql.Date(30, ql.September, 2009)))
print('Next WD :', ql.Date.nextWeekday(ql.Date(1, ql.September, 2009), ql.Friday))
print('n-th WD :', ql.Date.nthWeekday(3, ql.Wednesday, ql.September, 2009))


print("\n-----Period-----\n")
print("ql.Period(n, units)")
print("ql.Period(1, ql.Days):",ql.Period(1, ql.Days))
print("\nql.Period(periodString)")
print("ql.Period('1d')",ql.Period('1d'))
print("\nql.Period(frequency)")
print("ql.Period(ql.Annual):",ql.Period(ql.Annual))

print("\n-----Calendar-----\n")
print("Available Calendars")
calendar1 = ql.UnitedKingdom()
print("calendar1 = ql.UnitedKingdom():",calendar1)

print("\n TARGET > Trans-European Automated Real-time Gross settlement Express Transfer. \n")
calendar2 = ql.TARGET()
print("calendar2 = ql.TARGET():",calendar2)

print("\nSome commonly used member functions:")
cal = ql.TARGET()
mydate = ql.Date(1, ql.May, 2017)

print('Is BD :', cal.isBusinessDay(mydate))
print('Is Holiday :', cal.isHoliday(mydate))
print('Is Weekend :', cal.isWeekend(ql.Friday))
print('Is Last BD :', cal.isEndOfMonth(ql.Date(5, ql.April, 2018)))
print('Last BD :', cal.endOfMonth(mydate))

#print(ql.TARGET().holidayList(ql.Date(1,1,2026),ql.Date(31,12, 2026), False))
print("\nCustom Holiday List\n")
cal = ql.TARGET()

day1 = ql.Date(26, 2, 2020)
day2 = ql.Date(10, 4, 2020)

print('Is Business Day : ', cal.isBusinessDay(day1))
print('Is Business Day : ', cal.isBusinessDay(day2))

cal.addHoliday(day1)
cal.removeHoliday(day2)

print('Is Business Day : ', cal.isBusinessDay(day1))
print('Is Business Day : ', cal.isBusinessDay(day2))

myCalendar = ql.WeekendsOnly()
days = [1,14,15,1,21,26,2,16,15,18,19,9,27,1,19,8,17,25,31]
months =[1,4,4,5,5,6,8,9,9,10,10,11,12,12,12,12]
name =['Año Nuevo','Viernes Santo','Sabado Santo','Dia del Trabajo','Dia de las Glorias Navales','San Pedro y San Pablo','Elecciones Primarias','Dia de la Virgen del Carmen','Asuncion de la Virgen','Independencia Nacional','Glorias del Ejercito','Encuentro de dos mundos','Día de las Iglesias Evangélicas y Protestantes','Día de todos los Santos','Elecciones Presidenciales y Parlamentarias','Inmaculada Concepción','Segunda vuelta Presidenciales','Navidad','Feriado Bancario']
start_year = 2018
n_years = 10

for i in range(n_years + 1):
    for x,y in zip(days, months):
        date = ql.Date(x,y, start_year+1)
        myCalendar.addHoliday(date)
        
print("\nHoliday List")
print("eturns the holidays between two dates.\n")

print("ql.Calendar.holidayList(calendar, from, to, includeWeekEnds=False)")
print(ql.Calendar.holidayList(ql.TARGET(), ql.Date(1,12,2019), ql.Date(31,12,2019)))

print("\nnumber of working days between the dates")
cal = ql.TARGET()

firstDate = ql.Date(31, ql.January, 2018)
secondDate = ql.Date(1, ql.April, 2018)

print('Date 2 Adj :', cal.adjust(secondDate, ql.Preceding))
print('Date 2 Adj :', cal.adjust(secondDate, ql.ModifiedPreceding))

mat = ql.Period(2, ql.Months)

print('Date 1 Month Adv : ',
      cal.advance(firstDate,mat, ql.Following, False))
print('Date 1 Month Adv : ',
      cal.advance(firstDate,mat, ql.ModifiedFollowing, False))

print('Business Days Between: ',
      cal.businessDaysBetween(
          ql.Date(5, ql.March, 2018), ql.Date(30, ql.March, 2018),
          True, True))


print("\njoint Calendar")
print("ql.JointCalendar(calendar1, calendar2, calendar3, calendar4, JointCalendarRule=JoinHolidays)\n")
joint_calander = ql.JointCalendar(ql.TARGET(), ql.Poland())
print("ql.JointCalendar(ql.TARGET(), ql.Poland():\n",joint_calander)


print("\n-----DayCounter-----\n")
dayCounters = {
    'SimpleDayCounter': ql.SimpleDayCounter(),
    'Thirty360':ql.Thirty360(ql.Thirty360.ISDA),
    'Actual360':ql.Actual360(),
    'Actual365Fixed': ql.Actual365Fixed(),
    'Actual365Fixed(Canadian)': ql.Actual365Fixed(ql.Actual365Fixed.Canadian),
    'Actual365FixedNoLeap': ql.Actual365Fixed(ql.Actual365Fixed.NoLeap),
    'ActualActual':ql.ActualActual(ql.ActualActual.ISDA),
    'Business252':ql.Business252()
}
#print("dayCounters:\n",dayCounters)

startDate = ql.Date(15,5,2015)
endDate = ql.Date(15,6,2015)
r = 0.05
nominal = 100e6

for name, dc in dayCounters.items():
    amount = ql.FixedRateCoupon(endDate, nominal, r, dc, startDate, endDate).amount()
    print(name, f"{amount:,.2f}")
    
    
print("\n-----Schedule-----")
print("Schedule(effectiveDate, terminationDate, tenor, calendar, convention, terminationDateConvention, rule, endOfMonth, firstDate=Date(), nextToLastDate=Date())")
effectiveDate = ql.Date(15,6,2020)
terminationDate = ql.Date(15,6,2022)
frequency = ql.Period('6M')
calendar = ql.TARGET()
convention = ql.ModifiedFollowing
terminationDateConvention = ql.ModifiedFollowing
rule = ql.DateGeneration.Backward
endOfMonth = False
schedule = ql.Schedule(effectiveDate, terminationDate, frequency, calendar, convention, terminationDateConvention, rule, endOfMonth)


print("\n-----MakeSchedule-----\n")
print("ql.MakeSchedule(effectiveDate, terminationDate, frequency)")
effectiveDate = ql.Date(15,6,2020)
terminationDate= ql.Date(15,6,2022)
frequency = ql.Period('6M')
schedule = ql.MakeSchedule(effectiveDate, terminationDate, frequency)
#print("schedule: ", schedule)


print("\n-----TimeGrid-----\n")
print("1-year grid with 12 regular steps")
grid1 = ql.TimeGrid(1.0, 12)
print("\nIf there are certain times that need to appear in the TimeGrid, pass them in as a list")
print("ql.TimeGrid(requiredTimes, steps)")
[t for t in ql.TimeGrid([1, 2,5, 4],10)]

grid1 = ql.TimeGrid(1.0, 12)
print("ql.TimeGrid(1.0, 12):\n",grid1)
print("ql.TimeGrid(end, steps)")
t = ql.TimeGrid(10,5)
print("t.dt(4): ",t.dt(4))