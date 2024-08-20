from datetime import datetime


__all__ = ['year_month_to_decimal_year', 'date_to_decimal_year']


def year_month_to_decimal_year(year_month: str | int) -> float:
    if isinstance(year_month, int):
        year_month = str(year_month)
    date = datetime.strptime(year_month, '%Y%m').timetuple()
    mon_num = date.tm_mon
    year = date.tm_year
    decimal_year = year + (mon_num - 1) / 12 + 1/24
    return decimal_year


def date_to_decimal_year(year_month_day: str | int) -> float:
    if isinstance(year_month_day, int):
        year_month_day = str(year_month_day)
    if year_month_day[-2:] == '00':
        year_month_day = year_month_day.replace('00', '01')
    date = datetime.strptime(year_month_day, '%Y%m%d').timetuple()
    day_num = date.tm_yday
    year = date.tm_year
    decimal_year = year + (day_num-1)/365.25 + 1/(365.25*2)
    return decimal_year
