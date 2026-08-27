import calendar

def get_month_info(year_month_str):
    '''takes a string link "2020-07 and returns a '2020', '07', '01', '31'
        in other words, the first day of the month and the last day of the month '''
    year, month = map(int, year_month_str.split("-"))
    month_str = f"{month:02d}"               # ensures '01', '09', etc.
    first_day_str = "01"
    last_day_str = f"{calendar.monthrange(year, month)[1]:02d}"
    return str(year), month_str, first_day_str, last_day_str
