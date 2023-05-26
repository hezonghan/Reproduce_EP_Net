

import time
from datetime import date


def get_date_time_str():
    td = date.today()
    t = time.strftime('%H%M%S', time.localtime())
    return '{}_{:02d}{:02d}_{}'.format(td.year, td.month, td.day, t)

