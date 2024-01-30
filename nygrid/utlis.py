"""
Functions that support other NYGrid modules.

Known Issues/Wishlist:

"""
import datetime
import os
import pandas as pd
import string


def format_date(in_date):
    """
    Formats an input date so that it can be correctly written to the namelist.

    Parameters
    ----------
    in_date : str
        Date to be formatted.

    Returns
    -------
    out_date : datetime.datetime
        Formatted date.
    """
    for fmt in ('%b %d %Y', '%B %d %Y', '%b %d, %Y', '%B %d, %Y',
                '%m-%d-%Y', '%m.%d.%Y', '%m/%d/%Y',
                '%Y-%m-%d', '%Y.%m.%d', '%Y/%m/%d',
                '%b %d %Y %H', '%B %d %Y %H', '%b %d, %Y %H', '%B %d, %Y %H',
                '%m-%d-%Y %H', '%m.%d.%Y %H', '%m/%d/%Y %H'):
        try:
            return datetime.datetime.strptime(in_date, fmt)
        except ValueError:
            pass
    raise ValueError('No valid date format found; please use a common US format (e.g., Jan 01, 2011 00)')
