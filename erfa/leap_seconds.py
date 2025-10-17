# Licensed under a 3-clause BSD style license - see LICENSE.rst

# `leap_seconds` used to be implemented as a class. Some of the design desicions below
# might seem strange, but they are required to maintain backwards compatibility.

"""Leap second management.

This module allows access to ERFA's leap second table,
using the functions 'get', 'set', and 'update'.

One can also check expiration with two attributes:

    expired
        Whether the leap second table is valid beyond the present.
    expires
        The expiration date of the current ERFA table.

        This is either a date inferred from the last table used to update or
        set the leap-second array, or a number of days beyond the last leap
        second.

Note that usage of the module is similar to a ``ScienceState`` class,
but it cannot be used as a context manager.
"""

__all__ = ["get", "set", "update", "validate"]

from datetime import datetime, timedelta
from warnings import warn

import numpy as np

from .core import ErfaWarning

from .ufunc import get_leap_seconds, set_leap_seconds, dt_eraLEAPSECOND


NUMPY_LT_2_0 = np.__version__.startswith("1.")


def __getattr__(name):
    if name == "expires":
        return _expires_property()
    if name == "expired":
        return _expires_property() < datetime.now()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return ["expired", "expires", "get", "set", "update", "validate"]


_expires = None
"""Explicit expiration date inferred from leap-second table."""
_expiration_days = 180
"""Number of days beyond last leap second at which table expires."""


def get():
    """Get the current leap-second table used internally."""
    return get_leap_seconds()


def validate(table):
    """Validate a leap-second table.

    Parameters
    ----------
    table : array_like
        Must have 'year', 'month', and 'tai_utc' entries.  If a 'day'
        entry is present, it will be checked that it is always 1.
        If ``table`` has an 'expires' attribute, it will be interpreted
        as an expiration date.

    Returns
    -------
    array : `~numpy.ndarray`
        Structures array with 'year', 'month', 'tai_utc'.
    expires: `~datetime.datetime` or None
        Possible expiration date inferred from the table.  `None` if not
        present or if not a `~datetime.datetime` or `~astropy.time.Time`
        instance and not parsable as a 'dd month yyyy' string.

    Raises
    ------
    ValueError
        If the leap seconds in the table are not on the 1st of January or
        July, or if the sorted TAI-UTC do not increase in increments of 1.
    """
    try:
        day = table["day"]
    except Exception:
        day = 1

    expires = getattr(table, "expires", None)
    if expires is not None and not isinstance(expires, datetime):
        # Maybe astropy Time? Cannot go via strftime, since that
        # might need leap-seconds.  If not, try standard string
        # format from leap_seconds.dat and leap_seconds.list
        isot = getattr(expires, "isot", None)
        try:
            if isot is not None:
                expires = datetime.strptime(isot.partition("T")[0],
                                            "%Y-%m-%d")
            else:
                expires = datetime.strptime(expires, "%d %B %Y")

        except Exception as exc:
            warn(f"ignoring non-datetime expiration {expires}; "
                 f"parsing it raised {exc!r}", ErfaWarning)
            expires = None

    # Take care of astropy Table.
    if hasattr(table, "__array__"):
        table = table.__array__()[list(dt_eraLEAPSECOND.names)]

    table = np.array(table, dtype=dt_eraLEAPSECOND, ndmin=1,
                     copy=False if NUMPY_LT_2_0 else None)

    # Simple sanity checks.
    if table.ndim > 1:
        raise ValueError("can only pass in one-dimensional tables.")

    if not np.all(((day == 1) &
                   (table["month"] == 1) | (table["month"] == 7)) |
                  (table["year"] < 1972)):
        raise ValueError("leap seconds inferred that are not on "
                         "1st of January or 1st of July.")

    if np.any((table["year"][:-1] > 1970) &
              (np.diff(table["tai_utc"]) != 1)):
        raise ValueError("jump in TAI-UTC by something else than one.")

    return table, expires


def set(table=None):
    """Set the ERFA leap second table.

    Note that it is generally safer to update the leap-second table than
    to set it directly, since most tables do not have the pre-1970 changes
    in TAI-UTC that are part of the built-in ERFA table.

    Parameters
    ----------
    table : array_like or `None`
        Leap-second table that should at least hold columns of 'year',
        'month', and 'tai_utc'.  Only simple validation is done before it
        is being used, so care need to be taken that entries are correct.
        If `None`, reset the ERFA table to its built-in values.

    Raises
    ------
    ValueError
        If the leap seconds in the table are not on the 1st of January or
        July, or if the sorted TAI-UTC do not increase in increments of 1.
    """
    global _expires
    if table is None:
        expires = None
    else:
        table, expires = validate(table)

    set_leap_seconds(table)
    _expires = expires


def _expires_property():
    """The expiration date of the current ERFA table.

    This is either a date inferred from the last table used to update or
    set the leap-second array, or a number of days beyond the last leap
    second.
    """
    if _expires is None:
        last = get()[-1]
        return (datetime(last["year"], last["month"], 1) +
                timedelta(_expiration_days))
    else:
        return _expires


def update(table):
    """Add any leap seconds not already present to the ERFA table.

    This function matches leap seconds with those present in the ERFA table,
    and extends the latter as necessary.

    If the ERFA leap seconds file was corrupted, it will be reset.

    If the table is corrupted, the ERFA file will be unchanged.

    Parameters
    ----------
    table : array_like or `~astropy.utils.iers.LeapSeconds`
        Array or table with TAI-UTC from leap seconds.  Should have
        'year', 'month', and 'tai_utc' columns.

    Returns
    -------
    n_update : int
        Number of items updated.

    Raises
    ------
    ValueError
        If the leap seconds in the table are not on the 1st of January or
        July, or if the sorted TAI-UTC do not increase in increments of 1.
    """
    global _expires
    table, expires = validate(table)

    # Get erfa table and check it is OK; if not, reset it.
    try:
        erfa_ls, _ = validate(get())
    except Exception:
        set()
        erfa_ls = get()

    # Create the combined array and use it (validating the combination).
    ls = np.union1d(erfa_ls, table)
    set(ls)

    # If the update table has an expiration beyond that inferred from
    # the new leap second second array, use it (but, now that the new
    # array is set, do not allow exceptions due to misformed expires).
    try:
        if expires is not None and expires > _expires_property():
            _expires = expires

    except Exception as exc:
        warn("table 'expires' attribute ignored as comparing it "
             "with a datetime raised an error:\n" + str(exc),
             ErfaWarning)

    return len(ls) - len(erfa_ls)
