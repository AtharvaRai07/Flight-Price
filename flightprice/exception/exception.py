import sys
from networksecurity.logging import logger

class FlightException(Exception):
    """Base exception class for the Flight application."""
    def __init__(self, error_message, error_detail: sys):
        self.error_message = error_message
        _,_,exe_tb = error_detail.exc_info()

        self.lineno = exe_tb.tb_lineno
        self.filename = exe_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return "Error occurred in script: {0} at line number: {1} with message: {2}".format(
            self.filename, self.lineno, self.error_message
        )
