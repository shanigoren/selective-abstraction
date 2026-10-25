"""
Time utility functions for the selective abstraction pipeline.
"""

from datetime import datetime


def get_timestamp():
    """Get current time formatted as HH:MM:SS"""
    return datetime.now().strftime("%H:%M:%S")


def format_duration(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"