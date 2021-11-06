class NoHistoryException(Exception):
    "Tracking was disabled during the last simulation/optimization. Retry after activating tracking."
