
def warning_on_one_line(message, category, filename, lineno, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)