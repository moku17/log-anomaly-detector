import pandas as pd
import re
from datetime import datetime

def parse_log_line(line):
    pattern = r'(?P<ip>\S+) - - \[(?P<datetime>[^\]]+)\] "(?P<method>\S+) (?P<url>\S+) \S+" (?P<status>\d{3}) (?P<size>\d+)'
    match = re.match(pattern, line)
    if match:
        data = match.groupdict()
        data['datetime'] = datetime.strptime(data['datetime'], '%d/%b/%Y:%H:%M:%S %z')
        return data
    return None

def load_logs(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = [parse_log_line(line) for line in lines if parse_log_line(line)]
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = load_logs("access.log")
    print(df.head())