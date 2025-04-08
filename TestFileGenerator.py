from datetime import date
import random
from random import randint, choice
import sys
import time
import faker
from datetime import datetime
import os
os.environ['TZ'] = 'Asia/Kolkata'
fak = faker.Faker()

class LogGenerator:
    def __init__(self):
        self.dictionary = {
            'request': ['GET', 'POST', 'PUT', 'DELETE'],
            'endpoint': ['/usr', '/usr/admin', '/usr/admin/developer', '/usr/login', '/usr/register'],
            'statuscode': ['303', '404', '500', '403', '502', '304', '200'],
            'username': ['james', 'adam', 'eve', 'alex', 'smith', 'isabella', 'david', 'angela', 'donald', 'hilary'],
            'ua': [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:84.0) Gecko/20100101 Firefox/84.0',
                'Mozilla/5.0 (Android 10; Mobile; rv:84.0) Gecko/84.0 Firefox/84.0',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36',
                'Mozilla/5.0 (Linux; Android 10; ONEPLUS A6000) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Mobile Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4380.0 Safari/537.36 Edg/89.0.759.0',
                'Mozilla/5.0 (Linux; Android 10; ONEPLUS A6000) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.116 Mobile Safari/537.36 EdgA/45.12.4.5121',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36 OPR/73.0.3856.329',
                'Mozilla/5.0 (Linux; Android 10; ONEPLUS A6000) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Mobile Safari/537.36 OPR/61.2.3076.56749',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/7046A194A',
                'Mozilla/5.0 (iPhone; CPU iPhone OS 12_4_9 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.1.2 Mobile/15E148 Safari/604.1'
            ],
            'referrer': ['-', fak.uri()]
        }

    def str_time_prop(self, start, end, format, prop):
        stime = time.mktime(time.strptime(start, format))
        etime = time.mktime(time.strptime(end, format))
        ptime = stime + prop * (etime - stime)
        return time.strftime(format, time.localtime(ptime))

    def random_date(self, start, end, prop):
        return self.str_time_prop(start, end, '%d/%b/%Y:%I:%M:%S %z', prop)

    def generate_log_entry(self):
        """Generate a single log entry"""
        return '%s - - [%s] "%s %s HTTP/1.0" %s %s "%s" "%s" %s\n' % (
            fak.ipv4(),
            self.random_date("01/Jan/2018:12:00:00 +0530", "01/Jan/2020:12:00:00 +0530", random.random()),
            choice(self.dictionary['request']),
            choice(self.dictionary['endpoint']),
            choice(self.dictionary['statuscode']),
            str(int(random.gauss(5000, 50))),
            choice(self.dictionary['referrer']),
            choice(self.dictionary['ua']),
            random.randint(1, 5000)
        )

    def generate_log_file(self, output_file, num_entries=1000000):
        """Generate a log file with specified number of entries"""
        with open(output_file, "w") as f:
            for _ in range(1, num_entries + 1):
                f.write(self.generate_log_entry())

def main():
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
        num_entries = int(sys.argv[2]) if len(sys.argv) > 2 else 1000000
    else:
        output_file = "logfiles.log"
        num_entries = 1000000

    generator = LogGenerator()
    print(f"Generating {num_entries} log entries to {output_file}...")
    generator.generate_log_file(output_file, num_entries)
    print("Log generation complete!")

if __name__ == "__main__":
    main()