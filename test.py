regex_pattern = r"^(\w{3}\.){3}(\w{3})$"	# Do not delete 'r'.

import re
import sys

test_string = '123.123.123.132.123.123'

match = re.match(regex_pattern, test_string) is not None


print(re.match(regex_pattern, test_string))

print(str(match).lower())