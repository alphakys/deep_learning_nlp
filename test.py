
import re

N = input()
regex_pattern = r"<(\w+)"
test_string = ''

for i in range(int(N)):
    test_string += input()

match = re.findall(regex_pattern, test_string)
s = set()
for i in match:
    s.add(i)

s = sorted(s)
print(';'.join(s))


exit()
test_string = '''<p><a href="http://www.quackit.com/html/tutorial/html_links.cfm">Example Link</a></p>
<div class="more-info"><a href="http://www.quackit.com/html/examples/html_links_examples.cfm">More Link Examples...</a></div>'''

match = re.findall(regex_pattern, test_string)



print(s)


#print(str(match).lower())