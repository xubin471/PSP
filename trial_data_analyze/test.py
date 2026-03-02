import re

content = "'Liver', abdladgja; jagdag1 45455"
pattern = fr"'[(A-Z,a-z)+]'"
matches = re.findall(pattern, content)