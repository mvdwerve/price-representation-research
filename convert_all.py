import parallel
import os

commands = []

for folder in os.listdir("logs/"):
    commands.append('python -O convert.py "%s"' % os.path.join("logs", folder))

# all the commands to run
print("Going to run following commands (%d):" % (len(commands)))
for command in commands:
    print("-", command)


# run all commands in parallel
parallel.run(commands, num=2)
