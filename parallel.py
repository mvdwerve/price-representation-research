
import time
import subprocess

def run(commands, num=4, refresh=30):
    # the processes
    procs = []

    # iterate over the first num commands
    for cmd in commands[:num]:
        # start a new subprocess
        procs.append(
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        )

    # counters
    completed = 0
    total = len(commands)

    # subdivide the commands
    running = commands[:num]
    commands = commands[num:]

    # we are going to keep waiting
    while len(procs) >= 1:
        # iterate over all open processes
        for i in range(len(procs)):
            # poll will be none if process is not finished yet
            if procs[i].poll() is None:
                continue

            # it is done, YES! remove it
            procs.pop(i)
            running.pop(i)

            # one more completed
            completed += 1

            # leap out
            if len(commands) == 0:
                break

            # and make a new process
            running.append(commands[0])
            procs.append(
                subprocess.Popen(
                    commands[0], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            )

            # remove the command
            commands.pop(0)

            # break out
            break

        # stop if no more processes monitored
        if len(procs) == 0:
            break

        # print the running commands
        print("running", completed, total, float(completed) / total * 100.0, running)

        # going to wait 60 seconds
        time.sleep(refresh)
