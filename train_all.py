import parallel

loss = [
    "InfoNCE",
    "VAE",
    "BCE-Movement",
    "BCE-Up-Movement",
    "BCE-Anomaly",
    "BCE-Future-Anomaly",
]
bars = ["time", "volume", "dollars"]

commands = []

# we do 1000 epochs each
num_epochs = 1000

for a in bars:
    for b in loss:
        commands.append(
            "python -O train.py --validate --bar_type %s --loss %s --num_epochs %d"
            % (a, b, num_epochs)
        )

# all the commands to run
print("Going to run following commands (%d):" % (len(commands)))
for command in commands:
    print("-", command)

# run all commands
parallel.run(commands)
