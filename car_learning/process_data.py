import pandas as pd

# load F:\AA-Private\car\all_data\data_good\data.csv

df = pd.read_csv('F:\AA-Private\car\all_data\data_good\data_processed.csv')
print(df.head())

# steering 	forward 	backward 	boost 	recording 	exit 	timestep 	idx 	steer_history 	forward_history

# create list from steering and forward
steering = df['steering'].tolist()
forward = df['forward'].tolist()
# go througt all lines in df and add to each steer_history 10 previus steer fro the list
steer_history = []
forward_history = []

for index, row in df.iterrows():
    if index < 10:
        steer_history.append([0]*10)
        forward_history.append([0]*10)
    steer_history.append(steering[index-10:index])
    forward_history.append(forward[index-10:index])

df['steer_history'] = steer_history
df['forward_history'] = forward_history