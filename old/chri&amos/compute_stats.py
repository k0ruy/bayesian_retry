# Libraries:
from pathlib import Path

import numpy as np
import pandas as pd

from project.config import GAMES_CSV, GAMES_DETAILS_CSV, PLAYERS_CSV, DATA


def main() -> None:
    # Load the datasets from kaggle.com on the NBA:
    # low_memory = False to avoid mixed types warnings.
    games = pd.read_csv(GAMES_CSV, low_memory=False)
    games_details = pd.read_csv(GAMES_DETAILS_CSV, low_memory=False)

    # Merge the games and games_details datasets:
    games = games.merge(games_details, on='GAME_ID', how='left')

    # keep only the games in the 2019-2020 season, since that's the season we're interested in.
    games = games[games['SEASON'] == 2019]

    # get the player statistics we need for our analysis:
    players = games.groupby('PLAYER_ID').agg({'PTS': lambda x: list(x)})

    # clean the missing values from the points' column:
    players['PTS'] = players['PTS'].apply(lambda x: [i for i in x if str(i) != 'nan'])

    # remove the players that didn't play any game
    players = players[players['PTS'].apply(lambda x: len(x) > 0)]

    # compute the mean points scored by all players:
    mean = players['PTS'].apply(lambda x: np.mean(x)).mean()
    # compute the standard deviation of the points scored by all players:
    std = players['PTS'].apply(lambda x: np.std(x)).mean()

    # add the mean and std to the players dataframe:
    df = pd.DataFrame({'mean': [mean], 'std': [std]})

    # save the mean and standard deviation to a csv file:
    df.to_csv(Path(DATA, 'prior_information.csv'), index=False)


if __name__ == '__main__':
    main()
