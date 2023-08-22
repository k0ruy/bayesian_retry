# Libraries:
from pathlib import Path
import numpy as np
import pandas as pd
from project.config import PLAYERS_CSV, GAMES_DETAILS_CSV, GAMES_CSV, DATA, TEAMS_CSV, RANKING_CSV
pd.set_option('display.max_columns', None)

# printing options:
np.set_printoptions(suppress=True)


def main() -> None:
    # Load the datasets from kaggle.com on the NBA:
    # low_memory = False to avoid mixed types warnings.
    games = pd.read_csv(GAMES_CSV, low_memory=False)
    games_details = pd.read_csv(GAMES_DETAILS_CSV, low_memory=False)
    players = pd.read_csv(PLAYERS_CSV, low_memory=False)

    # Merge the games and games_details datasets:
    games = games.merge(games_details, on='GAME_ID', how='left')
    # remove the games after the 2019-2020 season
    games = games[games['SEASON'] == 2019]

    # Get the players active in the 2019-2020 season
    players_2019_2020_ids = players[players['SEASON'] == 2019]['PLAYER_ID'].values

    # keep only the games played by the players active in the 2019-2020 season
    games = games[games['PLAYER_ID'].isin(players_2019_2020_ids)]

    # get the player statistics we need for our analysis:
    players = games.groupby('PLAYER_ID').agg({'GAME_ID': 'count', 'PLAYER_NAME': 'first',
                                              'PTS': lambda x: list(x)})

    # rename GAME_ID to GAMES_PLAYED:
    players.rename(columns={'GAME_ID': 'GAMES_PLAYED'}, inplace=True)

    # clean the missing values from the points' column:
    players['PTS'] = players['PTS'].apply(lambda x: [i for i in x if str(i) != 'nan'])
    # remove the players that didn't play any game
    players = players[players['PTS'].apply(lambda x: len(x) > 0)]

    # To simplify the computations, we select only 1 player per role, to make it interesting we picked
    # some of the best players in the league for the 2019-2020 season:
    # 1. Point Guard: Jamal Murray
    # 2. Center: Nikola Jokic
    # 3. Shooting Guard: James Harden
    # 4. Small Forward: LeBron James
    # 5. Power Forward: Anthony Davis
    # 6. Wayne Ellington, selected as a problematic player with too few games played to
    #    use an unpooled model.

    # get the ids of the players of interest:
    players_of_interest = players[players['PLAYER_NAME'].isin(['Jamal Murray', 'Nikola Jokic',
                                                                'James Harden', 'LeBron James',
                                                                'Anthony Davis'])]

    players_of_interest.reset_index(inplace=True)

    # compute the mean of the points scored by each player
    points_of_interest = players_of_interest['PTS'].loc[0:79].copy()
    mean_points = points_of_interest.apply(lambda x: np.array(x).mean())
    # compute the mean of all players for centering the data
    mean = mean_points.mean()

    # create the dataset we will work on:
    dataset = np.zeros([5, 84], dtype=np.ndarray)
    for index, row in players_of_interest.iterrows():
        # assign the player ID:
        dataset[index, 0] = row['PLAYER_ID']
        dataset[index, 1] = row['PLAYER_NAME']

        for j in range(len(row["PTS"])):
            # assign the points scored by the player in each game, centered around the mean
            dataset[index, j + 2] = row["PTS"][j: 40 + j]
            dataset[index, j + 2] = [score - mean for score in dataset[index, j + 2]]
            dataset[index, j + 2] = np.round(dataset[index, j + 2], 3)
            # assign the points for each player in the target column
            dataset[index, 43 + j] = np.round(row["PTS"][40 + j], 3)
            if j > 39:
                dataset[index, 83] = np.round(row["PTS"][81], 3)
                break

    # cast dataset into a dataframe:
    dataset = pd.DataFrame(dataset)
    # rename the columns:
    dataset.rename(columns={0: 'PLAYER_ID', 1: 'PLAYER_NAME'}, inplace=True)

    # add a row with a problematic player with only 40 games played:
    points_w_e = players['PTS'].loc[201961].copy()
    points_w_e_norm = [score - mean for score in points_w_e]
    players['PTS'].loc[201961] = points_w_e_norm
    dataset.loc[5] = [201961, "Wayne Ellington", players['PTS'].loc[201961]] + [np.nan] * 81

    # save the dataset:
    dataset.to_csv(Path(DATA, 'dataset.csv'), index=False)
    dataset.to_pickle(Path(DATA, 'dataset.pkl'))

    # split between train and test by keeping the last match and the last previous match lists in the test set:
    # the test columns are 0, 41 and 81:
    test = dataset[['PLAYER_ID', 42, 83]].copy()
    train = dataset.drop(columns=[42, 83])

    # rename the 2-42 columns to x_train_1-x_train_40:
    train.rename(columns={i: f'x_train_{i - 1}' for i in range(2, 42)}, inplace=True)
    # rename the 43-83 columns to y_train_1-y_train_40:
    train.rename(columns={i: f'y{i - 42}' for i in range(43, 83)}, inplace=True)

    # rename the 42 column to x_test_1 and the 83 column to y_test_1:
    test.rename(columns={42: 'x_test_1', 83: 'y_test_1'}, inplace=True)

    # save the scores to a pickle file:
    train.to_pickle(Path(DATA, 'train.pkl'))
    test.to_pickle(Path(DATA, 'test.pkl'))

    # save the scores to a csv file:
    train.to_csv(Path(DATA, 'train.csv'), index=False)
    test.to_csv(Path(DATA, 'test.csv'), index=False)


if __name__ == '__main__':
    main()
