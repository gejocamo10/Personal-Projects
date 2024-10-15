import pandas as pd
import numpy as np

# Load the user's dataset
new_user_file_path = '/mnt/data/data_ligainglesa_chatGPT.csv'
user_data_cleaned_fresh = pd.read_csv(new_user_file_path).drop(columns=['DateConverted'], errors='ignore')

# Convert 'Date' to datetime format and sort the data by date from oldest to newest
user_data_cleaned_fresh['Date'] = pd.to_datetime(user_data_cleaned_fresh['Date'], format='%d/%m/%Y', errors='coerce')
user_data_cleaned_fresh = user_data_cleaned_fresh.sort_values(by='Date').reset_index(drop=True)

# Step 2: Correct the ELO calculation process
# Initialize parameters for ELO
initial_elo = 1500
k_factor = 10

# Initialize a dictionary to store ELO ratings
elo_ratings_final = {}


# Function to calculate expected score
def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


# Function to update ELO rating
def update_elo(current_rating, expected, actual):
    return current_rating + k_factor * (actual - expected)


# Create ELO ratings for each team across all matches (without adding to dataframe yet)
for index, match in user_data_cleaned_fresh.iterrows():
    home_team = match['HomeTeam']
    away_team = match['AwayTeam']
    home_goals = match['FTHG']
    away_goals = match['FTAG']

    # Initialize ELO ratings for new teams
    if home_team not in elo_ratings_final:
        elo_ratings_final[home_team] = initial_elo
    if away_team not in elo_ratings_final:
        elo_ratings_final[away_team] = initial_elo

    # Get current ratings
    home_rating = elo_ratings_final[home_team]
    away_rating = elo_ratings_final[away_team]

    # Calculate expected scores
    home_expected = expected_score(home_rating, away_rating)
    away_expected = expected_score(away_rating, home_rating)

    # Determine the result of the match
    if home_goals > away_goals:
        home_actual, away_actual = 1, 0
    elif home_goals < away_goals:
        home_actual, away_actual = 0, 1
    else:
        home_actual, away_actual = 0.5, 0.5

    # Update ELO ratings
    elo_ratings_final[home_team] = update_elo(home_rating, home_expected, home_actual)
    elo_ratings_final[away_team] = update_elo(away_rating, away_expected, away_actual)

# Step 3: Attach final ELO ratings to the dataset
user_data_cleaned_fresh['HomeTeamELO'] = user_data_cleaned_fresh['HomeTeam'].map(elo_ratings_final)
user_data_cleaned_fresh['AwayTeamELO'] = user_data_cleaned_fresh['AwayTeam'].map(elo_ratings_final)

# Now, the ELOs are correctly attached to the dataset based on the final ratings.

# Proceed with the rest of the calculations
user_data_cleaned_fresh['HomeTeamAverageGoals'] = user_data_cleaned_fresh.groupby('HomeTeam')['FTHG'].transform('mean')
user_data_cleaned_fresh['AwayTeamAverageGoals'] = user_data_cleaned_fresh.groupby('AwayTeam')['FTAG'].transform('mean')
user_data_cleaned_fresh['HomeTeamMedianGoals'] = user_data_cleaned_fresh.groupby('HomeTeam')['FTHG'].transform('median')
user_data_cleaned_fresh['AwayTeamMedianGoals'] = user_data_cleaned_fresh.groupby('AwayTeam')['FTAG'].transform('median')
user_data_cleaned_fresh['Total'] = user_data_cleaned_fresh['FTHG'] + user_data_cleaned_fresh['FTAG']
user_data_cleaned_fresh['DifferenceGoals'] = user_data_cleaned_fresh['FTHG'] - user_data_cleaned_fresh['FTAG']
user_data_cleaned_fresh['DifferenceELO'] = user_data_cleaned_fresh['HomeTeamELO'] - user_data_cleaned_fresh[
    'AwayTeamELO']
user_data_cleaned_fresh['DifferenceAverageGoalsHomeAway'] = user_data_cleaned_fresh['HomeTeamAverageGoals'] - \
                                                            user_data_cleaned_fresh['AwayTeamAverageGoals']
user_data_cleaned_fresh['DifferenceMedianGoalsHomeAway'] = user_data_cleaned_fresh['HomeTeamMedianGoals'] - \
                                                           user_data_cleaned_fresh['AwayTeamMedianGoals']

# Group by matches to calculate differences by match
user_data_cleaned_fresh['MatchTeams'] = user_data_cleaned_fresh[['HomeTeam', 'AwayTeam']].apply(tuple, axis=1)
user_data_cleaned_fresh['DifferenceAverageGoalsByMatch'] = user_data_cleaned_fresh.groupby('MatchTeams')[
    'DifferenceGoals'].transform('mean')
user_data_cleaned_fresh['DifferenceMedianGoalsByMatch'] = user_data_cleaned_fresh.groupby('MatchTeams')[
    'DifferenceGoals'].transform('median')

# Calculate overall averages and medians for goals
user_data_cleaned_fresh['HomeTeamAverageGoalsTotal'] = user_data_cleaned_fresh.groupby('HomeTeam')['FTHG'].transform(
    'mean')
user_data_cleaned_fresh['AwayTeamAverageGoalsTotal'] = user_data_cleaned_fresh.groupby('AwayTeam')['FTAG'].transform(
    'mean')
user_data_cleaned_fresh['HomeTeamMedianGoalsTotal'] = user_data_cleaned_fresh.groupby('HomeTeam')['FTHG'].transform(
    'median')
user_data_cleaned_fresh['AwayTeamMedianGoalsTotal'] = user_data_cleaned_fresh.groupby('AwayTeam')['FTAG'].transform(
    'median')
user_data_cleaned_fresh['DifferenceAverageGoalsTotal'] = user_data_cleaned_fresh['HomeTeamAverageGoalsTotal'] - \
                                                         user_data_cleaned_fresh['AwayTeamAverageGoalsTotal']
user_data_cleaned_fresh['DifferenceMedianGoalsTotal'] = user_data_cleaned_fresh['HomeTeamMedianGoalsTotal'] - \
                                                        user_data_cleaned_fresh['AwayTeamMedianGoalsTotal']

# Calculate shots on target proportions
user_data_cleaned_fresh['HomeTeamProportionShotsOnTarget'] = user_data_cleaned_fresh['HST'] / user_data_cleaned_fresh[
    'HS']
user_data_cleaned_fresh['AwayTeamProportionShotsOnTarget'] = user_data_cleaned_fresh['AST'] / user_data_cleaned_fresh[
    'AS']
user_data_cleaned_fresh['DifferenceProportionShotsOnTarget'] = user_data_cleaned_fresh[
                                                                   'HomeTeamProportionShotsOnTarget'] - \
                                                               user_data_cleaned_fresh[
                                                                   'AwayTeamProportionShotsOnTarget']
user_data_cleaned_fresh['HomeTeamAverageProportionShotsOnTarget'] = user_data_cleaned_fresh.groupby('HomeTeam')[
    'HomeTeamProportionShotsOnTarget'].transform('mean')
user_data_cleaned_fresh['AwayTeamAverageProportionShotsOnTarget'] = user_data_cleaned_fresh.groupby('AwayTeam')[
    'AwayTeamProportionShotsOnTarget'].transform('mean')
user_data_cleaned_fresh['HomeTeamMedianProportionShotsOnTarget'] = user_data_cleaned_fresh.groupby('HomeTeam')[
    'HomeTeamProportionShotsOnTarget'].transform('median')
user_data_cleaned_fresh['AwayTeamMedianProportionShotsOnTarget'] = user_data_cleaned_fresh.groupby('AwayTeam')[
    'AwayTeamProportionShotsOnTarget'].transform('median')
user_data_cleaned_fresh['DifferenceAverageProportionShotsOnTargetHomeAway'] = user_data_cleaned_fresh[
                                                                                  'HomeTeamAverageProportionShotsOnTarget'] - \
                                                                              user_data_cleaned_fresh[
                                                                                  'AwayTeamAverageProportionShotsOnTarget']
user_data_cleaned_fresh['DifferenceMedianProportionShotsOnTargetHomeAway'] = user_data_cleaned_fresh[
                                                                                 'HomeTeamMedianProportionShotsOnTarget'] - \
                                                                             user_data_cleaned_fresh[
                                                                                 'AwayTeamMedianProportionShotsOnTarget']

# Group by matches to calculate shots on target by match
user_data_cleaned_fresh['DifferenceAverageProportionShotsOnTargetByMatch'] = \
user_data_cleaned_fresh.groupby('MatchTeams')['DifferenceProportionShotsOnTarget'].transform('mean')
user_data_cleaned_fresh['DifferenceMedianProportionShotsOnTargetByMatch'] = \
user_data_cleaned_fresh.groupby('MatchTeams')['DifferenceProportionShotsOnTarget'].transform('median')

# Calculate average and median shots on target
user_data_cleaned_fresh['HomeTeamAverageShotsOnTarget'] = user_data_cleaned_fresh.groupby('HomeTeam')['HST'].transform(
    'mean')
user_data_cleaned_fresh['AwayTeamAverageShotsOnTarget'] = user_data_cleaned_fresh.groupby('AwayTeam')['AST'].transform(
    'mean')
user_data_cleaned_fresh['HomeTeamMedianShotsOnTarget'] = user_data_cleaned_fresh.groupby('HomeTeam')['HST'].transform(
    'median')
user_data_cleaned_fresh['AwayTeamMedianShotsOnTarget'] = user_data_cleaned_fresh.groupby('AwayTeam')['AST'].transform(
    'median')
user_data_cleaned_fresh['DifferenceAverageShotsOnTargetHomeAway'] = user_data_cleaned_fresh[
                                                                        'HomeTeamAverageShotsOnTarget'] - \
                                                                    user_data_cleaned_fresh[
                                                                        'AwayTeamAverageShotsOnTarget']
user_data_cleaned_fresh['DifferenceMedianShotsOnTargetHomeAway'] = user_data_cleaned_fresh[
                                                                       'HomeTeamMedianShotsOnTarget'] - \
                                                                   user_data_cleaned_fresh[
                                                                       'AwayTeamMedianShotsOnTarget']

# Group by matches to calculate shots on target by match
user_data_cleaned_fresh['DifferenceShotsOnTarget'] = user_data_cleaned_fresh['HST'] - user_data_cleaned_fresh['AST']
user_data_cleaned_fresh['HomeTeamAverageShotsOnTargetByMatch'] = user_data_cleaned_fresh.groupby('MatchTeams')[
    'HST'].transform('mean')
user_data_cleaned_fresh['AwayTeamAverageShotsOnTargetByMatch'] = user_data_cleaned_fresh.groupby('MatchTeams')[
    'AST'].transform('mean')
user_data_cleaned_fresh['HomeTeamMedianShotsOnTargetByMatch'] = user_data_cleaned_fresh.groupby('MatchTeams')[
    'HST'].transform('median')
user_data_cleaned_fresh['AwayTeamMedianShotsOnTargetByMatch'] = user_data_cleaned_fresh.groupby('MatchTeams')[
    'AST'].transform('median')
user_data_cleaned_fresh['DifferenceAverageShotsOnTargetByMatch'] = user_data_cleaned_fresh.groupby('MatchTeams')[
    'DifferenceShotsOnTarget'].transform('mean')
user_data_cleaned_fresh['DifferenceMedianShotsOnTargetByMatch'] = user_data_cleaned_fresh.groupby('MatchTeams')[
    'DifferenceShotsOnTarget'].transform('median')
