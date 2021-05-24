import pandas as pd
import requests

from bs4 import BeautifulSoup

from constants import TEAM_NAME_MAP


def get_postseason_results():
    df = pd.read_csv('https://github.com/chadwickbureau/baseballdatabank/blob/master/core/SeriesPost.csv?raw=true')
    return df


def get_batting_stats():
    df = pd.read_csv('https://github.com/chadwickbureau/baseballdatabank/blob/master/core/Batting.csv?raw=true')
    return df


def get_pitching_stats():
    df = pd.read_csv('https://github.com/chadwickbureau/baseballdatabank/blob/master/core/Pitching.csv?raw=true')
    return df


def get_fielding_positions():
    df = pd.read_csv('https://github.com/chadwickbureau/baseballdatabank/blob/master/core/Appearances.csv?raw=true')
    return df


def get_historical_all_star_appearances():
    df = pd.read_csv('https://github.com/chadwickbureau/baseballdatabank/blob/master/core/AllstarFull.csv?raw=true')
    return df


def get_player_info():
    df = pd.read_csv('https://github.com/chadwickbureau/baseballdatabank/blob/master/core/People.csv?raw=true')
    return df


def get_team_records():
    df = pd.read_csv('https://github.com/chadwickbureau/baseballdatabank/blob/master/core/Teams.csv?raw=true')
    return df


def get_spring_training_rosters(year, team_name_map):
    """
    Finding MLB spring training rosters can be difficult. They can luckily be scraped from Wikipedia. Once the season
    starts, you can easily get rosters from other sources. Therefore, this is really only useful for getting 40-man
    rosters during spring training.

    Scrapes Wikipedia's tables of MLB spring training rosters and returns a pandas dataframe with three columns:
    yearID, teamID, and player_name. The first two columns are named as such to be consistent with the Lahman database.
    Likewise, the team name map will ensure the teamID is consistent with the Lahman database. Additional wrangling
    of the player_name will be needed to make this data consistent with something like the Lahman database.

    :param year: current year
    :type year: int
    :param team_name_map: mapping of Wikipedia's team names to the Lahman database's teamID
    :type team_name_map: dict
    :returns: pandas dataframe
    """
    url = "https://en.wikipedia.org/wiki/List_of_Major_League_Baseball_team_rosters"
    res = requests.get(url).text
    soup = BeautifulSoup(res, 'html.parser')

    players = []
    teams = []

    for table in soup.find_all('table'):
        for tr in table.find_all('tr'):
            team_players = []
            for a in tr.find_all('a', href=True):
                team_players.append(a.string)
            players.append(team_players)

    for th in soup.find_all('th'):
        teams.append(th.text)

    teams = [x for x in teams if x not in ['40-man roster\n', 'Non-roster invitees\n', 'Coaches/Other\n',
                                           'vteMajor League Baseball team rosters and affiliated Minor League Baseball team rosters',
                                           'MLB', 'MiLB', 'Triple-A', 'Double-A', 'High-A', 'Low-A', 'Rookie',
                                           'Offseason']]
    teams = [x.replace(f' {year} spring training rostervte\n', '') for x in teams]
    teams = [x.rstrip() for x in teams]

    players = [l for l in players if l != []]
    players = [l for l in players if l != ['v', 't', 'e']]
    players = players[:len(players) - 15]

    teams_df = pd.DataFrame()
    for index, team in enumerate(teams):
        if index <= 29:
            players[index] = [x for x in players[index] if x != None]
            temp_df = pd.DataFrame({team: players[index]})
            teams_df = pd.concat([teams_df, temp_df], axis=1)
    teams_df = teams_df.head(40)

    final_df = pd.DataFrame()
    teams = list(teams_df)
    for team in teams:
        temp_df = teams_df[[team]]
        temp_df.rename(columns={team: 'player_name'}, inplace=True)
        temp_df['teamID'] = team
        final_df = final_df.append(temp_df)

    final_df['yearID'] = year
    final_df['teamID'] = final_df['teamID'].map(team_name_map)
    final_df['player_name'] = final_df['player_name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').\
        str.decode('utf-8')
    final_df['player_name'] = final_df['player_name'].str.replace(' Jr.', '')
    final_df['player_name'] = final_df['player_name'].str.replace('A. J.', 'AJ')
    return final_df