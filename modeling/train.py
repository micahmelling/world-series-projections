import pandas as pd
import warnings

from sklearn.metrics import log_loss

from data.data import get_postseason_results, get_batting_stats, get_pitching_stats, \
    get_historical_all_star_appearances, get_player_info, get_team_records, get_fielding_positions, get_spring_training_rosters
from helpers.helpers import create_target_dataframe, clean_batting_and_pitching_players, map_id_to_player_name, \
    add_2021_batters_and_pitchers, prep_team_level_dataframes, calculate_batting_stats, calculate_pitching_stats, \
    append_all_star_appearances, add_column_name_prefixes, merge_dataframes, \
    create_modeling_and_prediction_dataframes, split_modeling_and_prediction_dataframes, create_train_test_split, \
    make_directories_if_not_exists
from modeling.config import TARGET, TEST_SET_START_YEAR, MODEL_TRAINING_DICT, CV_SCORING, CLASS_CUTOFF, \
    MODEL_EVALUATION_LIST, CV_SPLITS
from modeling.model import train_model
from modeling.pipeline import construct_pipeline
from modeling.evaluate import run_omnibus_model_evaluation
from modeling.explain import run_omnibus_model_explanation


warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)


def assemble_modeling_and_prediction_data():
    print('assembling training data...')
    batting_df = get_batting_stats()
    pitching_df = get_pitching_stats()
    all_star_df = get_historical_all_star_appearances()
    postseason_df = get_postseason_results()
    team_records_df = get_team_records()
    player_df = get_player_info()
    positions_df = get_fielding_positions()
    current_rosters_df = get_spring_training_rosters()
    target_df = create_target_dataframe(postseason_df)
    batting_2021_df = map_id_to_player_name(player_df, current_rosters_df)
    pitching_2021_df = map_id_to_player_name(player_df, current_rosters_df)
    batting_df, pitching_df = clean_batting_and_pitching_players(batting_df, pitching_df, positions_df)
    batting_df, pitching_df = add_2021_batters_and_pitchers(batting_df, pitching_df, batting_2021_df, pitching_2021_df)
    team_records_df, postseason_df = prep_team_level_dataframes(team_records_df, postseason_df)
    batting_df = calculate_batting_stats(player_df, batting_df)
    pitching_df = calculate_pitching_stats(player_df, pitching_df)
    batting_df, pitching_df = append_all_star_appearances(all_star_df, batting_df, pitching_df)
    team_records_df, batting_df, pitching_df = add_column_name_prefixes(team_records_df, batting_df, pitching_df)
    teams_df, teams_batting_df, teams_pitching_df = merge_dataframes(team_records_df, postseason_df, batting_df,
                                                                     pitching_df)
    teams_df = create_modeling_and_prediction_dataframes(teams_df, teams_batting_df, teams_pitching_df, target_df)
    modeling_df, prediction_df = split_modeling_and_prediction_dataframes(teams_df)
    return modeling_df, prediction_df


def prepare_training_data(df):
    return create_train_test_split(df, TARGET, TEST_SET_START_YEAR)


def train_and_evaluate_models(x_train, x_test, y_train, y_test):
    for model_name, model_config in MODEL_TRAINING_DICT.items():
        make_directories_if_not_exists([model_name])
        pipeline = train_model(x_train, y_train, construct_pipeline, model_name, model_config[0], model_config[1],
                               model_config[2], CV_SPLITS, CV_SCORING)
        run_omnibus_model_evaluation(pipeline, model_name, x_test, y_test, CLASS_CUTOFF, TARGET, MODEL_EVALUATION_LIST)
        run_omnibus_model_explanation(pipeline, x_test, y_test, x_train, y_train, log_loss, CV_SCORING, 'probability',
                                      model_name, True)


def main():
    modeling_df, prediction_df = assemble_modeling_and_prediction_data()
    x_train, x_test, y_train, y_test = prepare_training_data(modeling_df)
    train_and_evaluate_models(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
