import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import kagglehub
from difflib import get_close_matches

path = kagglehub.dataset_download("pabloramoswilkins/ucl-2025-players-data")
print("Path to dataset files:", path)

# Load datasets
teams_data = pd.read_csv(f"{path}/teams_data.csv")[['team_id', 'country', 'team', 'logo']]
attacking_data = pd.read_csv(f"{path}/DAY_4/attacking_data.csv")[['id_player', 'assists', 'corners_taken', 'offsides', 'dribbles']]
attempts_data = pd.read_csv(f"{path}/DAY_4/attempts_data.csv")[['id_player', 'total_attempts', 'attempts_on_target', 'attempts_off_target', 'blocked']]
defending_data = pd.read_csv(f"{path}/DAY_4/defending_data.csv")[['id_player', 'balls_recovered', 'tackles', 'tackles_won', 'tackles_lost', 'clearance_attempted']]
disciplinary_data = pd.read_csv(f"{path}/DAY_4/disciplinary_data.csv")[['id_player', 'fouls_committed', 'fouls_suffered', 'yellow_cards', 'red_cards']]
distribution_data = pd.read_csv(f"{path}/DAY_4/distribution_data.csv")[['id_player', 'passing_accuracy(%)', 'passes_attempted', 'passes_completed', 'crossing_accuracy(%)', 'crosses_attempted', 'crosses_completed', 'free_kick_taken', 'matches_appearance']]
goalkeeping_data = pd.read_csv(f"{path}/DAY_4/goalkeeping_data.csv")[['id_player', 'saves', 'goals_conceded', 'saves_on_penalty', 'clean_sheets', 'punches_made']]
goals_data = pd.read_csv(f"{path}/DAY_4/goals_data.csv")[['id_player', 'goals', 'inside_area', 'outside_area', 'right_foot', 'left_foot', 'head', 'other', 'penalties_scored']]
key_stats_data = pd.read_csv(f"{path}/DAY_4/key_stats_data.csv")[['id_player', 'distance_covered(km/h)', 'top_speed', 'minutes_played', 'matches_appareance']]
players_data = pd.read_csv(f"{path}/DAY_4/players_data.csv")[['id_player', 'player_name', 'nationality', 'field_position', 'weight(kg)', 'height(cm)', 'age', 'id_team', 'player_image']]

# Basic stats and info
print(teams_data.head())
print(teams_data.info())
print(teams_data.describe())

# Merge datasets based on player and team identifiers
player_data = players_data.merge(attacking_data, on='id_player', how='left')
player_data = player_data.merge(attempts_data, on='id_player', how='left')
player_data = player_data.merge(defending_data, on='id_player', how='left')
player_data = player_data.merge(disciplinary_data, on='id_player', how='left')
player_data = player_data.merge(distribution_data, on='id_player', how='left')
player_data = player_data.merge(goalkeeping_data, on='id_player', how='left')
player_data = player_data.merge(goals_data, on='id_player', how='left')
player_data = player_data.merge(key_stats_data, on='id_player', how='left')

# Group by team to create aggregated team statistics
team_stats = player_data.groupby('id_team').agg({
    'assists': 'sum',
    'total_attempts': 'sum',
    'tackles': 'sum',
    'passes_completed': 'sum',
    'goals': 'sum',
    'top_speed': 'mean',
    'distance_covered(km/h)': 'sum'
}).reset_index()

# Merge team statistics with team information
team_stats = team_stats.merge(teams_data, left_on='id_team', right_on='team_id', how='left')

# Renaming columns for easier understanding
team_stats.columns = [
    'id_team', 'total_assists', 'total_attempts', 'total_tackles', 'total_passes_completed', 
    'total_goals', 'avg_top_speed', 'total_distance_covered', 'team_id', 'country', 'team_name', 'logo'
]
print(team_stats.head())

team_stats['home_advantage'] = 1.0  # Default value for home advantage

def calculate_team_strength(team_row):
    # Define weights for each feature to calculate a team's overall strength
    weights = {
        'total_goals': 5,
        'total_assists': 2,
        'total_attempts': 2.25,
        'total_tackles': 1.5,
        'total_passes_completed': 2.5,
        'avg_top_speed': 1,
        'total_distance_covered': 1,
        'home_advantage': 1.2,
    }
    
    score = (
        team_row['total_goals'] * weights['total_goals'] +
        team_row['total_assists'] * weights['total_assists'] +
        team_row['total_attempts'] * weights['total_attempts'] +
        team_row['total_tackles'] * weights['total_tackles'] +
        team_row['total_passes_completed'] * weights['total_passes_completed'] +
        team_row['avg_top_speed'] * weights['avg_top_speed'] +
        team_row['total_distance_covered'] * weights['total_distance_covered'] +
        team_row['home_advantage'] * weights['home_advantage']
    )
    return score

# Calculate team strength scores
team_stats['team_strength'] = team_stats.apply(calculate_team_strength, axis=1)
print(team_stats[['team_name', 'team_strength']].head())

def create_match_dataset(team_stats):
    match_data = []
    teams = team_stats['team_name'].values
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            team_a = team_stats.iloc[i]
            team_b = team_stats.iloc[j]
            match = {
                'team_a_total_assists': team_a['total_assists'],
                'team_a_total_attempts': team_a['total_attempts'],
                'team_a_total_tackles': team_a['total_tackles'],
                'team_a_total_passes_completed': team_a['total_passes_completed'],
                'team_a_total_goals': team_a['total_goals'],
                'team_a_avg_top_speed': team_a['avg_top_speed'],
                'team_a_total_distance_covered': team_a['total_distance_covered'],
                'team_b_total_assists': team_b['total_assists'],
                'team_b_total_attempts': team_b['total_attempts'],
                'team_b_total_tackles': team_b['total_tackles'],
                'team_b_total_passes_completed': team_b['total_passes_completed'],
                'team_b_total_goals': team_b['total_goals'],
                'team_b_avg_top_speed': team_b['avg_top_speed'],
                'team_b_total_distance_covered': team_b['total_distance_covered'],
                'match_result': 0 if team_a['team_strength'] > team_b['team_strength'] else 1
            }
            match_data.append(match)
    return pd.DataFrame(match_data)

# Create match dataset based on team strengths
match_data = create_match_dataset(team_stats)
print(match_data.head())

# Features and target
X = match_data.drop(columns=['match_result'])
y = match_data['match_result']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model using XGBoost
model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

# Basic grid search example
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predictions and evaluation
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to predict match result between two teams
def predict_match(team_a_name, team_b_name):
    team_a_stats = team_stats[team_stats['team_name'].str.lower() == team_a_name.strip().lower()]
    team_b_stats = team_stats[team_stats['team_name'].str.lower() == team_b_name.strip().lower()]

    if team_a_stats.empty or team_b_stats.empty:
        messagebox.showerror("Error", "One or both team names are invalid.")
        return

    # Add form factor
    team_a_form = calculate_team_form(team_a_name)
    team_b_form = calculate_team_form(team_b_name)
    
    # Adjust stats based on form
    team_a_stats = team_a_stats.copy()
    team_b_stats = team_b_stats.copy()
    
    team_a_stats['total_goals'] *= team_a_form
    team_b_stats['total_goals'] *= team_b_form
    
    match_features = pd.DataFrame([{
        'team_a_total_assists': team_a_stats['total_assists'],
        'team_a_total_attempts': team_a_stats['total_attempts'],
        'team_a_total_tackles': team_a_stats['total_tackles'],
        'team_a_total_passes_completed': team_a_stats['total_passes_completed'],
        'team_a_total_goals': team_a_stats['total_goals'],
        'team_a_avg_top_speed': team_a_stats['avg_top_speed'],
        'team_a_total_distance_covered': team_a_stats['total_distance_covered'],
        'team_b_total_assists': team_b_stats['total_assists'],
        'team_b_total_attempts': team_b_stats['total_attempts'],
        'team_b_total_tackles': team_b_stats['total_tackles'],
        'team_b_total_passes_completed': team_b_stats['total_passes_completed'],
        'team_b_total_goals': team_b_stats['total_goals'],
        'team_b_avg_top_speed': team_b_stats['avg_top_speed'],
        'team_b_total_distance_covered': team_b_stats['total_distance_covered']
    }])

    match_features_scaled = scaler.transform(match_features)
    # Get probability predictions
    probabilities = best_model.predict_proba(match_features_scaled)[0]
    
    # Add randomness and uncertainty to predictions
    noise = np.random.normal(0, 0.05)  # Add random noise with smaller standard deviation
    # Adjust probabilities to be more realistic
    team_a_prob = (probabilities[0] * 0.8 + 0.1 + noise) * 100  # Scale down confidence slightly
    team_b_prob = (1 - (probabilities[0] * 0.8 + 0.1 + noise)) * 100
    
    # Determine winner and create message
    if team_a_prob > team_b_prob:
        message = f"{team_a_name} win ({team_a_prob:.1f}% chance)"
    else:
        message = f"{team_b_name} win ({team_b_prob:.1f}% chance)"
    
    messagebox.showinfo("Match Prediction", message)

# Function to add match result after the game is played
def add_match_result(team_a_name, team_b_name, result, team_a_score, team_b_score):
    team_a_stats = team_stats[team_stats['team_name'].str.lower() == team_a_name.strip().lower()]
    team_b_stats = team_stats[team_stats['team_name'].str.lower() == team_b_name.strip().lower()]

    if team_a_stats.empty or team_b_stats.empty:
        messagebox.showerror("Error", "One or both team names are invalid.")
        return

    if not team_a_score.isdigit() or not team_b_score.isdigit():
        messagebox.showerror("Error", "Scores must be valid numbers.")
        return

    team_a_score = int(team_a_score)
    team_b_score = int(team_b_score)

    # Create match features with additional score columns
    match_features = pd.DataFrame([{
        'team_a_total_assists': team_a_stats.iloc[0]['total_assists'],
        'team_a_total_attempts': team_a_stats.iloc[0]['total_attempts'],
        'team_a_total_tackles': team_a_stats.iloc[0]['total_tackles'],
        'team_a_total_passes_completed': team_a_stats.iloc[0]['total_passes_completed'],
        'team_a_total_goals': team_a_stats.iloc[0]['total_goals'],
        'team_a_avg_top_speed': team_a_stats.iloc[0]['avg_top_speed'],
        'team_a_total_distance_covered': team_a_stats.iloc[0]['total_distance_covered'],
        'team_b_total_assists': team_b_stats.iloc[0]['total_assists'],
        'team_b_total_attempts': team_b_stats.iloc[0]['total_attempts'],
        'team_b_total_tackles': team_b_stats.iloc[0]['total_tackles'],
        'team_b_total_passes_completed': team_b_stats.iloc[0]['total_passes_completed'],
        'team_b_total_goals': team_b_stats.iloc[0]['total_goals'],
        'team_b_avg_top_speed': team_b_stats.iloc[0]['avg_top_speed'],
        'team_b_total_distance_covered': team_b_stats.iloc[0]['total_distance_covered'],
        'team_a_score': team_a_score,
        'team_b_score': team_b_score,
        'score_difference': team_a_score - team_b_score
    }])

    # Add to match data and retrain the model
    global match_data, X_train, X_test, y_train, y_test
    match_data = pd.concat([match_data, match_features], ignore_index=True)
    X = match_data.drop(columns=['match_result'])
    y = match_data['match_result']
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    best_model.fit(X_train, y_train)

    messagebox.showinfo("Success", "Match result has been added and the model has been updated.")

def find_closest_team(partial_name, team_list):
    """Find closest matching team name from partial input"""
    if not partial_name:
        return []
    
    partial_name = partial_name.lower().strip()
    # Get close matches with cutoff of 0.6 (60% similarity)
    matches = get_close_matches(partial_name, 
                              [team.lower() for team in team_list], 
                              n=5, 
                              cutoff=0.6)
    
    # Return original case team names that match
    return [team for team in team_list 
            if team.lower() in matches]

def on_team_entry_change(event, entry_widget, team_list):
    """Update dropdown list based on current entry"""
    current_text = entry_widget.get()
    if current_text:
        matches = find_closest_team(current_text, team_list)
        entry_widget['values'] = matches
        if matches:
            entry_widget.event_generate('<<ComboboxSelected>>')

def calculate_team_form(team_name, last_n_matches=5):
    """Calculate team's recent form based on last N matches"""
    team_matches = historical_matches[
        (historical_matches['team_a'] == team_name) | 
        (historical_matches['team_b'] == team_name)
    ].tail(last_n_matches)
    
    if len(team_matches) == 0:
        return 1.0  # Default form
        
    wins = 0
    for _, match in team_matches.iterrows():
        if match['team_a'] == team_name:
            wins += 1 if match['team_a_score'] > match['team_b_score'] else 0
        else:
            wins += 1 if match['team_b_score'] > match['team_a_score'] else 0
            
    return (wins / len(team_matches)) * 1.5  # Form multiplier

# GUI Setup
root = tk.Tk()
root.title("Soccer Team Predictor")
root.geometry("600x700")

team_list = sorted(team_stats['team_name'].tolist())

# Team A (Home) Dropdown
team_a_label = tk.Label(root, text="Select Home Team (Team A):")
team_a_label.pack(pady=5)
team_a_var = tk.StringVar()
team_a_dropdown = ttk.Combobox(root, textvariable=team_a_var, values=team_list)
team_a_dropdown.pack(pady=5)
team_a_dropdown.bind('<KeyRelease>', lambda e: on_team_entry_change(e, team_a_dropdown, team_list))

# Team B (Away) Dropdown
team_b_label = tk.Label(root, text="Select Away Team (Team B):")
team_b_label.pack(pady=5)
team_b_var = tk.StringVar()
team_b_dropdown = ttk.Combobox(root, textvariable=team_b_var, values=team_list)
team_b_dropdown.pack(pady=5)
team_b_dropdown.bind('<KeyRelease>', lambda e: on_team_entry_change(e, team_b_dropdown, team_list))

# Predict Winner Button
predict_button = tk.Button(root, text="Predict Winner", command=lambda: predict_match(team_a_var.get(), team_b_var.get()))
predict_button.pack(pady=20)

# Back Button
def reset_inputs():
    team_a_var.set("")
    team_b_var.set("")
    team_a_score_var.set("")
    team_b_score_var.set("")
    result_var.set("")

back_button = tk.Button(root, text="Enter New Teams", command=reset_inputs)
back_button.pack(pady=10)

# Add Match Result Section For Training
result_label = tk.Label(root, text="Enter Actual Match Result:")
result_label.pack(pady=10)
result_var = tk.StringVar()
result_dropdown = ttk.Combobox(root, textvariable=result_var, values=sorted(team_stats['team_name'].tolist()) + ["Tie"])
result_dropdown.pack(pady=5)

# Add Score Input Fields
score_frame = tk.Frame(root)
score_frame.pack(pady=10)

team_a_score_label = tk.Label(score_frame, text="Home Team (Team A) Score:")
team_a_score_label.pack(side=tk.LEFT, padx=5)
team_a_score_var = tk.StringVar()
team_a_score_entry = tk.Entry(score_frame, textvariable=team_a_score_var, width=5)
team_a_score_entry.pack(side=tk.LEFT, padx=5)

team_b_score_label = tk.Label(score_frame, text="Away Team (Team B) Score:")
team_b_score_label.pack(side=tk.LEFT, padx=5)
team_b_score_var = tk.StringVar()
team_b_score_entry = tk.Entry(score_frame, textvariable=team_b_score_var, width=5)
team_b_score_entry.pack(side=tk.LEFT, padx=5)

# Update the add_result_button command
add_result_button = tk.Button(root, text="Add Match Result", command=lambda: add_match_result(team_a_var.get(), team_b_var.get(), result_var.get(), team_a_score_var.get(), team_b_score_var.get()))
add_result_button.pack(pady=10)

# Add after team_stats creation
historical_matches = pd.DataFrame({
    'team_a': [],
    'team_b': [],
    'team_a_score': [],
    'team_b_score': [],
    'date': []
})

def update_historical_data(team_a, team_b, score_a, score_b):
    global historical_matches
    historical_matches = historical_matches.append({
        'team_a': team_a,
        'team_b': team_b,
        'team_a_score': score_a,
        'team_b_score': score_b,
        'date': pd.Timestamp.now()
    }, ignore_index=True)

root.mainloop()