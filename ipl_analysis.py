"""
╔══════════════════════════════════════════════════════╗
║         IPL CRICKET DATA ANALYSIS                    ║
║  Python | Pandas | NumPy | Matplotlib | Seaborn      ║
╚══════════════════════════════════════════════════════╝

Dataset: IPL matches data (2008-2022)
Download from: https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#  STYLE SETUP
# ============================================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
          '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

# ============================================================
#  LOAD DATA
# ============================================================
def load_data():
    """Load IPL matches and deliveries data"""
    print("\n📂 Loading IPL Dataset...")
    try:
        matches    = pd.read_csv('matches.csv')
        deliveries = pd.read_csv('deliveries.csv')
        print(f"✅ Matches loaded    : {matches.shape[0]} rows, {matches.shape[1]} columns")
        print(f"✅ Deliveries loaded : {deliveries.shape[0]} rows, {deliveries.shape[1]} columns")
        return matches, deliveries
    except FileNotFoundError:
        print("⚠️  Dataset not found! Generating sample data for demo...")
        return generate_sample_data()

def generate_sample_data():
    """Generate realistic sample IPL data for demo"""
    np.random.seed(42)
    teams = ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
             'Kolkata Knight Riders', 'Delhi Capitals', 'Sunrisers Hyderabad',
             'Rajasthan Royals', 'Punjab Kings']

    seasons = list(range(2008, 2023))
    n = 800

    matches = pd.DataFrame({
        'id'              : range(1, n+1),
        'season'          : np.random.choice(seasons, n),
        'city'            : np.random.choice(['Mumbai','Chennai','Bangalore','Kolkata','Delhi','Hyderabad'], n),
        'team1'           : np.random.choice(teams, n),
        'team2'           : np.random.choice(teams, n),
        'winner'          : np.random.choice(teams, n),
        'win_by_runs'     : np.random.randint(0, 100, n),
        'win_by_wickets'  : np.random.randint(0, 10, n),
        'player_of_match' : np.random.choice(['V Kohli','MS Dhoni','RG Sharma','AB de Villiers',
                                               'SK Raina','KA Pollard','SR Watson','DA Warner'], n),
        'toss_winner'     : np.random.choice(teams, n),
        'toss_decision'   : np.random.choice(['bat', 'field'], n),
    })

    players = ['V Kohli','MS Dhoni','RG Sharma','AB de Villiers','SK Raina',
               'KA Pollard','SR Watson','DA Warner','RR Pant','HH Pandya']
    n_del = 5000

    deliveries = pd.DataFrame({
        'match_id'         : np.random.randint(1, n+1, n_del),
        'inning'           : np.random.choice([1, 2], n_del),
        'batting_team'     : np.random.choice(teams, n_del),
        'bowling_team'     : np.random.choice(teams, n_del),
        'batsman'          : np.random.choice(players, n_del),
        'bowler'           : np.random.choice(players, n_del),
        'batsman_runs'     : np.random.choice([0,1,2,3,4,6], n_del, p=[0.35,0.30,0.12,0.03,0.12,0.08]),
        'extra_runs'       : np.random.choice([0,1,2], n_del, p=[0.85,0.10,0.05]),
        'total_runs'       : np.random.choice([0,1,2,3,4,6], n_del),
        'player_dismissed' : np.random.choice(players + [None]*5, n_del),
        'dismissal_kind'   : np.random.choice(['caught','bowled','lbw','run out', None], n_del),
    })

    return matches, deliveries

# ============================================================
#  ANALYSIS 1: Most Successful Teams
# ============================================================
def analysis_team_wins(matches):
    print("\n📊 Analysis 1: Most Successful Teams...")
    wins = matches['winner'].value_counts().head(8)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(wins.index, wins.values, color=COLORS[:len(wins)], edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, wins.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_title('🏆 Most Successful IPL Teams (By Wins)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Teams', fontsize=12)
    ax.set_ylabel('Number of Wins', fontsize=12)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig('1_team_wins.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 1_team_wins.png")

# ============================================================
#  ANALYSIS 2: Season-wise Match Count
# ============================================================
def analysis_season_matches(matches):
    print("\n📊 Analysis 2: Season-wise Match Count...")
    season_counts = matches.groupby('season').size().reset_index(name='matches')

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(season_counts['season'], season_counts['matches'],
            marker='o', linewidth=2.5, color='#FF6B6B', markersize=8, markerfacecolor='white',
            markeredgewidth=2)
    ax.fill_between(season_counts['season'], season_counts['matches'], alpha=0.2, color='#FF6B6B')

    for _, row in season_counts.iterrows():
        ax.annotate(str(int(row['matches'])),
                    (row['season'], row['matches']),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, fontweight='bold')

    ax.set_title('📅 Number of Matches Per IPL Season', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Number of Matches', fontsize=12)
    ax.set_xticks(season_counts['season'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('2_season_matches.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 2_season_matches.png")

# ============================================================
#  ANALYSIS 3: Top Run Scorers
# ============================================================
def analysis_top_batsmen(deliveries):
    print("\n📊 Analysis 3: Top Run Scorers...")
    top_batsmen = deliveries.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(top_batsmen.index[::-1], top_batsmen.values[::-1],
                   color=COLORS[:len(top_batsmen)], edgecolor='white')

    for bar, val in zip(bars, top_batsmen.values[::-1]):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontweight='bold', fontsize=10)

    ax.set_title('🏏 Top 10 Run Scorers in IPL History', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Total Runs', fontsize=12)
    ax.set_ylabel('Batsman', fontsize=12)
    plt.tight_layout()
    plt.savefig('3_top_batsmen.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 3_top_batsmen.png")

# ============================================================
#  ANALYSIS 4: Top Wicket Takers
# ============================================================
def analysis_top_bowlers(deliveries):
    print("\n📊 Analysis 4: Top Wicket Takers...")
    legal_dismissals = ['caught', 'bowled', 'lbw', 'stumped', 'caught and bowled', 'hit wicket']
    wickets = deliveries[deliveries['dismissal_kind'].isin(legal_dismissals)]
    top_bowlers = wickets.groupby('bowler')['player_dismissed'].count().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(top_bowlers.index, top_bowlers.values,
                  color=COLORS[:len(top_bowlers)], edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, top_bowlers.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_title('🎳 Top 10 Wicket Takers in IPL History', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Bowler', fontsize=12)
    ax.set_ylabel('Total Wickets', fontsize=12)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig('4_top_bowlers.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 4_top_bowlers.png")

# ============================================================
#  ANALYSIS 5: Toss Decision Impact
# ============================================================
def analysis_toss_impact(matches):
    print("\n📊 Analysis 5: Toss Decision Impact...")
    toss = matches.copy()
    toss['toss_win_match'] = toss['toss_winner'] == toss['winner']

    toss_impact = toss.groupby('toss_decision')['toss_win_match'].mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    axes[0].pie(toss_impact.values, labels=toss_impact.index,
                autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'],
                startangle=90, textprops={'fontsize': 12})
    axes[0].set_title('Toss Win → Match Win %', fontsize=13, fontweight='bold')

    # Bar chart - toss decision count
    toss_counts = matches['toss_decision'].value_counts()
    axes[1].bar(toss_counts.index, toss_counts.values, color=['#FF6B6B', '#4ECDC4'],
                edgecolor='white', linewidth=1.5)
    for i, (idx, val) in enumerate(toss_counts.items()):
        axes[1].text(i, val + 2, str(val), ha='center', fontweight='bold', fontsize=12)
    axes[1].set_title('Toss Decision: Bat vs Field', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Count')

    plt.suptitle('🪙 Toss Impact Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('5_toss_impact.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 5_toss_impact.png")

# ============================================================
#  ANALYSIS 6: Player of the Match
# ============================================================
def analysis_player_of_match(matches):
    print("\n📊 Analysis 6: Most Player of the Match Awards...")
    top_players = matches['player_of_match'].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(top_players.index[::-1], top_players.values[::-1],
                   color=COLORS[:len(top_players)], edgecolor='white')

    for bar, val in zip(bars, top_players.values[::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontweight='bold', fontsize=11)

    ax.set_title('⭐ Most Player of the Match Awards', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Awards', fontsize=12)
    ax.set_ylabel('Player', fontsize=12)
    plt.tight_layout()
    plt.savefig('6_player_of_match.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 6_player_of_match.png")

# ============================================================
#  ANALYSIS 7: Runs Distribution (Boundaries)
# ============================================================
def analysis_runs_distribution(deliveries):
    print("\n📊 Analysis 7: Runs Distribution per Ball...")
    runs_dist = deliveries['batsman_runs'].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(runs_dist.index.astype(str), runs_dist.values,
                  color=COLORS[:len(runs_dist)], edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, runs_dist.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(val), ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_title('📈 Runs Distribution per Ball', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Runs per Ball', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig('7_runs_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 7_runs_distribution.png")

# ============================================================
#  SUMMARY STATS
# ============================================================
def print_summary(matches, deliveries):
    print("\n" + "="*55)
    print("          📊 IPL DATA ANALYSIS SUMMARY")
    print("="*55)
    print(f"  Total Matches Played     : {len(matches)}")
    print(f"  Seasons Covered          : {matches['season'].min()} - {matches['season'].max()}")
    print(f"  Total Runs Scored        : {deliveries['total_runs'].sum():,}")
    print(f"  Total Boundaries (4s+6s) : {len(deliveries[deliveries['batsman_runs'].isin([4,6])]):,}")
    print(f"  Most Successful Team     : {matches['winner'].value_counts().index[0]}")
    print(f"  Top Run Scorer           : {deliveries.groupby('batsman')['batsman_runs'].sum().idxmax()}")
    print(f"  Most POM Awards          : {matches['player_of_match'].value_counts().index[0]}")
    print("="*55)

# ============================================================
#  MAIN
# ============================================================
def main():
    print("="*55)
    print("       🏏 IPL CRICKET DATA ANALYSIS")
    print("   Python | Pandas | NumPy | Matplotlib")
    print("="*55)

    # Load data
    matches, deliveries = load_data()

    # Run all analyses
    analysis_team_wins(matches)
    analysis_season_matches(matches)
    analysis_top_batsmen(deliveries)
    analysis_top_bowlers(deliveries)
    analysis_toss_impact(matches)
    analysis_player_of_match(matches)
    analysis_runs_distribution(deliveries)

    # Print summary
    print_summary(matches, deliveries)

    print("\n✅ All 7 analyses complete!")
    print("📁 7 charts saved as PNG files!")

if __name__ == "__main__":
    main()