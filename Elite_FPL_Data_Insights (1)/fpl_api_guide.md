# FPL API Endpoints Guide (Cached)

## Base URL
`https://fantasy.premierleague.com/api/`

## Key Endpoints for Data Collection:

### 1. General Information
- **Path:** `bootstrap-static/`
- **Returns:** events, teams, elements (players), element_types (positions), game_settings
- **Key for:** Player data, team info, current GW, prices, photo keys

### 2. Fixtures
- **Path:** `fixtures/` or `fixtures/?event={gw}`
- **Returns:** All fixtures with difficulty ratings, stats
- **Key for:** Fixture difficulty, blank/double GWs

### 3. Player Details
- **Path:** `element-summary/{element_id}/`
- **Returns:** fixtures (remaining), history (past matches), history_past (seasons)
- **Key for:** Historical performance, form data

### 4. Live Gameweek Data
- **Path:** `event/{event_id}/live/`
- **Returns:** Live stats and points breakdown
- **Key for:** Current GW performance

### 5. Set Piece Notes
- **Path:** `team/set-piece-notes/`
- **Returns:** Set piece taker information per team

## Photo URL Pattern
Based on bootstrap-static elements, photo URLs follow pattern:
`https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Simon_Murray.png/250px-Simon_Murray.png

## Historical Data Access
- Current season data: Available in bootstrap-static
- Historical seasons: Need to access element-summary for each player
- Past seasons in history_past section

## Data Collection Strategy
1. Start with bootstrap-static for current season foundation
2. Get fixtures for next 6 GWs and difficulty ratings
3. Loop through players for historical data via element-summary
4. Validate photo URLs with fallback handling
5. Collect set-piece information for bonus modeling
