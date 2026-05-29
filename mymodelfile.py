import os
import re
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error



_TEAM_NORM = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Delhi Daredevils":            "Delhi Capitals",
    "Kings XI Punjab":             "Punjab Kings",
    "Deccan Chargers":             "Sunrisers Hyderabad",
    "Rising Pune Supergiant":      "Rising Pune Supergiants",
    "Pune Warriors":               "Rising Pune Supergiants",
}

def _norm_team(t: str) -> str:
    s = str(t).strip()
    return _TEAM_NORM.get(s, s)



_VENUE_NORM = {
    "Feroz Shah Kotla":
        "Arun Jaitley Stadium",
    "Sardar Patel Stadium, Motera":
        "Narendra Modi Stadium, Ahmedabad",
    "M.Chinnaswamy Stadium":
        "M Chinnaswamy Stadium",
    "Subrata Roy Sahara Stadium":
        "Maharashtra Cricket Association Stadium",
    "Punjab Cricket Association Stadium, Mohali":
        "Punjab Cricket Association IS Bindra Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur":
        "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh",
    "Zayed Cricket Stadium, Abu Dhabi":
        "Sheikh Zayed Stadium",
}

def _norm_venue(raw: str) -> str:
    """Strip city suffix and apply rename aliases."""
    if not isinstance(raw, str):
        return "Unknown"
    v = raw.strip()
    # Full-string alias first (catches old names that include city)
    if v in _VENUE_NORM:
        return _VENUE_NORM[v]
    base = v.split(",")[0].strip()
    return _VENUE_NORM.get(base, base)




def _year_weight(year: int) -> float:
    if year >= 2024: return 6.0
    if year == 2023: return 4.0
    if year == 2022: return 3.0
    if year == 2021: return 2.0
    if year == 2020: return 1.5
    return 1.0  




def _build_alias_map(players_df: pd.DataFrame,
                     pp_df: pd.DataFrame) -> dict:
    """
    Build { delivery_abbreviated_name -> canonical_full_name }.

    Handles:
      "RG Sharma"  -> "Rohit Sharma"    (wrong extra initial in data)
      "SA Yadav"   -> "Surya Kumar Yadav" (wrong initial, unique by team)
      "DL Chahar"  -> "Deepak Chahar"   (initials-prefix match)
      "MW Short"   -> "Matthew William Short"
      "MS Dhoni"   -> "MS Dhoni"        (direct match)
      "V Kohli"    -> "Virat Kohli"
    No strings hardcoded — everything derived from the DataFrames.
    """

    canonical_names = list(players_df["Player_Name"].astype(str).str.strip())
    p_team = dict(zip(
        players_df["Player_Name"].astype(str).str.strip(),
        players_df["Team"].astype(str).str.strip()
    ))

  
    def _mode(s):
        vc = s.value_counts()
        return _norm_team(vc.index[0]) if len(vc) else ""

    dname_team: dict = {}
    for dn, t in pp_df.groupby("batsman")["batting_team"].apply(_mode).items():
        dname_team[dn] = t
    for dn, t in pp_df.groupby("bowler")["bowling_team"].apply(_mode).items():
        dname_team.setdefault(dn, t)

    all_dnames = set(dname_team.keys())

  
    by_s_t_fi: dict = {}
    for cn in canonical_names:
        parts = cn.split()
        if len(parts) < 2:
            continue
        team = p_team.get(cn, "")
        key  = (parts[-1], team, parts[0][0])
        by_s_t_fi.setdefault(key, []).append(cn)

    def _initials_candidates(dname: str) -> list:
        """Return canonical names whose surname and initial prefix match dname."""
        parts = dname.strip().split()
        if len(parts) < 2:
            return []
        surname  = parts[-1]
        initials = parts[0]       
        f_init   = initials[0]
        result   = []
        for cn in canonical_names:
            cparts = cn.split()
            if len(cparts) < 2 or cparts[-1] != surname:
                continue
            if cparts[0][0] != f_init:
                continue
            all_inits = "".join(cp[0] for cp in cparts[:-1])
           
            if (initials == all_inits
                    or all_inits.startswith(initials)
                    or initials.startswith(all_inits)):
                result.append(cn)
        return result

    alias: dict = {}

    for dn in all_dnames:
     
        if dn in canonical_names:
            alias[dn] = dn
            continue

        
        cands = _initials_candidates(dn)
        if len(cands) == 1:
            alias[dn] = cands[0]
            continue

        
        if len(cands) > 1:
            dt = dname_team.get(dn, "")
            tf = [c for c in cands if p_team.get(c, "") == dt]
            if len(tf) == 1:
                alias[dn] = tf[0]
            elif tf:
                alias[dn] = tf[0]  
            else:
                alias[dn] = cands[0]
            continue

        
        dparts = dn.strip().split()
        if len(dparts) < 2:
            continue
        dt  = dname_team.get(dn, "")
        key = (dparts[-1], dt, dparts[0][0])
        fb  = by_s_t_fi.get(key, [])
        if len(fb) == 1:
            alias[dn] = fb[0]
        elif len(fb) > 1:
            alias[dn] = fb[0]   

    return alias


def _build_id_to_name(players_df: pd.DataFrame) -> dict:
    """{ player_id(int) : canonical_full_name(str) }"""
    return {
        int(row.ID): str(row.Player_Name).strip()
        for row in players_df.itertuples(index=False)
    }


def _parse_ids(val) -> list:
    if val is None:
        return []
    s = str(val).strip()
    if s in ("", "nan"):
        return []
    ids = []
    for part in re.split(r"[,\s;]+", s):
        try:
            ids.append(int(float(part)))
        except ValueError:
            pass
    return ids


def _weighted_mean(v: np.ndarray, w: np.ndarray) -> float:
    ws = w.sum()
    return float((v * w).sum() / ws) if ws > 0 else float(v.mean())



class MyModel:
    """
    IPL Powerplay run predictor.

    fit(deliveries_df, players_df, matches_df)
    predict(test_df)  ->  DataFrame[id, predicted_score]
    """

    def __init__(self):
        self.model      = None
        self.avg_score  = None
        self._boost_offset = 0.0   

        self._id_to_name : dict = {}
        self._alias_map  : dict = {}
        self._bat_sr     : dict = {}    
        self._bowl_econ  : dict = {}
        self._team_bat   : dict = {}
        self._team_bowl  : dict = {}
        self._venue_avg  : dict = {}

        self._g_bat_sr    = None
        self._g_bowl_econ = None
        self._g_pp        = None

    def _resolve(self, dname: str) -> str:
        """Delivery name → canonical name (fallback: itself)."""
        return self._alias_map.get(dname.strip(), dname.strip())

    def _sr_for(self, name: str) -> float:
        return self._bat_sr.get(name, self._g_bat_sr)

    def _econ_for(self, name: str) -> float:
        return self._bowl_econ.get(name, self._g_bowl_econ)

    

    def fit(self,
            deliveries_df: pd.DataFrame,
            players_df:    pd.DataFrame,
            matches_df:    pd.DataFrame):
        """
        Train the model.

        Parameters
        ----------
        deliveries_df : deliveries_updated_ipl_upto_2025.csv
        players_df    : ipl_players_uniqueid.csv  (cols: ID, Player_Name, Team)
        matches_df    : matches_updated_ipl_upto_2025.csv
        """

        print(f"  Deliveries : {len(deliveries_df):,} rows")
        print(f"  Players    : {len(players_df):,} rows")
        print(f"  Matches    : {len(matches_df):,} rows")


        deliveries_df = deliveries_df.copy()
        deliveries_df["date"]         = pd.to_datetime(deliveries_df["date"], errors="coerce")
        deliveries_df["year"]         = deliveries_df["date"].dt.year
        deliveries_df["batting_team"] = deliveries_df["batting_team"].apply(_norm_team)
        deliveries_df["bowling_team"] = deliveries_df["bowling_team"].apply(_norm_team)

      
        matches_df = matches_df.copy()
        matches_df["venue_norm"] = matches_df["venue"].apply(_norm_venue)
        venue_map = (matches_df[["matchId", "venue_norm"]]
                     .drop_duplicates("matchId")
                     .set_index("matchId")["venue_norm"])
        deliveries_df["venue"] = deliveries_df["matchId"].map(venue_map).fillna("Unknown")


        pp_all = deliveries_df[deliveries_df["over"] < 6].copy()
        pp_all["total_runs"] = pp_all["batsman_runs"] + pp_all["extras"]

        self._id_to_name = _build_id_to_name(players_df)
        self._alias_map  = _build_alias_map(players_df, pp_all)

        n_resolved = sum(1 for cn in self._id_to_name.values()
                         if any(v == cn for v in self._alias_map.values()))
        print(f"  Aliases built    : {len(self._alias_map):,} "
              f"({n_resolved}/{len(self._id_to_name)} current players mapped)")

        bat_stats  = (pp_all.groupby("batsman", sort=False)
                      .agg(r=("batsman_runs", "sum"), b=("batsman_runs", "count"))
                      .eval("sr = r / b * 100"))
        bowl_stats = (pp_all.groupby("bowler", sort=False)
                      .agg(r=("total_runs", "sum"), b=("total_runs", "count"))
                      .eval("econ = r / (b / 6)"))

        self._g_bat_sr    = float(bat_stats["sr"].mean())
        self._g_bowl_econ = float(bowl_stats["econ"].mean())

        
        self._bat_sr = {}
        for dn, sr in bat_stats["sr"].items():
            cn = self._resolve(dn)
            self._bat_sr[dn] = self._bat_sr[cn] = float(sr)

        self._bowl_econ = {}
        for dn, ec in bowl_stats["econ"].items():
            cn = self._resolve(dn)
            self._bowl_econ[dn] = self._bowl_econ[cn] = float(ec)

        n_bat  = sum(1 for cn in self._id_to_name.values() if cn in self._bat_sr)
        n_bowl = sum(1 for cn in self._id_to_name.values() if cn in self._bowl_econ)
        print(f"  Stat coverage    : bat_sr {n_bat}/{len(self._id_to_name)}  "
              f"bowl_econ {n_bowl}/{len(self._id_to_name)}")

        pp_train = pp_all[pp_all["year"] >= 2019].copy()
        pp_train["w"] = pp_train["year"].apply(_year_weight)

        group_cols = ["matchId","inning","batting_team","bowling_team","venue","year"]
        pp_inn = (pp_train.groupby(group_cols, sort=False)
                  .agg(total_runs=("total_runs", "sum"))
                  .reset_index())
        pp_inn["w"] = pp_inn["year"].apply(_year_weight)

        def _wm_grp(df, col):
            return {g: _weighted_mean(s["total_runs"].values, s["w"].values)
                    for g, s in df.groupby(col)}

        self._g_pp      = _weighted_mean(pp_inn["total_runs"].values, pp_inn["w"].values)
        self.avg_score  = self._g_pp
        self._team_bat  = _wm_grp(pp_inn, "batting_team")
        self._team_bowl = _wm_grp(pp_inn, "bowling_team")
        self._venue_avg = _wm_grp(pp_inn, "venue")

        print(f"  Training innings : {len(pp_inn):,} (2019-{int(pp_train['year'].max())})")
        print(f"  Weighted PP mean : {self._g_pp:.1f}  "
              f"(unweighted = {pp_inn['total_runs'].mean():.1f})")
        print(f"  Venues indexed   : {len(self._venue_avg):,}")

      
        mid_arr  = pp_train["matchId"].values
        inn_arr  = pp_train["inning"].values
        bat_arr  = pp_train["batsman"].values
        bowl_arr = pp_train["bowler"].values
        gp  = self._g_pp
        g_b = self._g_bat_sr
        g_w = self._g_bowl_econ

        rows = []
        for row in pp_inn.itertuples(index=False):
            mask    = (mid_arr == row.matchId) & (inn_arr == row.inning)
            b_names = [self._resolve(n) for n in dict.fromkeys(bat_arr[mask])]
            w_names = [self._resolve(n) for n in dict.fromkeys(bowl_arr[mask])]
            bs = [self._sr_for(n)   for n in b_names] or [g_b]
            ws = [self._econ_for(n) for n in w_names] or [g_w]

            rows.append([
                self._team_bat.get(row.batting_team,  gp),   
                self._team_bowl.get(row.bowling_team, gp),   
                float(np.mean(bs)),                           
                float(max(bs)),                               
                float(np.mean(ws)),                        
                float(min(ws)),                              
                float(row.inning),                            
                self._venue_avg.get(row.venue, gp),        
                float(row.w),           
                float(row.total_runs),  
            ])

        arr  = np.array(rows, dtype=np.float64)
        X, sw, y = arr[:, :8], arr[:, 8], arr[:, 9]

        X_tr, X_te, y_tr, y_te, sw_tr, _ = train_test_split(
            X, y, sw, test_size=0.2, random_state=42
        )

      
        self.model = XGBRegressor(
            n_estimators=700,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
            verbosity=0,
        )
        self.model.fit(X_tr, y_tr, sample_weight=sw_tr,
                       eval_set=[(X_te, y_te)], verbose=False)

        val_raw  = self.model.predict(X_te)
        val_mae  = mean_absolute_error(y_te, val_raw)

 
        recent_mask = pp_inn["year"] >= 2022
        if recent_mask.sum() > 0:
            X_recent = arr[recent_mask.values, :8]
            y_recent = arr[recent_mask.values,  9]
            w_recent = arr[recent_mask.values,  8]
            pred_recent = self.model.predict(X_recent)
            actual_wmean = _weighted_mean(y_recent, w_recent)
            pred_wmean   = float(pred_recent.mean())
            self._boost_offset = max(0.0, actual_wmean - pred_wmean)
        else:
            self._boost_offset = 0.0

        self.model.fit(X, y, sample_weight=sw, verbose=False)

        val_boosted = val_raw + self._boost_offset
        print(f"\n  Validation MAE   : {val_mae:.2f} runs  "
              f"(boosted MAE ≈ {mean_absolute_error(y_te, val_boosted):.2f})")
        print(f"  Mean actual PP   : {y.mean():.1f}")
        print(f"  Mean val pred    : {val_raw.mean():.1f}  "
              f"→ boosted {val_raw.mean() + self._boost_offset:.1f}")
        print(f"  PP boost offset  : +{self._boost_offset:.2f} runs")

        return self


    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict powerplay scores.

        Columns matched flexibly by keyword.
        Venue names normalised automatically.
        Player IDs resolved via id_to_name map from players_df.
        Calibration boost applied automatically.
        """
        if self._g_pp is None:
            raise RuntimeError("Call fit() before predict().")

        gp  = self._g_pp
        g_b = self._g_bat_sr
        g_w = self._g_bowl_econ

        def _col(*kws):
            for c in test_df.columns:
                if all(k in c.lower() for k in kws):
                    return c
            return None

        bat_col   = _col("batsman") or _col("batter")
        bowl_col  = _col("bowler")
        inn_col   = _col("innings") or _col("inning")
        bt_col    = _col("batting", "team")
        bw_col    = _col("bowling", "team")
        venue_col = _col("venue")

        n     = len(test_df)
        X_arr = np.full((n, 8), gp, dtype=np.float64)

        for i, (_, row) in enumerate(test_df.iterrows()):

            bat_t       = _norm_team(row[bt_col]) if bt_col else ""
            X_arr[i, 0] = self._team_bat.get(bat_t, gp)

            bowl_t      = _norm_team(row[bw_col]) if bw_col else ""
            X_arr[i, 1] = self._team_bowl.get(bowl_t, gp)

            bat_ids = _parse_ids(row[bat_col] if bat_col else "")
            bs = [self._sr_for(self._id_to_name[p]) if p in self._id_to_name
                  else g_b for p in bat_ids] or [g_b]
            X_arr[i, 2] = float(np.mean(bs))
            X_arr[i, 3] = float(max(bs))

            bowl_ids = _parse_ids(row[bowl_col] if bowl_col else "")
            ws = [self._econ_for(self._id_to_name[p]) if p in self._id_to_name
                  else g_w for p in bowl_ids] or [g_w]
            X_arr[i, 4] = float(np.mean(ws))
            X_arr[i, 5] = float(min(ws))

            try:
                X_arr[i, 6] = float(row[inn_col]) if inn_col else 1.0
            except (ValueError, TypeError):
                X_arr[i, 6] = 1.0

            rv          = str(row[venue_col]).strip() if venue_col else ""
            nv          = _norm_venue(rv)
            X_arr[i, 7] = self._venue_avg.get(nv, self._venue_avg.get(rv, gp))

        raw    = self.model.predict(X_arr) if self.model else np.full(n, gp)

        boosted = raw + self._boost_offset
        scores  = np.clip(np.round(boosted), 0, None).astype(int)

        return pd.DataFrame({
            "id":              test_df["id"].values,
            "predicted_score": scores,
        })


