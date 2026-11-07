# Chord evaluation: eval/data/d2 vs @drop-passing

Created 2026-10-13 16:02. Items scored: 72; missing predictions: 0.

## Summary

| metric | value | Δ vs last run |
|---|---|---|
| root | 0.911 |  |
| majmin | 0.924 |  |
| majmin_inv | 0.914 |  |
| sevenths_inv | 0.920 |  |
| tetrads_inv | 0.875 |  |
| seg | 0.837 |  |
| short R majmin_inv (<=1 beat) | 0.056 |  |
| short P majmin_inv (<=1 beat) | 1.000 |  |
| short F1 majmin_inv (<=1 beat) | 0.106 |  |
| short F1 sevenths_inv (<=1 beat) | 0.106 |  |
| short R majmin_inv (<=0.6 s) | 0.048 |  |
| flicker / min | 0.000 |  |
| boundary F 100 ms | 0.788 |  |
| boundary F 1/4 beat | 0.788 |  |
| bass acc | 0.905 |  |
| out-of-chord bass acc | 0.338 |  |

## Standard metrics (mir_eval, mean over items)

| metric | value |
|---|---|
| root | 0.911 |
| majmin | 0.924 |
| majmin_inv | 0.914 |
| mirex | 0.895 |
| thirds | 0.899 |
| thirds_inv | 0.889 |
| triads | 0.888 |
| triads_inv | 0.878 |
| sevenths | 0.928 |
| sevenths_inv | 0.920 |
| tetrads | 0.883 |
| tetrads_inv | 0.875 |
| seg | 0.837 |
| overseg | 1.000 |
| underseg | 0.837 |

## Short segments (pooled)

Hit = >= 50 % overlap, onset within ±max(70 ms, ¼ beat), label match at the level.

| definition | level | n ref | recall | n est | precision | F1 |
|---|---|---|---|---|---|---|
| shortseg_1beat | root | 321 | 0.056 | 18 | 1.000 | 0.106 |
| shortseg_1beat | majmin_inv | 321 | 0.056 | 18 | 1.000 | 0.106 |
| shortseg_1beat | sevenths_inv | 321 | 0.056 | 18 | 1.000 | 0.106 |
| shortseg_0.6s | root | 249 | 0.048 | 12 | 1.000 | 0.092 |
| shortseg_0.6s | majmin_inv | 249 | 0.048 | 12 | 1.000 | 0.092 |
| shortseg_0.6s | sevenths_inv | 249 | 0.048 | 12 | 1.000 | 0.092 |

## Accuracy by reference duration (beats, seconds-weighted)

| bin | n | root | majmin_inv | sevenths_inv |
|---|---|---|---|---|
| (0,0.5] | 121 | 0.348 | 0.000 | 0.000 |
| (0.5,1] | 200 | 0.352 | 0.121 | 0.089 |
| (1,2] | 276 | 1.000 | 1.000 | 1.000 |
| (2,4] | 273 | 1.000 | 1.000 | 1.000 |
| >4 | 0 | – | – | – |

## Idiom recall (tagged reference chords)

| idiom | n | root | majmin_inv | sevenths_inv |
|---|---|---|---|---|
| augmented_passing | 24 | 0.000 | 0.000 | 0.000 |
| borrowed_iv | 60 | 0.000 | 0.000 | 0.000 |
| descending_canon | 27 | 0.000 | 0.000 | 0.000 |
| four_over_five | 18 | 0.000 | 0.000 | 0.000 |
| huijia_prechorus | 12 | 0.000 | 0.000 | 0.000 |
| passing_diminished | 27 | 0.000 | 0.000 | 0.000 |
| pedal_neighbour | 18 | 0.000 | 0.000 | 0.000 |
| rising_m7 | 24 | 0.250 | 0.250 | 0.250 |
| royal_road | 18 | 0.000 | 0.000 | 0.000 |
| secondary_dominant | 18 | 0.000 | 0.000 | 0.000 |
| sus_resolution | 27 | 0.000 | 0.000 | 0.000 |
| walk_up | 36 | 0.000 | 0.000 | 0.000 |

## By tempo (BPM)

| group | majmin_inv | short R majmin_inv (<=1 beat) | short F1 sevenths_inv (<=1 beat) | flicker / min | bass acc |
|---|---|---|---|---|---|
| 110.0 | 0.914 | 0.058 | 0.110 | 0.000 | 0.906 |
| 65.0 | 0.914 | 0.053 | 0.100 | 0.000 | 0.904 |

## By idiom

| group | majmin_inv | short R majmin_inv (<=1 beat) | short F1 sevenths_inv (<=1 beat) | flicker / min | bass acc |
|---|---|---|---|---|---|
| augmented_passing | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| borrowed_iv | 0.829 | 0.000 | 0.000 | 0.000 | 0.905 |
| descending_canon | 0.858 | 0.000 | 0.000 | 0.000 | 0.845 |
| four_over_five | 1.000 | 0.000 | 0.000 | 0.000 | 0.908 |
| huijia_prechorus | 0.819 | 0.000 | 0.000 | 0.000 | 0.893 |
| passing_diminished | 1.000 | 0.250 | 0.400 | 0.000 | 0.844 |
| pedal_neighbour | 0.915 | 0.000 | 0.000 | 0.000 | 1.000 |
| rising_m7 | 0.915 | 0.000 | 0.000 | 0.000 | 0.908 |
| royal_road | 0.915 | 0.000 | 0.000 | 0.000 | 0.970 |
| secondary_dominant | 0.914 | 0.000 | 0.000 | 0.000 | 0.906 |
| sus_resolution | 1.000 | 0.250 | 0.400 | 0.000 | 0.906 |
| walk_up | 0.800 | 0.000 | 0.000 | 0.000 | 0.781 |

## Per item

| item | root | majmin_inv | sevenths_inv | short hits | short P | flicker/min | bass |
|---|---|---|---|---|---|---|---|
| augmented_passing_Ab_065 | 1.000 | 1.000 | 1.000 | 0/2 | – | 0.000 | 1.000 |
| augmented_passing_Ab_110 | 1.000 | 1.000 | 1.000 | 0/4 | – | 0.000 | 1.000 |
| augmented_passing_C_065 | 1.000 | 1.000 | 1.000 | 0/2 | – | 0.000 | 1.000 |
| augmented_passing_C_110 | 1.000 | 1.000 | 1.000 | 0/4 | – | 0.000 | 1.000 |
| augmented_passing_E_065 | 1.000 | 1.000 | 1.000 | 0/2 | – | 0.000 | 1.000 |
| augmented_passing_E_110 | 1.000 | 1.000 | 1.000 | 0/4 | – | 0.000 | 1.000 |
| borrowed_iv_A_065 | 0.915 | 0.830 | 0.907 | 0/4 | – | 0.000 | 0.906 |
| borrowed_iv_A_110 | 0.914 | 0.827 | 0.906 | 0/8 | – | 0.000 | 0.906 |
| borrowed_iv_Db_065 | 0.915 | 0.830 | 0.907 | 0/4 | – | 0.000 | 0.906 |
| borrowed_iv_Db_110 | 0.907 | 0.827 | 0.899 | 0/8 | – | 0.000 | 0.899 |
| borrowed_iv_F_065 | 0.915 | 0.830 | 0.907 | 0/4 | – | 0.000 | 0.906 |
| borrowed_iv_F_110 | 0.914 | 0.827 | 0.906 | 0/8 | – | 0.000 | 0.906 |
| descending_canon_Bb_065 | 0.887 | 0.858 | 0.858 | 0/3 | – | 0.000 | 0.844 |
| descending_canon_Bb_110 | 0.885 | 0.863 | 0.863 | 0/6 | – | 0.000 | 0.851 |
| descending_canon_D_065 | 0.887 | 0.858 | 0.858 | 0/3 | – | 0.000 | 0.844 |
| descending_canon_D_110 | 0.885 | 0.856 | 0.856 | 0/6 | – | 0.000 | 0.844 |
| descending_canon_Fs_065 | 0.887 | 0.858 | 0.858 | 0/3 | – | 0.000 | 0.844 |
| descending_canon_Fs_110 | 0.885 | 0.856 | 0.856 | 0/6 | – | 0.000 | 0.844 |
| four_over_five_Ab_065 | 0.915 | 1.000 | 1.000 | 0/2 | – | 0.000 | 0.906 |
| four_over_five_Ab_110 | 0.921 | 1.000 | 1.000 | 0/4 | – | 0.000 | 0.914 |
| four_over_five_C_065 | 0.915 | 1.000 | 1.000 | 0/2 | – | 0.000 | 0.906 |
| four_over_five_C_110 | 0.914 | 1.000 | 1.000 | 0/4 | – | 0.000 | 0.906 |
| four_over_five_E_065 | 0.915 | 1.000 | 1.000 | 0/2 | – | 0.000 | 0.906 |
| four_over_five_E_110 | 0.914 | 1.000 | 1.000 | 0/4 | – | 0.000 | 0.906 |
| huijia_prechorus_B_065 | 0.932 | 0.816 | 0.809 | 0/7 | – | 0.000 | 0.893 |
| huijia_prechorus_B_110 | 0.935 | 0.824 | 0.817 | 0/7 | – | 0.000 | 0.893 |
| huijia_prechorus_Eb_065 | 0.932 | 0.816 | 0.809 | 0/7 | – | 0.000 | 0.893 |
| huijia_prechorus_Eb_110 | 0.935 | 0.823 | 0.817 | 0/7 | – | 0.000 | 0.893 |
| huijia_prechorus_G_065 | 0.932 | 0.816 | 0.809 | 0/7 | – | 0.000 | 0.893 |
| huijia_prechorus_G_110 | 0.935 | 0.823 | 0.817 | 0/7 | – | 0.000 | 0.893 |
| passing_diminished_A_065 | 0.858 | 1.000 | 1.000 | 1/4 | 1.000 | 0.000 | 0.844 |
| passing_diminished_A_110 | 0.856 | 1.000 | 1.000 | 2/8 | 1.000 | 0.000 | 0.844 |
| passing_diminished_Db_065 | 0.858 | 1.000 | 1.000 | 1/4 | 1.000 | 0.000 | 0.844 |
| passing_diminished_Db_110 | 0.856 | 1.000 | 1.000 | 2/8 | 1.000 | 0.000 | 0.844 |
| passing_diminished_F_065 | 0.858 | 1.000 | 1.000 | 1/4 | 1.000 | 0.000 | 0.844 |
| passing_diminished_F_110 | 0.856 | 1.000 | 1.000 | 2/8 | 1.000 | 0.000 | 0.844 |
| pedal_neighbour_A_065 | 0.915 | 0.915 | 0.915 | 0/2 | – | 0.000 | 1.000 |
| pedal_neighbour_A_110 | 0.914 | 0.914 | 0.914 | 0/4 | – | 0.000 | 1.000 |
| pedal_neighbour_Db_065 | 0.915 | 0.915 | 0.915 | 0/2 | – | 0.000 | 1.000 |
| pedal_neighbour_Db_110 | 0.914 | 0.914 | 0.914 | 0/4 | – | 0.000 | 1.000 |
| pedal_neighbour_F_065 | 0.915 | 0.915 | 0.915 | 0/2 | – | 0.000 | 1.000 |
| pedal_neighbour_F_110 | 0.921 | 0.921 | 0.921 | 0/4 | – | 0.000 | 1.000 |
| rising_m7_Ab_065 | 0.915 | 0.915 | 0.915 | 0/2 | – | 0.000 | 0.906 |
| rising_m7_Ab_110 | 0.914 | 0.914 | 0.914 | 0/4 | – | 0.000 | 0.906 |
| rising_m7_C_065 | 0.915 | 0.915 | 0.915 | 0/2 | – | 0.000 | 0.906 |
| rising_m7_C_110 | 0.914 | 0.914 | 0.914 | 0/4 | – | 0.000 | 0.906 |
| rising_m7_E_065 | 0.915 | 0.915 | 0.915 | 0/2 | – | 0.000 | 0.906 |
| rising_m7_E_110 | 0.921 | 0.921 | 0.921 | 0/4 | – | 0.000 | 0.914 |
| royal_road_Bb_065 | 0.915 | 0.915 | 0.915 | 0/2 | – | 0.000 | 0.969 |
| royal_road_Bb_110 | 0.914 | 0.914 | 0.914 | 0/4 | – | 0.000 | 0.969 |
| royal_road_D_065 | 0.915 | 0.915 | 0.915 | 0/2 | – | 0.000 | 0.969 |
| royal_road_D_110 | 0.914 | 0.914 | 0.914 | 0/4 | – | 0.000 | 0.969 |
| royal_road_Fs_065 | 0.915 | 0.915 | 0.915 | 0/2 | – | 0.000 | 0.969 |
| royal_road_Fs_110 | 0.921 | 0.921 | 0.921 | 0/4 | – | 0.000 | 0.976 |
| secondary_dominant_Bb_065 | 0.915 | 0.915 | 0.915 | 0/2 | – | 0.000 | 0.906 |
| secondary_dominant_Bb_110 | 0.914 | 0.914 | 0.914 | 0/4 | – | 0.000 | 0.906 |
| secondary_dominant_D_065 | 0.915 | 0.915 | 0.915 | 0/2 | – | 0.000 | 0.906 |
| secondary_dominant_D_110 | 0.914 | 0.914 | 0.914 | 0/4 | – | 0.000 | 0.906 |
| secondary_dominant_Fs_065 | 0.915 | 0.915 | 0.915 | 0/2 | – | 0.000 | 0.906 |
| secondary_dominant_Fs_110 | 0.914 | 0.914 | 0.914 | 0/4 | – | 0.000 | 0.906 |
| sus_resolution_B_065 | 0.915 | 1.000 | 1.000 | 1/4 | 1.000 | 0.000 | 0.906 |
| sus_resolution_B_110 | 0.914 | 1.000 | 1.000 | 2/8 | 1.000 | 0.000 | 0.906 |
| sus_resolution_Eb_065 | 0.915 | 1.000 | 1.000 | 1/4 | 1.000 | 0.000 | 0.906 |
| sus_resolution_Eb_110 | 0.914 | 1.000 | 1.000 | 2/8 | 1.000 | 0.000 | 0.906 |
| sus_resolution_G_065 | 0.915 | 1.000 | 1.000 | 1/4 | 1.000 | 0.000 | 0.906 |
| sus_resolution_G_110 | 0.914 | 1.000 | 1.000 | 2/8 | 1.000 | 0.000 | 0.906 |
| walk_up_B_065 | 0.858 | 0.801 | 0.801 | 0/4 | – | 0.000 | 0.781 |
| walk_up_B_110 | 0.856 | 0.799 | 0.799 | 0/8 | – | 0.000 | 0.781 |
| walk_up_Eb_065 | 0.858 | 0.801 | 0.801 | 0/4 | – | 0.000 | 0.781 |
| walk_up_Eb_110 | 0.856 | 0.799 | 0.799 | 0/8 | – | 0.000 | 0.781 |
| walk_up_G_065 | 0.858 | 0.801 | 0.801 | 0/4 | – | 0.000 | 0.781 |
| walk_up_G_110 | 0.856 | 0.799 | 0.799 | 0/8 | – | 0.000 | 0.781 |

## Listen list (missed short chords, majmin_inv, first 20)

| item | t0 | t1 | reference | estimate at mid |
|---|---|---|---|---|
| augmented_passing_Ab_065 | 3.269 | 4.192 | Ab:aug(b7) | Ab:maj |
| augmented_passing_Ab_065 | 11.115 | 11.577 | Ab:aug | Ab:maj |
| augmented_passing_Ab_110 | 2.136 | 2.682 | Ab:aug(b7) | Ab:maj |
| augmented_passing_Ab_110 | 6.773 | 7.045 | Ab:aug | Ab:maj |
| augmented_passing_Ab_110 | 10.864 | 11.409 | Ab:aug(b7) | Ab:maj |
| augmented_passing_Ab_110 | 15.500 | 15.773 | Ab:aug | Ab:maj |
| augmented_passing_C_065 | 3.269 | 4.192 | C:aug(b7) | C:maj |
| augmented_passing_C_065 | 11.115 | 11.577 | C:aug | C:maj |
| augmented_passing_C_110 | 2.136 | 2.682 | C:aug(b7) | C:maj |
| augmented_passing_C_110 | 6.773 | 7.045 | C:aug | C:maj |
| augmented_passing_C_110 | 10.864 | 11.409 | C:aug(b7) | C:maj |
| augmented_passing_C_110 | 15.500 | 15.773 | C:aug | C:maj |
| augmented_passing_E_065 | 3.269 | 4.192 | E:aug(b7) | E:maj |
| augmented_passing_E_065 | 11.115 | 11.577 | E:aug | E:maj |
| augmented_passing_E_110 | 2.136 | 2.682 | E:aug(b7) | E:maj |
| augmented_passing_E_110 | 6.773 | 7.045 | E:aug | E:maj |
| augmented_passing_E_110 | 10.864 | 11.409 | E:aug(b7) | E:maj |
| augmented_passing_E_110 | 15.500 | 15.773 | E:aug | E:maj |
| borrowed_iv_A_065 | 2.346 | 3.269 | G:7 | D:maj7 |
| borrowed_iv_A_065 | 3.269 | 4.192 | D:min6 | D:maj7 |
