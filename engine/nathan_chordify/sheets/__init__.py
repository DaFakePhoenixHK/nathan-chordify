"""Find a song's chord sheet online, line it up with the recording, and use it to refine the chords.

Modules: query (title -> search terms), fetch (polite cached fetcher), parsers (pages -> bar-text
charts), sources (per-site search + chart fetch), align (chart -> recording, 12 transpositions,
verdict), guide (chart-guided re-decode / apply as written), lookup (the whole flow).
"""
