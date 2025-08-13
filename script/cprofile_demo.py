import cProfile
import pstats

# The function you want to test
def some_function():
    # ... complex and slow operations ...
    sum([i*i for i in range(100000)])

# Create a Profile object
profiler = cProfile.Profile()

# --- Start profiling ---
profiler.enable()

some_function() # Call the code you want to measure

# --- Stop profiling ---
profiler.disable()

# Print the stats
# You can sort by 'cumulative', 'tottime', 'calls', etc.
# The '10' limits the output to the top 10 lines.
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(100)