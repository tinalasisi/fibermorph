#%% Imports
import shutil
import os

# %% delete cache

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
cache = "./results_cache"

print("Deleting {}".format(os.path.abspath(cache)))
shutil.rmtree(cache)

