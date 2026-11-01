from __future__ import print_function
import sys
import os

try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    HAS_ROOT = True
except ImportError:
    ROOT = None
    HAS_ROOT = False

try:
    import scipy.stats
    HAS_SCIPY = True
except ImportError:
    scipy = None
    HAS_SCIPY = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    matplotlib.use('Agg')
    HAS_MPL = True
except ImportError:
    matplotlib = None
    plt = None
    Ellipse = None
    HAS_MPL = False

sys.path.insert(0, os.getcwd())
try:
    import coffea4bees.plots.ROOTPlotTools as ROOTPlotTools
except ImportError:
    ROOTPlotTools = None

try:
    from coffea4bees.stats_analysis.make_variable_binning import make_variable_binning, rebin_histogram
except ImportError:
    make_variable_binning = None
    rebin_histogram = None

CMURED = '#d34031'
# https://xkcd.com/color/rgb/
COLORS = ['xkcd:purple', 'xkcd:green', 'xkcd:blue', 'xkcd:teal', 'xkcd:orange', 'xkcd:cherry', 'xkcd:bright red',
          'xkcd:pine', 'xkcd:magenta', 'xkcd:cerulean', 'xkcd:eggplant', 'xkcd:coral', 'xkcd:blue purple',
          'xkcd:tea', 'xkcd:burple', 'xkcd:deep aqua', 'xkcd:orange pink', 'xkcd:terracota']

color_multijet = '(1.0, 0.8745, 0.4980)'
color_TTbar    = '(0.5216, 0.8196, 0.9843)'

regionName = {
    'SB': 'Sideband',
    'CR': 'Control Region',
    'SR': 'Signal Region',
    'notSR': 'Sideband',
    'SRNoHH': 'Signal Region (Veto HH)',
}

# Runtime execution context
args = None
channel = None
classifier = None
rebin = None
rebin_label = None
output_dir = None
closure_file_out = None
closure_file_out_pkl = None
closure_file_out_log = None
log_file = None
lumi = None
closure_fit_x_min = 0
maxBasisEnsemble = 5
maxBasisClosure = 5
ttAverage = False
doSpuriousSignal = True
dataAverage = True
nMixes = 15
probThreshold = 0.05
mixes = []
BEs = []
BE = []

def set_args(new_args):
    global args
    args = new_args

def set_log_file(f):
    global log_file
    log_file = f
