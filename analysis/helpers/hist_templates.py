from src.hist_tools.object import LorentzVector, Jet
from src.hist_tools import H, Template
import numpy as np

class SvBHists(Template):
    ps      = H((50, 0, 1, ('ps', "Regressed P(Signal)")))
    ptt     = H((50, 0, 1, ('ptt', "Regressed P(tT)")))

    tt_vs_mj     = H((50, 0, 1, ('tt_vs_mj', "P(tT) | Background")))

    ps_zz   = H((25, 0, 1, ('ps_zz', "Regressed P(Signal) $|$ P(ZZ) is largest ")))
    ps_zh   = H((20, 0, 1, ('ps_zh', "Regressed P(Signal) $|$ P(ZH) is largest ")))

    ### var_binning makes the Run2 SvB_MA signal distribution flat
    var_binning = np.array([0.        , 0.17276639, 0.26010802, 0.32549336, 0.38053438,
       0.42957123, 0.47136053, 0.51007601, 0.54459632, 0.57495467,
       0.60259078, 0.62742396, 0.64944198, 0.67054542, 0.68904503,
       0.70681051, 0.72300105, 0.73822085, 0.75198387, 0.76605212,
       0.7796761 , 0.79188894, 0.80312279, 0.81341206, 0.82374613,
       0.83389092, 0.84299264, 0.85179326, 0.86086487, 0.86925629,
       0.87753836, 0.8851288 , 0.89212982, 0.89898318, 0.90569564,
       0.91213127, 0.91841945, 0.92447081, 0.93053227, 0.93653864,
       0.94229502, 0.94825389, 0.95395487, 0.95998911, 0.96638473,
       0.97275653, 0.98      , 1.        ])
    ps_hh   = H((var_binning, ('ps_hh', "Regressed P(Signal) $|$ P(HH) is largest ")))

    ps_zz_fine   = H((240, 0, 1, ('ps_zz', "Regressed P(Signal) $|$ P(ZZ) is largest ")))
    ps_zh_fine   = H((240, 0, 1, ('ps_zh', "Regressed P(Signal) $|$ P(ZH) is largest ")))
    ps_hh_fine   = H((240, 0, 1, ('ps_hh', "Regressed P(Signal) $|$ P(HH) is largest ")))

    phh_hh_fine   = H((240, 0, 1, ('phh_hh', "Regressed P(HH) $|$ P(HH) is largest ")))
    phh_fine      = H((240, 0, 1, ('phh', "Regressed P(HH)  ")))


class ttHbbSvBHists(Template):
    ps      = H((50, 0.01, 1, ('ps', "Regressed P(Signal)")))
    ptt     = H((50, 0, 1, ('ptt', "Regressed P(tT)")))
    tt_vs_mj     = H((50, 0, 1, ('tt_vs_mj', "P(tT) | Background")))

    ### var_binning_ps_ttHbb defines the 240 quantile intervals (1.0 -> 0.01) for signal
    var_binning_ps_ttHbb = np.array([
        0.010000, 0.017529, 0.024765, 0.032247, 0.039473, 0.046866, 0.054177, 0.061489, 0.068816,
        0.076134, 0.083471, 0.091210, 0.098885, 0.106620, 0.114323, 0.122037, 0.129781,
        0.137750, 0.145330, 0.153154, 0.160982, 0.168893, 0.176586, 0.184492, 0.192197,
        0.199633, 0.207286, 0.215085, 0.222669, 0.230570, 0.238377, 0.245930, 0.253616,
        0.260955, 0.268646, 0.276078, 0.283359, 0.290635, 0.297854, 0.305091, 0.312336,
        0.319486, 0.326774, 0.333938, 0.341052, 0.348357, 0.355417, 0.362446, 0.369581,
        0.376585, 0.383457, 0.390221, 0.396851, 0.403590, 0.410307, 0.416743, 0.423252,
        0.429584, 0.435874, 0.442245, 0.448397, 0.454612, 0.460691, 0.466952, 0.472902,
        0.478882, 0.485101, 0.490950, 0.496879, 0.502491, 0.508027, 0.513923, 0.519332,
        0.524708, 0.530167, 0.535436, 0.540807, 0.545979, 0.551233, 0.556472, 0.561499,
        0.566570, 0.571576, 0.576431, 0.581221, 0.586021, 0.590712, 0.595412, 0.600003,
        0.604727, 0.609273, 0.613702, 0.618097, 0.622544, 0.626778, 0.631040, 0.635300,
        0.639443, 0.643563, 0.647638, 0.651577, 0.655694, 0.659659, 0.663704, 0.667571,
        0.671440, 0.675238, 0.678990, 0.682649, 0.686224, 0.689906, 0.693531, 0.696911,
        0.700500, 0.703971, 0.707357, 0.710775, 0.714165, 0.717382, 0.720627, 0.723883,
        0.727127, 0.730403, 0.733600, 0.736706, 0.739755, 0.742780, 0.745814, 0.748699,
        0.751666, 0.754607, 0.757431, 0.760326, 0.763183, 0.765978, 0.768759, 0.771496,
        0.774145, 0.776757, 0.779408, 0.781894, 0.784465, 0.787070, 0.789621, 0.792093,
        0.794607, 0.797133, 0.799677, 0.802036, 0.804465, 0.806805, 0.809126, 0.811425,
        0.813701, 0.815940, 0.818101, 0.820289, 0.822453, 0.824632, 0.826771, 0.828888,
        0.830979, 0.833030, 0.835113, 0.837154, 0.839155, 0.841115, 0.843120, 0.845101,
        0.847077, 0.849004, 0.850942, 0.852860, 0.854720, 0.856629, 0.858518, 0.860287,
        0.862101, 0.863875, 0.865657, 0.867475, 0.869266, 0.871025, 0.872751, 0.874501,
        0.876148, 0.877919, 0.879579, 0.881227, 0.882816, 0.884551, 0.886193, 0.887831,
        0.889421, 0.891028, 0.892624, 0.894225, 0.895783, 0.897370, 0.898947, 0.900497,
        0.902011, 0.903595, 0.905160, 0.906728, 0.908276, 0.909849, 0.911434, 0.913007,
        0.914506, 0.916019, 0.917607, 0.919146, 0.920716, 0.922261, 0.923828, 0.925393,
        0.926951, 0.928515, 0.930132, 0.931674, 0.933311, 0.934967, 0.936640, 0.938311,
        0.940025, 0.941757, 0.943559, 0.945363, 0.947212, 0.949136, 0.951156, 0.953290,
        0.955459, 0.957721, 0.960348, 0.963245, 0.966371, 0.970208, 0.975576, 1.000000
    ])
    ps_ttHbb = H((240, 0, 1, ('ps_ttHbb', "Cumulative Signal Quantile (240 bins)")))


class DijetSvBHists(Template):
    lead_m         = H((50, 0, 250, ("lead_m", 'Lead DiJet Mass (SvB > 0.8) [GeV]')))
    subl_m         = H((50, 0, 250, ("subl_m", 'Subl DiJet Mass (SvB > 0.8) [GeV]')))
    lead_vs_subl_m = H((50, 0, 250, ('lead_m', 'Lead DiJet Mass [GeV]')),
                       (50, 0, 250, ('subl_m', 'Subl DiJet Mass [GeV]')))


class FeynNetSvBHists(Template):
    p_ggHH_vs_bkg  = H((50, 0, 1, ('p_ggHH_vs_bkg',  "FeynNet P(ggF vs bkg)")))
    # Same granularity as SvB_MA.ps_hh_fine: the score piles up near 1, and the 50-bin version
    # cannot resolve its top 0.02, which holds most of the signal. Combine input.
    p_ggHH_vs_bkg_fine = H((240, 0, 1, ('p_ggHH_vs_bkg', "FeynNet P(ggF vs bkg)")))
    p_qqHH_vs_bkg  = H((50, 0, 1, ('p_qqHH_vs_bkg',  "FeynNet P(VBF vs bkg)")))
    p_ZZ_vs_bkg    = H((50, 0, 1, ('p_ZZ_vs_bkg',    "FeynNet P(ZZ vs bkg)")))
    p_ZH_vs_bkg    = H((50, 0, 1, ('p_ZH_vs_bkg',    "FeynNet P(ZH vs bkg)")))



class FvTHists(Template):
    FvT  = H((50, 0, 5, ('FvT', 'FvT reweight')))
    FvT_l = H((50, 0, 50, ('FvT', 'FvT reweight')))
    pd4  = H((50, 0, 1, ("pd4",   'FvT Regressed P(Four-tag Data)')))
    pd3  = H((50, 0, 1, ("pd3",   'FvT Regressed P(Three-tag Data)')))
    pt4  = H((50, 0, 1, ("pt4",   'FvT Regressed P(Four-tag t#bar{t})')))
    pt3  = H((50, 0, 1, ("pt3",   'FvT Regressed P(Three-tag t#bar{t})')))
    pm4  = H((50, 0, 1, ("pm4",   'FvT Regressed P(Four-tag Multijet)')))
    pm3  = H((50, 0, 1, ("pm3",   'FvT Regressed P(Three-tag Multijet)')))
    pt   = H((50, 0, 1, ("pt",    'FvT Regressed P(t#bar{t})')))
    std  = H((50, 0, 3, ("std",   'FvT Standard Deviation')))
    frac_err = H((50, 0, 5, ("frac_err",  'FvT std/FvT')))
    #'q_1234', 'q_1324', 'q_1423',
    d3_to_t3     = H((50, 0, 1, ('d3_to_t3', "P(tT 3b) | Data 3b")))
    d4_to_t4     = H((50, 0, 1, ('d4_to_t4', "P(tT 4b) | Data 4b")))
    d3_to_t4     = H((50, 0, 1, ('d3_to_t4', "P(tT 4b) | Data 3b")))


class MvDHists(Template):
    MvD  = H((50, 0, 5, ('MvD', 'MvD reweight')))
    MvD_l = H((50, 0, 50, ('MvD', 'MvD reweight')))
    # pd4  = H((50, 0, 1, ("pd4",   'MvD Regressed P(Four-tag Data)')))
    # pmix4  = H((50, 0, 1, ("pmix4",   'MvD Regressed P(mix 4b)')))
    # pt4  = H((50, 0, 1, ("pt4",   'MvD Regressed P(Four-tag t#bar{t})')))
    # pm4  = H((50, 0, 1, ("pm4",   'MvD Regressed P(Four-tag Multijet)')))
    #
    # # frac_err = H((50, 0, 5, ("frac_err",  'MvD std/MvD')))
    # #'q_1234', 'q_1324', 'q_1423',
    # mix4_to_t4     = H((50, 0, 1, ('mix4_to_t4', "P(tT 4b) | mix 4b")))


class QuadJetHistsBasic(Template):
    dr              = H((50,     0, 5,   ("dr",          'Diboson Candidate $\\Delta$R(d,d)')))
    dphi            = H((50, -3.2, 3.2, ("dphi",        'Diboson Candidate $\\Delta$R(d,d)')))
    deta            = H((50,   -5, 5,   ("deta",        'Diboson Candidate $\\Delta$R(d,d)')))
    xZZ             = H((50, 0, 10,     ("xZZ",         'Diboson Candidate zZZ')))
    xZH             = H((50, 0, 10,     ("xZH",         'Diboson Candidate zZH')))
    xHH             = H((50, 0, 10,     ("xHH",         'Diboson Candidate zHH')))

    lead_vs_subl_m   = H((100, 0, 1000, ('lead.mass', 'Lead Boson Candidate Mass')),
                         (100, 0, 1000, ('subl.mass', 'Subl Boson Candidate Mass')))

    close_vs_other_m = H((100, 0, 1000, ('close.mass', 'Close Boson Candidate Mass')),
                         (100, 0, 1000, ('other.mass', 'Other Boson Candidate Mass')))

class QuadJetHistsSelected(QuadJetHistsBasic):

    lead            = LorentzVector.plot_pair(('...', R'Lead Boson Candidate'),  'lead',  skip=['n'], bins={"pt": (50, 0, 1000)})
    subl            = LorentzVector.plot_pair(('...', R'Subl Boson Candidate'),  'subl',  skip=['n'], bins={"pt": (50, 0, 1000)})

class QuadJetHistsMinDr(QuadJetHistsBasic):
    close           = LorentzVector.plot_pair(('...', R'Close Boson Candidate'), 'close', skip=['n'], bins={"pt": (50, 0, 1000)})
    other           = LorentzVector.plot_pair(('...', R'Other Boson Candidate'), 'other', skip=['n'], bins={"pt": (50, 0, 1000)})

class QuadJetHistsUnsup(Template):
    dr              = H((50,     0, 5,   ("dr",          'Diboson Candidate $\\Delta$R(d,d)')))
    dphi            = H((100, -3.2, 3.2, ("dphi",        'Diboson Candidate $\\Delta$R(d,d)')))
    deta            = H((100,   -5, 5,   ("deta",        'Diboson Candidate $\\Delta$R(d,d)')))

    lead_vs_subl_m   = H((50, 0, 250, ('lead.mass', 'Lead Boson Candidate Mass')),
                         (50, 0, 250, ('subl.mass', 'Subl Boson Candidate Mass')))

    close_vs_other_m = H((50, 0, 250, ('close.mass', 'Close Boson Candidate Mass')),
                         (50, 0, 250, ('other.mass', 'Other Boson Candidate Mass')))

    lead            = LorentzVector.plot_pair(('...', R'Lead Boson Candidate'),  'lead',  skip=['n'])
    subl            = LorentzVector.plot_pair(('...', R'Subl Boson Candidate'),  'subl',  skip=['n'])
    close           = LorentzVector.plot_pair(('...', R'Close Boson Candidate'), 'close', skip=['n'])
    other           = LorentzVector.plot_pair(('...', R'Other Boson Candidate'), 'other', skip=['n'])

class QuadJetHistsSRSingle(Template):
    lead_m           = H((50, 0, 250, ("lead_m",        'Lead Boson Candidate Mass')))
    subl_m           = H((50, 0, 250, ("subl_m",        'Subl Boson Candidate Mass')))
    lead_vs_subl_m   = H((50, 0, 250, ('lead_m', 'Lead Boson Candidate Mass')),
                         (50, 0, 250, ('subl_m', 'Subl Boson Candidate Mass')))

class WCandHists(Template):

    p  = LorentzVector.plot(('...', R'W Candidate'), 'p',  skip=['n'], bins={"mass": (60, 0, 600), "pt": (60, 0, 600)})
    pW = LorentzVector.plot(('...', R'W Candidate'), 'pW', skip=['n'], bins={"mass": (60, 0, 600), "pt": (60, 0, 600)})

    j = Jet.plot(('...', R'W j jet Candidate'), 'j',     skip=['deepjet_c','n'], bins={"mass": (50, 0, 100)})
    l = Jet.plot(('...', R'W l jet Candidate'), 'l',     skip=['deepjet_c','n'], bins={"mass": (50, 0, 100)})

class TopCandHists(Template):

    t = LorentzVector.plot(('...', R'Top Candidate'), 'p', skip=['n'], bins={"mass": (80, 0, 800), "pt": (50, 0, 1000)})
    b = Jet.plot(('...', R'Top b jet Candidate'), 'b', skip=['deepjet_c','n'], bins={"mass": (50, 0, 100)})
    W = WCandHists(('...', R'W boson Candidate'), 'W')

    mbW  = H(( 50, 80, 280,   ("mbW",  'm_{b,W}')))
    xWt  = H(( 24, 0,  6,   ("xWt",  "X_{W,t}")))
    xWbW = H(( 24, 0,  6,   ("xWbW", "X_{W,bW}")))
    rWbW = H(( 24, 0,  6,   ("rWbW", "r_{W,bW}")))
    xbW  = H(( 60, -15,  15,   ("xbW",  "X_{W,bW}")))
    xW   = H(( 24, 0,  6,   ("xW",   'X_{W}')))

    mW_vs_mt  = H((50,  0, 250, ('W.p.mass', 'W Candidate Mass [GeV]')),
                  (50, 80, 280, ('p.mass',   'Top Candidate Mass [GeV]')))

    mW_vs_mbW = H((50,  0, 250, ('W.p.mass', 'W Candidate Mass [GeV]')),
                  (50, 80, 280, ('mbW',   'm_{b,W} [GeV]')))

    xW_vs_xt  = H((24, 0,  6,   ("xW",   'X_{W}')),
                  (24, 0,  6,   ("xt",   'X_{t}')))

    xW_vs_xbW  = H((24, 0,  6,   ("xW",   'X_{W}')),
                   (24, 0,  6,   ("xbW",  'X_{bW}')))

class TrigEmHists(Template):
    pfjetht      = H((50, 0, 1500, ('pfjetht',   "h_{T} [GeV]")))
    calojetht    = H((50, 0, 1500, ('calojetht', "h_{T} [GeV]")))

    pt1    = H((60, 0, 300, ('pt1', "Jet 1 p_{T} [GeV]")))
    pt2    = H((60, 0, 300, ('pt2', "Jet 2 p_{T} [GeV]")))
    pt3    = H((60, 0, 300, ('pt3', "Jet 3 p_{T} [GeV]")))
    pt4    = H((60, 0, 300, ('pt4', "Jet 4 p_{T} [GeV]")))

    btagTMean = H((50, 0, 5, ('btagTMean', 'mean Top 2 btagScores')))
