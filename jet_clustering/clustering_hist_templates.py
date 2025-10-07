import awkward as ak
from src.hist_tools.object import LorentzVector, Jet
from src.hist_tools import H, Template

class ClusterHists(Template):
    pt        = H((100,  0, 300, ('pt',    "pt [GeV]")))
    pt_l      = H((100,  0, 500, ('pt',    "pt [GeV]")))

    mA        = H((100, 0, 100,  ('mA', "mA [GeV]")))
    mA_l      = H((100, 0, 400,  ('mA', "mA [GeV]")))
    mA_vl     = H((100, 0, 1000, ('mA', "mA [GeV]")))

    mB        = H((100, 0,  60,  ('mB', "mB [GeV]")))
    mB_l      = H((100, 0, 400,  ('mB', "mB [GeV]")))
    mB_vl     = H((100, 0, 600,  ('mB', "mB [GeV]")))

    zA        = H((100,  0.5, 1.3, ('zA', "z fraction")))
    zA_l      = H((100,  0, 1.5, ('zA', "z fraction")))
    zA_vl      = H((100,  -3, 3, ('zA', "z fraction")))

    decay_phi = H((100, -0.1, 3.2, ('decay_phi', "decay angle")))
    thetaA    = H((100,  0, 1.5, ('thetaA',    "theta angle")))

    n         = H((0, 3,             ('n', 'Number')), n=ak.num)

    mA_rot    = H((100, 0, 100,  ('mA_rotated', "mA [GeV]")))
    mB_rot    = H((100, 0, 60,   ('mB_rotated', "mB [GeV]")))


    #
    #  For the PDFS
    #

    mA_pT = H((5, 50, 500, ("pt", "pT")),
              (100, 0, 100,  ('mA', 'mA [GeV]')))

    mB_pT = H((5, 50, 500, ("pt", "pT")),
              (100, 0, 60,  ('mB', 'mB [GeV]')))

    mA_r_pT = H((5, 50, 500, ("pt", "pT")),
                (100, 0, 100,  ('mA_rotated', 'mA [GeV]')))

    mB_r_pT = H((5, 50, 500, ("pt", "pT")),
                (100, 0, 60,  ('mB_rotated', 'mB [GeV]')))


    mA_l_pT = H((5, 50, 500, ("pt", "pT")),
                (100, 0, 400,  ('mA', 'mA [GeV]')))

    mB_l_pT = H((5, 50, 500, ("pt", "pT")),
                (100, 0, 400,  ('mB', 'mB [GeV]')))


    mA_vl_pT = H((5, 50, 500, ("pt", "pT")),
                (100, 0, 1000,  ('mA', 'mA [GeV]')))

    mB_vl_pT = H((5, 50, 500, ("pt", "pT")),
                (100, 0, 600,  ('mB', 'mB [GeV]')))


    decay_phi_pT = H((5, 50, 500, ("pt", "pT")),
                     (50 , -0.1, 3.2, ('decay_phi', "decay angle")))

    zA_vs_thetaA_pT = H((5, 50, 500, ("pt", "pT")),
                        (50,  0.5, 1.3, ('zA', "z fraction")),
                        (50,  0, 1.5, ('thetaA',    "theta angle")))

    zA_l_vs_thetaA_pT = H((5, 50, 500, ("pt", "pT")),
                          (50,  0, 1.5, ('zA', "z fraction")),
                          (50,  0, 1.5, ('thetaA',    "theta angle")))


    rhoA_pT = H((5, 50, 500, ("pt", "pT")),
              (50, 0, 0.5,  ('rhoA', 'rhoA (mass/pt)')))

    rhoB_pT = H((5, 50, 500, ("pt", "pT")),
                (50, 0, 0.5,  ('rhoB', 'rhoB (mass/pt)')))


class ClusterHistsBoosted(Template):
    pt_l      = H((100,  0, 1000, ('pt',    "pt [GeV]")))

    mA        = H((100, 0, 100,  ('mA', "mA [GeV]")))
    mA_l      = H((100, 0, 400,  ('mA', "mA [GeV]")))
    mA_vl     = H((100, 0, 1000, ('mA', "mA [GeV]")))

    mB        = H((100, 0,  60,  ('mB', "mB [GeV]")))
    mB_l      = H((100, 0, 400,  ('mB', "mB [GeV]")))
    mB_vl     = H((100, 0, 600,  ('mB', "mB [GeV]")))

    zA        = H((100,  0.5, 1.3, ('zA', "z fraction")))
    zA_l      = H((100,  0.45, 1.0, ('zA', "z fraction")))
    zA_vl      = H((100,  -3, 3, ('zA', "z fraction")))

    decay_phi = H((100, -0.1, 3.2, ('decay_phi', "decay angle")))
    thetaA    = H((100,  0, 0.6, ('thetaA',    "theta angle")))

    n         = H((0, 3,             ('n', 'Number')), n=ak.num)

    mA_rot    = H((100, 0, 100,  ('mA_rotated', "mA [GeV]")))
    mB_rot    = H((100, 0, 60,   ('mB_rotated', "mB [GeV]")))

    tau21_A     = H((100, -0.2, 1.2, ('part_A.tau21', "tau21 A")))
    tau21_B     = H((100, -0.2, 1.2, ('part_B.tau21', "tau21 B")))


    #
    #  For the PDFS
    #
    pt_binning = (7, 300, 1000, ("pt", "pT"))

    mA_pT = H(pt_binning,
              (100, 0, 100,  ('mA', 'mA [GeV]')))

    mB_pT = H(pt_binning,
              (100, 0, 60,  ('mB', 'mB [GeV]')))

    mA_r_pT = H(pt_binning,
                (100, 0, 200,  ('mA_rotated', 'mA [GeV]')))

    mB_r_pT = H(pt_binning,
                (100, 0, 100,  ('mB_rotated', 'mB [GeV]')))


    mA_l_pT = H(pt_binning,
                (100, 0, 400,  ('mA', 'mA [GeV]')))

    mB_l_pT = H(pt_binning,
                (100, 0, 400,  ('mB', 'mB [GeV]')))


    mA_vl_pT = H(pt_binning,
                (100, 0, 1000,  ('mA', 'mA [GeV]')))

    mB_vl_pT = H(pt_binning,
                (100, 0, 600,  ('mB', 'mB [GeV]')))


    decay_phi_pT = H(pt_binning,
                     (50 , -0.1, 3.2, ('decay_phi', "decay angle")))

    zA_vs_thetaA_pT = H(pt_binning,
                        (50,  0.5, 1.0, ('zA', "z fraction")),
                        (50,    0, 0.6, ('thetaA',    "theta angle")))

    zA_l_vs_thetaA_pT = H(pt_binning,
                          (50,  0.45, 1.0, ('zA', "z fraction")),
                          (150,  0, 0.6, ('thetaA',    "theta angle")))

    rhoA_pT = H(pt_binning,
                (150, 0, 0.5,  ('rhoA', 'rhoA (mass/pt)')))

    rhoB_pT = H(pt_binning,
                (150, 0, 0.5,  ('rhoB', 'rhoB (mass/pt)')))





class ClusterHistsDetailed(ClusterHists):
    dpt_AB        = H((50,  -50, 50, ('dpt_AB',    "pt [GeV]")))
    rpt_A         = H((50,  -0.1, 1.1, ('rpt_A',    "ptA / pt")))
    rpt_B         = H((50,  -0.1, 1.1, ('rpt_B',    "ptB / pt")))

    rpt_AB        = H((50,  -0.1, 1.1, ('rpt_AB',    "ptB / ptA")))
    rpt_AB_l      = H((50,  -0.1, 3.1, ('rpt_AB',    "ptB / ptA")))

    pt_A      = H((100,  0, 300, ('part_A.pt',    "pt [GeV]")))
    eta_A      = H((50,  -3, 3, ('part_A.eta',    "eta A")))
    phi_A      = H((50,  -3.2, 3.2, ('part_A.phi',    "phi A")))

    pt_B      = H((100,  0, 300, ('part_B.pt',    "pt [GeV]")))
    eta_B      = H((50,  -3, 3, ('part_B.eta',    "eta B")))
    phi_B      = H((50,  -3.2, 3.2, ('part_B.phi',    "phi B")))

    zA_pT = H((5, 50, 500, ("pt", "pT")),
              (50, 0.5, 1.3,  ('zA', 'z fraction')))

    thetaA_pT = H((5, 50, 500, ("pt", "pT")),
                  (50, 0.0, 1.5,  ('thetaA', 'theta angle')))

    decay_phi_l = H((100, -3.2, 5, ('decay_phi', "decay angle")))


    pz        = H((100, -500, 500, ('pz',    "pz [GeV]")))
    eta        = H((50,  -3, 3, ('eta',    "eta")))

    massAB       = H((100,  40, 400, ('mass_AB',    "mass [GeV]")))

    rhoA      = H((100, 0, 0.5,  ('rhoA', "rho A (mass/pt)")))
    rhoB      = H((100, 0, 1,  ('rhoB', "rho B (mass/pt)")))

    mA_vs_mB   = H((50, 0, 50, ('part_A.mass', 'Mass A [GeV]')),
                  (50, 0, 50, ('part_B.mass', 'Mass B [GeV]')))

    mA_vs_pTA   = H((50, 0, 50,  ('part_A.mass', 'Mass A [GeV]')),
                    (50, 0, 250, ('part_A.pt', '$p_T$ A [GeV]')))

    rhoA_vs_pTA   = H((50, 0, 1,  ('rhoA', 'rho A')),
                      (50, 0, 250, ('part_A.pt', '$p_T$ A [GeV]')))

    mB_vs_pTB   = H((50, 0, 50,  ('part_B.mass', 'Mass B [GeV]')),
                    (50, 0, 250, ('part_B.pt', '$p_T$ B [GeV]')))

    rhoB_vs_pTB   = H((50, 0, 1,  ('rhoB', 'rho B')),
                      (50, 0, 250, ('part_B.pt', '$p_T$ B [GeV]')))

    drAB      = H((100, 0, 5,   ('dr_AB', "$\Delta$ R AB")))
    tan_thetaA    = H((100,  0, 10, ('tan_thetaA',    "tan (theta angle)")))


    mA_vs_thetaA = H((50,  0, 100, ('mA', "mA [GeV]")),
                     (50,  0, 0.5, ('thetaA',    "theta angle")))

    mB_vs_thetaA = H((50,  0, 50, ('mB', "mB [GeV]")),
                     (50,  0, 0.5, ('thetaA',    "theta angle")))

    rhoA_vs_thetaA = H((50,  0, 0.5, ('rhoA', "rhoA")),
                       (50,  0, 0.5, ('thetaA',    "theta angle")))

    rhoB_vs_thetaA = H((50,  0, 0.5, ('rhoB', "rhoB ")),
                       (50,  0, 0.5, ('thetaA',    "theta angle")))




    zA_vs_thetaA = H((50,  0.5, 1.5, ('zA', "z fraction")),
                     (50,  0, 1.5, ('thetaA',    "theta angle")))

    zA_l_vs_thetaA = H((50,  0, 1.5, ('zA', "z fraction")),
                       (50,  0, 1.5, ('thetaA',    "theta angle")))



    zA_vs_decay_phi = H((50,  0.5, 1.5, ('zA', "z fraction")),
                     (50,  -0.1, 3.2, ('decay_phi',    "decay angle")))

    thetaA_vs_decay_phi = H((50,  0, 1.5, ('thetaA',    "theta angle")),
                            (50,  -0.1, 3.2, ('decay_phi',    "decay angle")))


    zA_vs_pT = H((50,  0.5, 2, ('zA', "z fraction")),
                 (50,  50, 300, ('pt',    "pt")))

    thetaA_vs_pT = H((50,  0, 1.5, ('thetaA',    "theta angle")),
                     (50,  50, 300, ('pt',    "pt")))


    decay_phi_vs_pT = H((50 , -0.1, 3.2, ('decay_phi', "decay angle")),
                        (100,  50, 300, ('pt', "pT")))


    mA_eta = H((5, 0, 3, ("abs_eta", "eta")),
                (50, 0, 50,  ('mA', 'mA [GeV]')))

    mB_eta = H((5, 0, 3, ("abs_eta", "eta")),
                (50, 0, 50,  ('mB', 'mB [GeV]')))


    decay_phi_eta = H((5, 0, 3, ("abs_eta", "eta")),
                     (50 , -0.1, 3.2, ('decay_phi', "decay angle")))


    zA_vs_thetaA_eta = H((5, 0, 3, ("abs_eta", "eta")),
                         (50,  0.5, 1.3, ('zA', "z fraction")),
                         (50,  0, 1.5, ('thetaA',    "theta angle")))


def ClusterHistsDetailedBoosted( config_pair, name ):
    return ClusterHistsDetailed( config_pair, name, bins={"pt":   (50, 200, 1000),
                                                          "part_A.pt": (50,   0, 1000),
                                                          "dr_AB": (100,   0, 1),
                                                          "rhoB": (100,   0, 0.5),
                                                          "zA_l": (100,   0.45, 1.0),
                                                          }
                                )
