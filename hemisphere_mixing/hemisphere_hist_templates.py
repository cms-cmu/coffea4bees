import awkward as ak
from src.hist_tools.object import LorentzVector, Jet
from src.hist_tools import H, Template

class HemisphereHists(Template):

    pz            = H((100, -1000, 1000, ('pz', 'pz [GeV]')))
    sumPt_T       = H((100, 0,1000, ('sumPt_T', 'Sum Pt along Thrust Axis [GeV]')))
    sumPt_T_minor = H((100, 0, 500, ('sumPt_T_minor', 'Sum Pt perpendicular to Thrust Axis [GeV]')))
    combinedMass  = H((100,  0, 500, ('combinedMass', 'Combined Mass [GeV]')))


    #  hPz_sig       = thisDir.make<TH1F>("Pz_sig",     (m_name+"/Pz_sig; ;Entries").c_str(),     100,-10,10);
    #  hSumPt_T_sig  = thisDir.make<TH1F>("SumPt_T_sig",     (m_name+"/SumPt_T_sig; ;Entries").c_str(),     100,-0.1,10);
    #  hSumPt_Ta_sig = thisDir.make<TH1F>("SumPt_Ta_sig",     (m_name+"/SumPt_Ta_sig; ;Entries").c_str(),     100,-0.1,10);
    #  hCombMass_sig = thisDir.make<TH1F>("CombMass_sig",     (m_name+"/CombMass_sig; ;Entries").c_str(),     100,-0.1,10);
    #  heventWeight  = thisDir.make<TH1F>("eventWeight",     (m_name+"/eventWeight; ;Entries").c_str(),     100,-0.1,1.1);
