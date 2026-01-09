
def additional_poi(analysis_key):
    """Generate string to define all othersignal POIs"""
    othersignal = config['channels'][analysis_key]["othersignal"]
    if othersignal.strip():
        return " ".join([f"--PO 'map=.*/{sig}:r{sig}[1,-10,10]'" for sig in othersignal.split()])
    return ""

def set_parameters(analysis_key, value=0):
    """Generate string to set all othersignal POIs to zero for a given analysis"""
    othersignal = config['channels'][analysis_key]["othersignal"]
    if othersignal.strip():
        return "--setParameters " + ",".join([f"r{sig}={value}" for sig in othersignal.split()]) 
    return ""

def set_parameters_ranges(analysis_key):
    """Generate string to set all othersignal POIs to zero for a given analysis"""
    othersignal = config['channels'][analysis_key]["othersignal"]
    if othersignal.strip():
        return ":" + ":".join([f"r{sig}=0,0" for sig in othersignal.split()]) 
    return ""

def freeze_parameters(analysis_key):
    """Generate string to set all othersignal POIs to zero for a given analysis"""
    othersignal = config['channels'][analysis_key]["othersignal"]
    if othersignal.strip():
        return "--freezeParameters " + ",".join([f"r{sig}" for sig in othersignal.split()])
    return ""