from __future__ import print_function
from .config import scipy, HAS_SCIPY

def pearsonr(x, y, n=None):
    r, p_raw = scipy.stats.pearsonr(x, y)
    if n is None:
        return (r, p_raw)
    dist = scipy.stats.beta(n / 2. - 1, n / 2. - 1, loc=-1, scale=2)
    p_cor = 2 * dist.cdf(-abs(r))
    return (r, p_cor)


def fTest(chi2_1, chi2_2, ndf_1, ndf_2):
    print(f'chi2_1, chi2_2, ndf_1, ndf_2 = {chi2_1}, {chi2_2}, {ndf_1}, {ndf_2}')
    d1 = (ndf_1 - ndf_2)
    d2 = ndf_2
    print(f'd1, d2 = {d1}, {d2}')
    if d1 <= 0 or d2 <= 0:
        print("Warning: invalid NDF for fTest (d1 <= 0 or d2 <= 0), returning 1.0")
        return 1.0
    N = (chi2_1 - chi2_2) / d1
    D = chi2_2 / d2
    print(f'N, D = {N}, {D}')
    if D <= 0:
        return 1.0
    fStat = N / D
    fProb = scipy.stats.f.cdf(fStat, d1, d2)
    expectedFStat = scipy.stats.distributions.f.isf(0.05, d1, d2)
    print('    f(%i,%i) = %f (expected at 95%%: %f)' % (d1, d2, fStat, expectedFStat))
    print('f.cdf(%i,%i) = %3.0f%%' % (d1, d2, 100 * fProb))
    print()
    return fProb
