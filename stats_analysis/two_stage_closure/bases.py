from __future__ import print_function
from .config import ROOT, HAS_ROOT

def make_basis_elements(basis_type='fourier'):
    elements = []
    if basis_type == 'fourier':
        elements = ['1']
        for k in range(1, 11):
            elements.append(f'sin({k}*pi*x)')
            elements.append(f'cos({k}*pi*x)')
    elif basis_type == 'bernstein':
        import math
        degree = 10
        elements = ['1']
        for i in range(1, degree + 1):
            c = math.comb(degree, i)
            term = f'{c}'
            term += f' * (x^{i})' if i > 1 else ' * x'
            if degree - i > 0:
                p = degree - i
                term += f' * ((1-x)^{p})' if p > 1 else ' * (1-x)'
            elements.append(term)
    else:
        raise ValueError(f"Unknown basis_type: {basis_type}")

    tf1_list = []
    if HAS_ROOT:
        for i, s in enumerate(elements):
            tf1_list.append(ROOT.TF1(f'BE{i}', s, 0, 1))
    return elements, tf1_list

BEs, BE = make_basis_elements('fourier')
