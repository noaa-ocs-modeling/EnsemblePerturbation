import os

from matplotlib import pyplot
import numpy as np


def build_pc_expansion(
    x_filename: os.PathLike = 'xdata.dat',
    y_filename: os.PathLike = 'qoi.dat',
    output_filename: str = None,
    pc_type: str = 'HG',
    poly_order: int = 5,
    lambda_regularization: int = 0,
):
    """
    builds the polynomial-chaos expansion

    regression function inputs: 
    -x "xdata filename" 
    -y "ydata filename" 
    -s "PCtype" 
    -o "Polynomial order"  
    -l "regularization lambda"
    """
    uqtk_cmd = f'regression -x {x_filename} -y {y_filename} -s {pc_type} -o {poly_order} -l {lambda_regularization} >& regr.log'
    os.system(uqtk_cmd)
    pc_coefficients = np.loadtxt('coeff.dat')
    if output_filename is not None:
        os.rename('coeff.dat', output_filename)

    return pc_coefficients


def evaluate_pc_expansion(
    x_filename: os.PathLike = 'xdata.dat',
    parameter_filename: os.PathLike = 'coeff.dat',
    output_filename: os.PathLike = None,
    pc_type: str = 'HG',
    poly_order: int = 5,
):
    """
    evaluates the polynomial-chaos expansion
    
    pce_eval function inputs: 
    -x "xdata filename" 
    -f "parameter filename" 
    -s "PC type" 
    -o "Polynomial order"  
    """
    assert x_filename == 'xdata.dat', 'x_filename needs to be xdata.dat'
    uqtk_cmd = f'pce_eval -f {parameter_filename} -s {pc_type} -o {poly_order} >& pc_eval.log'
    os.system(uqtk_cmd)
    qoi_pc = np.loadtxt('ydata.dat')
    if output_filename is not None:
        os.rename('ydata.dat', output_filename)
    return qoi_pc


def evaluate_pc_multiindex(
    multiindex_filename: os.PathLike = 'mindex.dat',
    multiindex_type: str = 'TO',
    poly_order: int = 3,
    pc_dimension: int = 1,
):
    """
    evaluates the multi-index of the PC expansion
    
    gen_mi function inputs (generates the multi-index files): 
    -x "Multiindex type" 
    -p "PC polynomial order" 
    -q "PC dimension (number of parameters)" 
    """

    # evaluate the multi-index file
    uqtk_cmd = f'gen_mi -x {multiindex_type} -p {poly_order} -q {pc_dimension} >& mi.log'
    os.system(uqtk_cmd)
    multiindex_filename = 'mindex.dat'


def evaluate_pc_sensitivity(
    parameter_filename: os.PathLike = 'coeff.dat',
    multiindex_filename: os.PathLike = 'mindex.dat',
    pc_type: str = 'HG',
):
    """
    evaluates the Sobol sensitivities of the PC expansion
    
    pce_sens function inputs (generates the Sobol sensitivity indices): 
    -x "PC type" 
    -f "PC coefficient filename"
    -m "Multiindex file"
    """

    # evaluating the sensitivities
    uqtk_cmd = (
        f'pce_sens -f {parameter_filename} -x {pc_type} -m {multiindex_filename} >& sens.log'
    )
    os.system(uqtk_cmd)
    sensitivities = {
        'main': np.loadtxt('mainsens.dat'),
        'joint': np.loadtxt('jointsens.dat'),
        'total': np.loadtxt('totsens.dat'),
    }
    return sensitivities


def evaluate_pc_distribution_function(
    parameter_filename: os.PathLike = 'coeff.dat',
    multiindex_filename: os.PathLike = 'mindex.dat',
    pc_type: str = 'HG',
    multiindex_type: str = 'TO',
    poly_order: int = 3,
    pc_dimension: int = 1,
    num_samples: int = 1000,
    pdf_bins: int = 100,
    custom_xlabel: str = None,
    figname: str = None,
):
    """
    evaluates the PDF & CDF of the surrogate output
    
    pce_rv function inputs (generates the random variable): 
    -x "PC type" 
    -n "num samples" 
    -o "PC polynomial order" 
    -p "PC dimension (number of parameters)" 
    -f "PC coefficient filename"
    -m "multi-index filename"
    -w "type of random variable"
    
    pdf_cl function inputs (generates the pdf): 
    -i "input random variable filename" (rvar.dat from pce_rv) 
    -g "number of bins in the pdf" 
    """

    # evaluating the PC-related random variables
    uqtk_cmd = f'pce_rv -x {pc_type} -n {num_samples} -o {poly_order} -p {pc_dimension} -f {parameter_filename} -m {multiindex_filename} -w PCmi >& pc_rv.log'
    os.system(uqtk_cmd)

    # evaluating the PDF of the PC expansion
    uqtk_cmd = f'pdf_cl -i rvar.dat -g {pdf_bins} >& pdf.log'
    os.system(uqtk_cmd)
    xtarget = np.loadtxt('dens.dat')[:, :-1].squeeze()
    pdf = np.loadtxt('dens.dat')[:, -1:].squeeze()
    cdf = np.cumsum(pdf) * np.diff(xtarget)[0]

    if figname is not None:
        pyplot.figure(figsize=(12, 8))

        pyplot.plot(xtarget, pdf)

        if custom_xlabel is not None:
            pyplot.xlabel(custom_xlabel)

        pyplot.ylabel('PDF')

        pyplot.suptitle(figname)
        output_filename = f'{figname}.png'
        pyplot.savefig(output_filename, bbox_inches='tight')

    # entering the PDF/CDF into a dictionary
    distribution_dict = {
        'x': xtarget,
        'pdf': pdf,
        'cdf': cdf,
    }
    return distribution_dict


def evaluate_pc_exceedance_heights(exceedance_probabilities: np.ndarray, pc_dict: dict):
    """
    Get the heights at the desired exceedance probabilities
    
    """

    return np.interp(1.0 - exceedance_probabilities, pc_dict['cdf'], pc_dict['x'])


def evaluate_pc_exceedance_probabilities(exceedance_heights: np.ndarray, pc_dict: dict):
    """
    Get the probabilities of exceedance above desired heights
    
    """

    return 1.0 - np.interp(exceedance_heights, pc_dict['x'], pc_dict['cdf'])
