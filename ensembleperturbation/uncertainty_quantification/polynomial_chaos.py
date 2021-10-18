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
    uqtk_cmd = f'regression -x {x_filename} -y {y_filename} -s {pc_type} -o {poly_order} -l {lambda_regularization}'
    os.system(uqtk_cmd)
    if output_filename is not None:
        os.rename('coeff.dat', output_filename)
    return


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
    uqtk_cmd = f'pce_eval -f {parameter_filename} -s {pc_type} -o {poly_order}'
    os.system(uqtk_cmd)
    qoi_pc = np.loadtxt('ydata.dat')
    if output_filename is not None:
        os.rename('ydata.dat', output_filename)
    return qoi_pc


def evaluate_pc_sensitivity(
    parameter_filename: os.PathLike = 'coeff.dat',
    pc_type: str = 'HG',
    multiindex_type: str = 'TO',
    poly_order: int = 3,
    pc_dimension: int = 1,
):
    """
    evaluates the Sobol sensitivities of the PC expansion
    
    gen_mi function inputs (generates the multi-index files): 
    -x "Multiindex type" 
    -p "PC polynomial order" 
    -q "PC dimension (number of parameters)" 
    
    pce_sens function inputs (generates the Sobol sensitivity indices): 
    -x "PC type" 
    -f "PC coefficient filename"
    -m "Multiindex file (mindex.dat by default)"
    """
    # evaluate the multi-index file
    uqtk_cmd = f'gen_mi -x {multiindex_type} -p {poly_order} -q {pc_dimension}'
    os.system(uqtk_cmd)

    # evaluating the sensitivities
    uqtk_cmd = f'pce_sens -f {parameter_filename} -x {pc_type}'
    os.system(uqtk_cmd)
    main_sensitivity = np.loadtxt('mainsens.dat')
    joint_sensitivity = np.loadtxt('jointsens.dat')
    total_sensitivity = np.loadtxt('totsens.dat')
    return main_sensitivity, joint_sensitivity, total_sensitivity

def evaluate_pc_pdf(
    parameter_filename: os.PathLike = 'coeff.dat',
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
    evaluates the PDF of the surrogate output
    
    gen_mi function inputs (generates the multi-index files): 
    -x "Multiindex type" 
    -p "PC polynomial order" 
    -q "PC dimension (number of parameters)" 
    
    pce_rv function inputs (generates the random variable): 
    -x "PC type" 
    -n "num samples" 
    -o "PC polynomial order" 
    -p "PC dimension (number of parameters)" 
    -f "PC coefficient filename"
    -w "type of random variable"
    
    pdf_cl function inputs (generates the pdf): 
    -i "input random variable filename" (rvar.dat from pce_rv) 
    -g "number of bins in the pdf" 
    """

    # evaluate the multi-index file
    uqtk_cmd = f'gen_mi -x {multiindex_type} -p {poly_order} -q {pc_dimension}'
    os.system(uqtk_cmd)

    # evaluating the PC-related random variables
    uqtk_cmd = f'pce_rv -x {pc_type} -n {num_samples} -o {poly_order} -p {pc_dimension} -f {parameter_filename} -m mindex.dat -w PCmi'
    os.system(uqtk_cmd)

    # evaluating the PDF of the PC expansion
    uqtk_cmd = f'pdf_cl -i rvar.dat -g {pdf_bins}'
    os.system(uqtk_cmd)
    xtarget = np.loadtxt('dens.dat')[:, :-1].squeeze()
    probability = np.loadtxt('dens.dat')[:, -1:].squeeze()

    if figname is not None:
        pyplot.figure(figsize=(12, 8))

        pyplot.plot(xtarget, probability)

        if custom_xlabel is not None:
            pyplot.xlabel(custom_xlabel)

        pyplot.ylabel('PDF')

        pyplot.suptitle(figname)
        output_filename = f'{figname}.png'
        pyplot.savefig(output_filename, bbox_inches='tight')

    return xtarget, probability
