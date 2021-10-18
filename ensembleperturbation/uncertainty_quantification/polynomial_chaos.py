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
    if output_filename is not None:
        os.rename('ydata.dat', output_filename)
    return


def evaluate_pc_sensitivity(
    parameter_filename: os.PathLike = 'coeff.dat',
    pc_type: str = 'HG',
    multiindex_type: str = 'TO',
    poly_order: int = 5,
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


def evaluate_pc_pdf(
    uqtkbin: os.PathLike,
    k: int = 0,
    pc_type: str = 'HG',
    multiindex_type: str = 'TO',
    poly_order: int = 5,
    pc_dimension: int = 1,
    num_samples: int = 10000,
    custom_xlabel: str = None,
    figname: str = None,
):
    """
    evaluates the PDF of the surrogate output
    
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

    # evaluating the random variables for the PC expansion
    uqtk_cmd = f"pce_rv -w 'PCmi' -n {num_samples} -p {pc_dimension} -f 'cfs' -m 'mi' -x {pc_type} > pcrv.log"
    os.system(uqtk_cmd)

    cmd = f'{uqtkbin} pdf_cl -i rvar.dat -g 1000 > pdfcl.log'
    os.system(cmd)
    xtarget = np.loadtxt('dens.dat')[:, :-1]
    dens = np.loadtxt('dens.dat')[:, -1:]

    # rv=np.loadtxt('rvar.dat')
    # xtarget=np.linspace(rv.min(),rv.max(),100)
    # kernlin=stats.kde.gaussian_kde(rv)
    # dens=kernlin.evaluate(xtarget)

    np.savetxt('pcdens.dat', np.vstack((xtarget, dens)).T)

    pyplot.figure(figsize=(12, 8))

    pyplot.plot(xtarget, dens)

    if custom_xlabel is not None:
        pyplot.xlabel(custom_xlabel)

    pyplot.ylabel('PDF')

    if figname is not None:
        pyplot.suptitle(figname)
        output_filename = f'{figname}.png'
    else:
        output_filename = 'pdf.png'

    pyplot.savefig(output_filename, bbox_inches='tight')
