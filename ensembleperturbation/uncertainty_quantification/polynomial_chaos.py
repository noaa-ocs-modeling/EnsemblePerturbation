import os

def build_pc_expansion(
    x_filename: str = 'xdata.dat', y_filename: str = 'qoi.dat', output_filename: str = None,
    pc_type: str = 'HG', poly_order: int = 5, lambda_regularization: int = 0, 
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
    uqtk_cmd = 'regression -x ' + x_filename + ' -y ' + y_filename + ' -s ' + pc_type + ' -o ' + str(poly_order) + ' -l ' + str(lambda_regularization)
    os.system(uqtk_cmd)
    if output_filename is not None:
        os.rename('coeff.dat',output_filename)
    return

def evaluate_pc_expansion(
    x_filename: str = 'xdata.dat', parameter_filename: str = 'coeff.dat', output_filename: str = None,
    pc_type: str = 'HG', poly_order: int = 5, 
):
    """
    evaluates the polynomial-chaos expansion
    
    pce_eval function inputs: 
    -x "xdata filename" 
    -f "parameter filename" 
    -s "PCtype" 
    -o "Polynomial order"  
    """
    assert x_filename == 'xdata.dat', 'x_filename needs to be xdata.dat'
    uqtk_cmd = 'pce_eval -f ' + parameter_filename + ' -s ' + pc_type + ' -o ' + str(poly_order)
    os.system(uqtk_cmd)
    if output_filename is not None:
        os.rename('ydata.dat',output_filename)
    return
