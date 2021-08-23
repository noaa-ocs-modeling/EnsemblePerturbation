

read_hdf.py : this is essentially the notebook the William shared. I am saving the input and output into text files for the next scripts to use.

run_pc.py : an example of polynomial chaos (PC, not to confuse with principal components:) ) surrogate construction. Uses UQTk apps using system call, so make sure <uqtk-install-location>/bin is in your system's path, or add it explicitly into the script.

run_kl.py : an example of Karhunen-Loeve construction. I found a pure python implementation (it is bunch of linear algebra), so this does not rely on UQTk. This is a simple demo as a proof of concept - see the comments I left. Basically, instead of the full 200K output points, you should build PC surrogate for the first, say neig xi's (KL coefficients corresponding to first neig eigenmodes). As I said, this is a glorified principal components analysis (PCA) - feel free to use your own PCA, but the idea is the same. 

Now, the caveat: in run_kl.py, I picked every 100th grid point, otherwise the code chokes (KL relies on eigendecomposition that is impossible for 200K grid points, since it requires 200K by 200K matrix inversion). There are sparse covariance methods at the expense of more loss of accuracy, and they are quite complex. The easiest way-around for me was to pick every, say, 100th grid point. 
You can check out the map with read_hdf.py or your notebook with every 100th point and see if it captures the interesting locations... Otherwise, do your own downselection (sparsification) of the regional grid looking at the longitude/latitude pairs more meaningfully.

