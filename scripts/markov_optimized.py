"""Author's L2 optimizer; certify the finite model for the saved decimal nodes."""
import ast, math
import numpy as np
from flint import arb
from scipy.special import gamma,gammainc
from scipy.optimize import minimize,lsq_linear
import markov_suite as suite
source=suite.ROOT/"00_references/RoughKernel.py"
names={"sort","exp_underflow","error_l2_optimal_weights","optimize_error_l2"}
tree=ast.parse(source.read_text(encoding="utf8"))
tree.body=[f for f in tree.body if isinstance(f,ast.FunctionDef) and f.name in names]
exec(compile(tree,str(source),"exec"))
def optimized(H,N,L):
    h=float(H); t=float(L)
    initial=np.geomspace(1/t,(8**(N-1))/t,N)
    _,x,w=optimize_error_l2(h,N,t,init_nodes=initial,tol=1e-7)
    w=w*gamma(h+.5)
    assert np.all(w>0) and np.all(x>0)
    return [arb(str(float(z))) for z in x],[arb(str(float(z))) for z in w]
if __name__=="__main__":
    suite.main(rule=optimized,filename="markov_optimized")


