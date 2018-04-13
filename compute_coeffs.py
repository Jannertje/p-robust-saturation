from __future__ import print_function, division
import math
import numpy as np
from numpy.linalg import norm
from scipy.sparse import csc_matrix, kron
from scipy.sparse.linalg import spsolve, eigsh
import matplotlib.pyplot as plt

"""
#######################################################
Basis Function abstract base class and children classes
#######################################################
"""
class BasisFunction(object):
    def __init__(self):
        pass

    def degree(self): assert(False)
    def H1inp(self, other): assert(False)
    def L2inp(self, other): assert(False)
    def pointEvalIn1(self): assert(False)

class LegendreBasisFunction(BasisFunction):
    _degree = -1
    def __init__(self,degree):
        self._degree = degree

    def degree(self): return self._degree

    def H1inp(self, other):
        assert(False)

    def L2inp(self, other):
        if isinstance(other, LegendreBasisFunction):
            # legendre basis is L2-orthonormal
            return 1.0 * (self.degree() == other.degree())
        elif isinstance(other, BabuskaShenBasisFunction):
            k = other.degree()
            m = self.degree()
            #print("L2-inner product <xi_%d, phi_%d>" % (k, m))
            # L2-inner product <xi_k, phi_m>
            if k == 0:
                assert(False)
            elif k == 1:
                if m == 0:               return 1.0
                elif m == 1:
                    if other.is_prime(): return math.sqrt(3.0)/3.0
                    else:                return -math.sqrt(3.0)/3.0
                else:                    return 0.0
            elif k == m:                 return -1.0/(math.sqrt(2*k+1)*math.sqrt(2*k-1))
            elif k == m + 2:             return 1.0/(math.sqrt(2*k-1)*math.sqrt(2*k-3))
            else:                        return 0.0
        elif isinstance(other, ZeroMeanBabuskaShenBasisFunction):
            k = other.degree()
            m = self.degree()
            # L2-inner product <chi_k, phi_m>
            if k == 0:
                if m == 0:   return 1.0
                else:        return 0.0
            elif k == 1:
                if m == 0:   return 0.0
                elif m == 1: return -1.0/math.sqrt(3.0)
                else:        return 0.0
            elif k == 2:
                if m == 0:   return 0.0
                elif m == 1: return 0.0
                elif m == 2: return -1.0/math.sqrt(15.0)
                else:        return 0.0
            elif k == m:     return -1.0/(math.sqrt(2*k+1)*math.sqrt(2*k-1))
            elif k == m + 2: return 1.0/(math.sqrt(2*k-1)*math.sqrt(2*k-3))
            else:            return 0.0
        else:
            assert(False)

# chi_0 := 1/sqrt(2)
# chi_1 := xi_1 - <xi_1, 1>/<1,1> = -sqrt(2) x/2
# chi_2 := xi_2 - <xi_2, 1>/<1,1> = (1 - 3 x^2)/(2 sqrt(6))
# chi_k := xi_k  (k >= 3)
class ZeroMeanBabuskaShenBasisFunction(BasisFunction):
    _degree = -1
    def __init__(self, degree):
        self._degree = degree

    def degree(self): return self._degree

    def pointEvalIn1(self):
        if self.degree() == 0:      return 1.0/math.sqrt(2.0)
        elif self.degree() == 1:    return -1.0/math.sqrt(2.0)
        elif self.degree() == 2:    return -1.0/math.sqrt(6.0)
        else:                       return 0.0

    def H1inp(self, other):
        if isinstance(other, ZeroMeanBabuskaShenBasisFunction):
            k = self.degree()
            m = other.degree()
            if m != k:   return 0.0
            elif m == 0: return 0.0
            else:        return 1.0
        assert(False)

    def L2inp(self, other):
        if isinstance(other, LegendreBasisFunction):
            return other.L2inp(self)
        elif isinstance(other, ZeroMeanBabuskaShenBasisFunction):
            if other.degree() < self.degree():
                return other.L2inp(self)
            k = self.degree()
            m = other.degree()
            # k <= m
            if k == 0:
                if m == 0:   return 1.0
                else:        return 0.0
            elif k == 1:
                if m == 1:   return 1.0/3.0
                elif m == 2: return 0.0
                elif m == 3: return -1.0/(3.0 * math.sqrt(5))
                else:        return 0.0
            elif k == 2:
                if m == 2:   return 1.0/15.0
                elif m == 3: return 0.0
                elif m == 4: return -1.0/(5.0 * math.sqrt(21))
                else:        return 0.0
            # see [CNSV18] formula (2.3)
            if k == m:       return 2.0/((2*k-3) * (2*k+1))
            elif k + 2 == m: return -1.0/((2*k + 1)*math.sqrt((2*k-1)*(2*k+3)))
            else:            return 0.0
        assert(False)

class BabuskaShenBasisFunction(BasisFunction):
    _degree = -1
    def __init__(self,degree):
        assert(degree >= 2)
        self._degree = degree

    def degree(self): return self._degree

    def pointEvalIn1(self):
        return 0.0

    def H1inp(self, other):
        if isinstance(other, BabuskaShenBasisFunction):
            k = self.degree()
            m = other.degree()
            if m < k:  return other.H1inp(self)
            if m == k: return 1.0
            else:      return 0.0
        assert(False)

    def L2inp(self, other):
        if isinstance(other, BabuskaShenBasisFunction):
            k = self.degree()
            m = other.degree()
            if m < k:        return other.L2inp(self)
            # see [CNSV18] formula (2.3)
            if k == m:       return 2.0/((2*k-3) * (2*k+1))
            elif k + 2 == m: return -1.0/((2*k + 1)*math.sqrt((2*k-1)*(2*k+3)))
            else:            return 0.0
        assert(False)

class FirstBabuskaShenBasisFunction(BabuskaShenBasisFunction):
    _zero_in_1 = False
    def __init__(self,zero_in_1):
        self._zero_in_1 = zero_in_1

    def is_zero_in_1(self):
        return self._zero_in_1

    def is_prime(self):
        return not self._zero_in_1

    def degree(self):
        return 1

    def pointEvalIn1(self):
        return math.sqrt(2) * (not self._zero_in_1)

    def H1inp(self, other):
        if isinstance(other, BabuskaShenBasisFunction):
            if other.degree() == 1:
                if (self.is_prime() and other.is_prime()) or (not self.is_prime() and not other.is_prime()):
                    return 1.0
                else:
                    return -1.0
            else:
                return 0.0
        else:
            assert(False)

    def L2inp(self, other):
        if isinstance(other, BabuskaShenBasisFunction):
            if other.degree() == 1:
                if (self.is_prime() and other.is_prime()) or (not self.is_prime() and not other.is_prime()):
                    return 4.0/3.0
                else:
                    return 2.0/3.0
            elif other.degree() == 2: return math.sqrt(3.0)/3.0
            elif other.degree() == 3:
                if self.is_prime():   return 1.0/(3.0*math.sqrt(5))
                else:                 return -1.0/(3.0*math.sqrt(5))
            else:                     return 0.0
        else:
            assert(False)

# evaluate the function in the point x=1
class PointEvalBasisFunction(BasisFunction):
    def __init__(self): pass
    def degree(self): assert(False)
    def H1inp(self, other): assert(False)
    def L2inp(self, other): return other.pointEvalIn1()

"""
#################################################
1D Basis abstract base class and children classes
#################################################
"""
class Basis1D(object):
    def __init__(self):
        # we need to subclass this ABS
        assert(False)

    def H1inp(self, other):
        assert(isinstance(other, Basis1D))
        h1inp = csc_matrix([[pol1.H1inp(pol2) for pol1 in self._basis] for pol2 in other._basis], dtype=np.float64)
        return h1inp

    def L2inp(self, other):
        assert(isinstance(other, Basis1D))
        l2inp = csc_matrix([[pol1.L2inp(pol2) for pol1 in self._basis] for pol2 in other._basis], dtype=np.float64)
        return l2inp

class BabuskaShenBasis1D(Basis1D):
    def __init__(self,r, zero_in_m1, zero_in_1):
        basis = []
        if not zero_in_1:
            basis.append(FirstBabuskaShenBasisFunction(zero_in_1=False))
        if not zero_in_m1:
            basis.append(FirstBabuskaShenBasisFunction(zero_in_1=True))
        for k in range(2, r+1):
            basis.append(BabuskaShenBasisFunction(degree=k))

        self._basis = basis

class ZeroMeanBabuskaShenBasis1D(Basis1D):
    def __init__(self, r):
        basis = []
        for k in range(0, r+1):
            basis.append(ZeroMeanBabuskaShenBasisFunction(degree=k))
        self._basis = basis

class LegendreBasis1D(Basis1D):
    def __init__(self, r):
        basis = []
        for k in range(0, r+1):
            basis.append(LegendreBasisFunction(degree=k))
        self._basis = basis

class ZeroMeanLegendreBasis1D(Basis1D):
    def __init__(self, r):
        basis = []
        for k in range(1, r+1):
            basis.append(LegendreBasisFunction(degree=k))
        self._basis = basis

class PointEvalFunctionalBasis(Basis1D):
    def __init__(self):
        self._basis = [PointEvalBasisFunction()]

    def H1inp(self, other):
        assert(False)

"""
#################################################
2D Basis abstract base class and children classes
#################################################
"""
class Basis2D(object):
    def __init__(self, basis1, basis2):
        self._basis1 = basis1
        self._basis2 = basis2

    def startFrom(self):
        return 0

    def H1inp(self, other):
        assert(isinstance(other, Basis2D))
        h1inp = kron(self._basis1.H1inp(other._basis1), self._basis2.L2inp(other._basis2), format="csr") \
              + kron(self._basis1.L2inp(other._basis1), self._basis2.H1inp(other._basis2), format="csr")
        h1inp = h1inp[other.startFrom():, self.startFrom():]
        return h1inp

    def L2inp(self, other):
        assert(isinstance(other, Basis2D))
        l2inp = kron(self._basis1.L2inp(other._basis1), self._basis2.L2inp(other._basis2), format="csr")
        l2inp = l2inp[other.startFrom():, self.startFrom():]
        return l2inp

class ZeroMeanBabuskaShenBasis2D(Basis2D):
    def __init__(self, basis1, basis2):
        Basis2D.__init__(self, basis1, basis2) 

    def startFrom(self):
        return 1

"""
#####################################################
BasisBuilder abstract base class and children classes
#####################################################
"""
# we will subclass this abstract base class
class BasisBuilder(object):
    def __init__(self): assert(False)
    def basis(self, r): assert(False)

class BabuskaShenBasisBuilder(BasisBuilder):
    def __init__(self, zero_edges):
        self._zero_edges = zero_edges

    def basis(self, r):
        basis1 = BabuskaShenBasis1D(r,
                zero_in_m1 = self._zero_edges[2], zero_in_1 = self._zero_edges[0])
        basis2 = BabuskaShenBasis1D(r,
                zero_in_m1 = self._zero_edges[1], zero_in_1 = self._zero_edges[3])
        return Basis2D(basis1, basis2)

class ZeroMeanBabuskaShenBasisBuilder(BasisBuilder):
    def __init__(self): pass

    def basis(self, r):
        basis1 = ZeroMeanBabuskaShenBasis1D(r)
        basis2 = ZeroMeanBabuskaShenBasis1D(r)
        return ZeroMeanBabuskaShenBasis2D(basis1, basis2)

class LegendreBasisBuilder(BasisBuilder):
    def __init__(self): pass

    def basis(self, r):
        basis1 = LegendreBasis1D(r)
        basis2 = LegendreBasis1D(r)
        return Basis2D(basis1, basis2)

class EdgeFunctionalBasisBuilder(BasisBuilder):
    _zero_mean = False
    def __init__(self, zero_mean = False):
        self._zero_mean = zero_mean

    def basis(self, r):
        basis1 = PointEvalFunctionalBasis()
        if self._zero_mean:
            basis2 = ZeroMeanLegendreBasis1D(r)
        else:
            basis2 = LegendreBasis1D(r)
        return Basis2D(basis1, basis2)

"""
###############################################################################
Discrete Saturation Coefficient class that uses all of the above to compute the
coefficients.
###############################################################################
"""
class DiscreteSaturationCoefficient(object):
    # the space V is a subspace of H and we'll leave it implicit
    def __init__(self, Xi_H, Xi_F): 
        self._Xi_H = Xi_H
        self._Xi_V = Xi_H
        self._Xi_F = Xi_F

    def solve_system(self, stiff, rhs):
        return spsolve(stiff, rhs)

    def matrix_product(self, A, B):
        return A.dot(B)

    def largest_genev(self, A, M):
        if isinstance(A, np.ndarray):
            assert(False)
        return eigsh(A, k=1, M=M, which='LM')[0]

    def compute(self, p, q, r):
        # set up bases
        Xi_H = self._Xi_H.basis(r)
        Xi_V = self._Xi_V.basis(q)
        Xi_F = self._Xi_F.basis(p)

        # compute necessary inner products for H
        #print("Computing stiffness matrix <nabla Xi_%d, nabla Xi_%d>" % (r, r))
        stiff_H = Xi_H.H1inp(Xi_H)
        #print("Computing rhs-matrix <Phi_%d, Xi_%d>" % (p, r))
        rhs_H = Xi_F.L2inp(Xi_H)
        sol_H = self.solve_system(stiff_H, rhs_H)
        eigen_H = self.matrix_product(sol_H.T, rhs_H)

        # compute necessary inner products for V
        stiff_V = Xi_V.H1inp(Xi_V)
        rhs_V = Xi_F.L2inp(Xi_V)
        #plt.imshow(rhs_V.toarray())
        #plt.show()
        sol_V = self.solve_system(stiff_V, rhs_V)
        eigen_V = self.matrix_product(sol_V.T, rhs_V)

        mu = self.largest_genev(eigen_H, eigen_V)
        return mu

"""
We hebben de in-producten
    - <nabla xi_k, nabla xi_m>
    - <nabla chi_k, nabla chi_m>
    - <phi_k, phi_m>
    - <xi_k, phi_m>
    - <chi_k, phi_m>
    - <chi_k, chi_m>
    - <xi_k, xi_m>
    - xi_k(1), chi_k(1)
    """
def test_inner_products():
    def assert_small_difference(A, B):
        if (norm(A - B) < 1e-10):
            return
        else:
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    if norm(A[i,j] - B[i,j]) > 1e-10:
                        print("(%d,%d) = %f" % (i, j, A[i,j] - B[i,j]))
            assert(False)

    dXi_dXi_ref = np.eye(11); dXi_dXi_ref[1,0] = -1; dXi_dXi_ref[0,1] = -1
    dChi_dChi_ref = np.eye(11); dChi_dChi_ref[0,0] = 0
    Phi_Phi_ref = np.eye(7)
    # computed with mathematica
    Phi_Xi_ref = np.array([[1.0000000000000000000, 0.57735026918962576451, 0, 0, 0, 0, 
      0], [1.0000000000000000000, -0.57735026918962576451, 0, 0, 0, 0, 
      0], [0.57735026918962576451, 0, -0.25819888974716112568, 0, 0, 0, 
      0], [0, 0.25819888974716112568, 0, -0.16903085094570331550, 0, 0, 
      0], [0, 0, 0.16903085094570331550, 0, -0.12598815766974240907, 0, 
      0], [0, 0, 0, 0.12598815766974240907, 0, -0.10050378152592120755, 
      0], [0, 0, 0, 0, 0.10050378152592120755, 
      0, -0.083624201000709077071], [0, 0, 0, 0, 0, 
      0.083624201000709077071, 0], [0, 0, 0, 0, 0, 0, 
      0.071611487403943288051], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 
      0]])
    Phi_Chi_ref = np.array([[1.0000000000000000000, 0, 0, 0, 0, 0,
      0], [0, -0.57735026918962576451, 0, 0, 0, 0, 0], [0,
      0, -0.25819888974716112568, 0, 0, 0, 0], [0, 0.25819888974716112568,
       0, -0.16903085094570331550, 0, 0, 0], [0, 0,
      0.16903085094570331550, 0, -0.12598815766974240907, 0, 0], [0, 0, 0,
       0.12598815766974240907, 0, -0.10050378152592120755, 0], [0, 0, 0,
      0, 0.10050378152592120755, 0, -0.083624201000709077071], [0, 0, 0,
      0, 0, 0.083624201000709077071, 0], [0, 0, 0, 0, 0, 0,
      0.071611487403943288051], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,
      0]])
    Xi_Xi_ref = np.array([[1.3333333333333333333, 0.66666666666666666667, 
      0.57735026918962576451, 0.14907119849998597976, 0, 0, 0, 0, 0, 0, 
      0], [0.66666666666666666667, 1.3333333333333333333, 
      0.57735026918962576451, -0.14907119849998597976, 0, 0, 0, 0, 0, 0, 
      0], [0.57735026918962576451, 0.57735026918962576451, 
      0.40000000000000000000, 0, -0.043643578047198476253, 0, 0, 0, 0, 0, 
      0], [0.14907119849998597976, -0.14907119849998597976, 0, 
      0.095238095238095238095, 0, -0.021295885499997997109, 0, 0, 0, 0, 
      0], [0, 0, -0.043643578047198476253, 0, 0.044444444444444444444, 
      0, -0.012662286273293105426, 0, 0, 0, 0], [0, 0, 
      0, -0.021295885499997997109, 0, 0.025974025974025974026, 
      0, -0.0084045484276549866972, 0, 0, 0], [0, 0, 0, 
      0, -0.012662286273293105426, 0, 0.017094017094017094017, 
      0, -0.0059884534166270997763, 0, 0], [0, 0, 0, 0, 
      0, -0.0084045484276549866972, 0, 0.012121212121212121212, 
      0, -0.0044844852933087497806, 0], [0, 0, 0, 0, 0, 
      0, -0.0059884534166270997763, 0, 0.0090497737556561085973, 
      0, -0.0034844051632887785724], [0, 0, 0, 0, 0, 0, 
      0, -0.0044844852933087497806, 0, 0.0070175438596491228070, 0], [0, 
      0, 0, 0, 0, 0, 0, 0, -0.0034844051632887785724, 0, 
      0.0056022408963585434174]])
    Chi_Chi_ref = np.array([[1.0000000000000000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 
      0.33333333333333333333, 0, -0.14907119849998597976, 0, 0, 0, 0, 0, 
      0, 0], [0, 0, 0.066666666666666666667, 0, -0.043643578047198476253, 
      0, 0, 0, 0, 0, 0], [0, -0.14907119849998597976, 0, 
      0.095238095238095238095, 0, -0.021295885499997997109, 0, 0, 0, 0, 
      0], [0, 0, -0.043643578047198476253, 0, 0.044444444444444444444, 
      0, -0.012662286273293105426, 0, 0, 0, 0], [0, 0, 
      0, -0.021295885499997997109, 0, 0.025974025974025974026, 
      0, -0.0084045484276549866972, 0, 0, 0], [0, 0, 0, 
      0, -0.012662286273293105426, 0, 0.017094017094017094017, 
      0, -0.0059884534166270997763, 0, 0], [0, 0, 0, 0, 
      0, -0.0084045484276549866972, 0, 0.012121212121212121212, 
      0, -0.0044844852933087497806, 0], [0, 0, 0, 0, 0, 
      0, -0.0059884534166270997763, 0, 0.0090497737556561085973, 
      0, -0.0034844051632887785724], [0, 0, 0, 0, 0, 0, 
      0, -0.0044844852933087497806, 0, 0.0070175438596491228070, 0], [0, 
      0, 0, 0, 0, 0, 0, 0, -0.0034844051632887785724, 0, 
      0.0056022408963585434174]])
    Xi_eval1_ref = np.array([1.4142135623730950488, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((11,1))
    Chi_eval1_ref = np.array([0.70710678118654752440, -0.70710678118654752440, -0.40824829046386301637, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((11,1))

    Phi = LegendreBasis1D(6)
    Xi = BabuskaShenBasis1D(10, False, False)
    Chi = ZeroMeanBabuskaShenBasis1D(10)
    EvalIn1 = PointEvalFunctionalBasis()

    assert_small_difference(Xi.H1inp(Xi).toarray(), dXi_dXi_ref)
    assert_small_difference(Chi.H1inp(Chi).toarray(), dChi_dChi_ref)
    assert_small_difference(Phi.L2inp(Phi).toarray(), Phi_Phi_ref)
    assert_small_difference(Xi.L2inp(Xi).toarray(), Xi_Xi_ref)
    assert_small_difference(Chi.L2inp(Chi).toarray(), Chi_Chi_ref)
    assert_small_difference(Phi.L2inp(Xi).toarray(), Phi_Xi_ref)
    assert_small_difference(Phi.L2inp(Chi).toarray(), Phi_Chi_ref)

    assert_small_difference(EvalIn1.L2inp(Xi).toarray(), Xi_eval1_ref)
    Chi_eval1 = EvalIn1.L2inp(Chi).toarray()


def do_it():
    Xi_H_C1E1 = BabuskaShenBasisBuilder(zero_edges=[True, False, False, False])
    Xi_H_C1E2 = BabuskaShenBasisBuilder(zero_edges=[True, True, False, False])
    Xi_H_C1E3 = BabuskaShenBasisBuilder(zero_edges=[True, False, True, False])
    Xi_H_C1E4 = BabuskaShenBasisBuilder(zero_edges=[True, True, True, False])
    Xi_H_C1E5 = BabuskaShenBasisBuilder(zero_edges=[True, True, True, True])

    Xi_F_C1 = LegendreBasisBuilder()

    C1E1 = DiscreteSaturationCoefficient(Xi_H_C1E1, Xi_F_C1)
    C1E2 = DiscreteSaturationCoefficient(Xi_H_C1E2, Xi_F_C1)
    C1E3 = DiscreteSaturationCoefficient(Xi_H_C1E3, Xi_F_C1)
    C1E4 = DiscreteSaturationCoefficient(Xi_H_C1E4, Xi_F_C1)
    C1E5 = DiscreteSaturationCoefficient(Xi_H_C1E5, Xi_F_C1)

    Xi_H_C2E1 = BabuskaShenBasisBuilder(zero_edges=[False, True, False, False])
    Xi_H_C2E2 = BabuskaShenBasisBuilder(zero_edges=[False, False, True, False])
    Xi_H_C2E3 = BabuskaShenBasisBuilder(zero_edges=[False, True, True, False])
    Xi_H_C2E4 = BabuskaShenBasisBuilder(zero_edges=[False, True, True, True])

    Xi_F_C2 = EdgeFunctionalBasisBuilder()

    C2E1 = DiscreteSaturationCoefficient(Xi_H_C2E1, Xi_F_C2)
    C2E2 = DiscreteSaturationCoefficient(Xi_H_C2E2, Xi_F_C2)
    C2E3 = DiscreteSaturationCoefficient(Xi_H_C2E3, Xi_F_C2)
    C2E4 = DiscreteSaturationCoefficient(Xi_H_C2E4, Xi_F_C2)

    Xi_H_C3 = ZeroMeanBabuskaShenBasisBuilder()
    Xi_F_C3 = EdgeFunctionalBasisBuilder(zero_mean = True)
    C3 = DiscreteSaturationCoefficient(Xi_H_C3, Xi_F_C3)

    to_compute = []
    qs = [128]#[8, 16, 32, 64, 128]
    ps = [lambda q: q-4, lambda q: int(q/8*7), lambda q: int(q/2)]
    rs = [lambda q: 2*q, lambda q: 4*q, lambda q: 8*q]
    for p in ps:
        for q in qs:
            for r in rs:
                if (q == 8 and p(q) == 7) or r(q) > 256:
                    continue
                to_compute.append((p(q), q, r(q)))
    to_compute = list(set(to_compute))
    to_compute.sort()
    print(to_compute)
    with open("computed_values_new.out", "a") as fn:
        for (p, q, r) in to_compute:
            fn.write("%d %d %d %d %d: %.20f\n" % (1, 3, p, q, r, C1E3.compute(p, q, r)))
            fn.write("%d %d %d %d %d: %.20f\n" % (1, 4, p, q, r, C1E4.compute(p, q, r)))
            fn.write("%d %d %d %d %d: %.20f\n" % (1, 5, p, q, r, C1E5.compute(p, q, r)))
            fn.write("%d %d %d %d %d: %.20f\n" % (2, 1, p, q, r, C2E1.compute(p, q, r)))
            fn.write("%d %d %d %d %d: %.20f\n" % (2, 2, p, q, r, C2E2.compute(p, q, r)))
            fn.write("%d %d %d %d %d: %.20f\n" % (2, 3, p, q, r, C2E3.compute(p, q, r)))
            fn.write("%d %d %d %d %d: %.20f\n" % (2, 4, p, q, r, C2E4.compute(p, q, r)))
            fn.write("%d %d %d %d %d: %.20f\n" % (3, 1, p, q, r, C3.compute(p, q, r)))
            fn.write("%d %d %d %d %d: %.20f\n" % (1, 1, p, q, r, C1E1.compute(p, q, r)))
            fn.write("%d %d %d %d %d: %.20f\n" % (1, 2, p, q, r, C1E2.compute(p, q, r)))
            fn.flush()

if __name__ == "__main__":
    #test_inner_products()
    do_it()
