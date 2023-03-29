# Demonsrate batched solve for TS and TSadjoint solvers. 
# Consider a van der Pol ODE system with two initial states. The batched solve is achieved by stacking
# the states into a single state and form a combined system. For efficiency, KSPMatSolve() with HPDDM
# is used to solve a smaller system with multiple RHS. The KSP in TS is customized to use KSPPREONLY
# and a PCShell.
# 
# Options:
#     -mf : enable matrix-free implementation 
#     -imexform : swtich from CN to ARK

from petsc4py import PETSc
import sys, petsc4py

petsc4py.init(sys.argv)


class VDP(object):
    n = 4
    comm = PETSc.COMM_SELF

    def __init__(self, mu_=1.0e3, mf_=False, imex_=False):
        self.mu_ = mu_
        self.mf_ = mf_
        self.imex_ = imex_
        if self.mf_:
            self.Jim_ = PETSc.Mat().createDense([self.n, self.n], comm=self.comm)
            self.Jim_.setUp()
            self.Jim_inner_ = PETSc.Mat().createDense(
                [self.n // 2, self.n // 2], comm=self.comm
            )
            self.Jim_inner_.setUp()
            self.JimP_ = PETSc.Mat().createDense([self.n, 1], comm=self.comm)
            self.JimP_.setUp()
            self.Jex_ = PETSc.Mat().createDense([self.n, self.n], comm=self.comm)
            self.Jex_.setUp()
            self.JexP_ = PETSc.Mat().createDense([self.n, 1], comm=self.comm)
            self.JexP_.setUp()

    def initialCondition(self, u):
        mu = self.mu_
        u[0] = 2.0
        u[1] = -2.0 / 3.0 + 10.0 / (81.0 * mu) - 292.0 / (2187.0 * mu * mu)
        u[2] = u[0]
        u[3] = u[1]
        u.assemble()

    def evalRHSFunction(self, ts, t, u, f):
        mu = self.mu_
        f[0] = u[1]
        f[2] = u[3]
        if self.imex_:
            f[1] = 0.0
            f[3] = 0.0
        else:
            f[1] = mu * ((1.0 - u[0] * u[0]) * u[1] - u[0])
            f[3] = mu * ((1.0 - u[2] * u[2]) * u[3] - u[2])
        f.assemble()

    def evalRHSJacobian(self, ts, t, u, A, B):
        if not self.mf_:
            J = A
        else:
            J = self.Jex_
        mu = self.mu_
        J[0, 0] = 0
        J[0, 1] = 1.0
        J[2, 2] = 0
        J[2, 3] = 1.0
        if self.imex_:
            J[1, 0] = 0
            J[1, 1] = 0
            J[3, 2] = 0
            J[3, 3] = 0
        else:
            J[1, 0] = -mu * (2.0 * u[1] * u[0] + 1.0)
            J[1, 1] = mu * (1.0 - u[0] * u[0])
            J[3, 2] = -mu * (2.0 * u[3] * u[2] + 1.0)
            J[3, 3] = mu * (1.0 - u[2] * u[2])
        J.assemble()
        if A != B:
            B.assemble()
        return True  # same nonzero pattern

    def evalRHSJacobianP(self, ts, t, u, C):
        if not self.mf_:
            Jp = C
        else:
            Jp = self.JexP_
        if not self.imex_:
            Jp[0, 0] = 0
            Jp[1, 0] = (1.0 - u[0] * u[0]) * u[1] - u[0]
            Jp[2, 0] = 0
            Jp[3, 0] = (1.0 - u[2] * u[2]) * u[3] - u[2]
            Jp.assemble()
        return True

    def evalIFunction(self, ts, t, u, udot, f):
        mu = self.mu_
        if self.imex_:
            f[0] = udot[0]
            f[2] = udot[2]
        else:
            f[0] = udot[0] - u[1]
            f[2] = udot[2] - u[3]
        f[1] = udot[1] - mu * ((1.0 - u[0] * u[0]) * u[1] - u[0])
        f[3] = udot[3] - mu * ((1.0 - u[2] * u[2]) * u[3] - u[2])
        f.assemble()

    def evalIJacobian(self, ts, t, u, udot, shift, A, B):
        if not self.mf_:
            J = A
        else:
            J = self.Jim_
        mu = self.mu_
        if self.imex_:
            J[0, 0] = shift
            J[0, 1] = 0.0
            J[2, 2] = shift
            J[2, 3] = 0.0
        else:
            J[0, 0] = shift
            J[0, 1] = -1.0
            J[2, 2] = shift
            J[2, 3] = -1.0
        J[1, 0] = mu * (2.0 * u[1] * u[0] + 1.0)
        J[1, 1] = shift - mu * (1.0 - u[0] * u[0])
        J[3, 2] = mu * (2.0 * u[3] * u[2] + 1.0)
        J[3, 3] = shift - mu * (1.0 - u[2] * u[2])
        J.assemble()
        if A != B:
            B.assemble()
        if self.mf_:
            A.stateIncrease()  # change the state of the mat
            B.stateIncrease()  # change the state of the mat
            if self.imex_:
                self.Jim_inner_[0, 0] = shift
                self.Jim_inner_[0, 1] = 0.0
            else:
                self.Jim_inner_[0, 0] = shift
                self.Jim_inner_[0, 1] = -1.0
            self.Jim_inner_[1, 0] = mu * (2.0 * u[1] * u[0] + 1.0)
            self.Jim_inner_[1, 1] = shift - mu * (1.0 - u[0] * u[0])
            self.Jim_inner_.assemble()
        return True  # same nonzero pattern

    def evalIJacobianP(self, ts, t, u, udot, shift, C):
        if not self.mf_:
            Jp = C
        else:
            Jp = self.JimP_
        Jp[0, 0] = 0
        Jp[1, 0] = u[0] - (1.0 - u[0] * u[0]) * u[1]
        Jp[2, 0] = 0
        Jp[3, 0] = u[2] - (1.0 - u[2] * u[2]) * u[3]
        Jp.assemble()
        return True


class JacShell:
    def __init__(self, ode):
        self.ode_ = ode

    def mult(self, A, x, y):
        "y <- A * x"
        self.ode_.Jex_.mult(x, y)

    def multTranspose(self, A, x, y):
        "y <- A' * x"
        self.ode_.Jex_.multTranspose(x, y)


class JacPShell:
    def __init__(self, ode):
        self.ode_ = ode

    def multTranspose(self, A, x, y):
        "y <- A' * x"
        self.ode_.JexP_.multTranspose(x, y)


class IJacShell:
    def __init__(self, ode):
        self.ode_ = ode

    def mult(self, A, x, y):
        "y <- A * x"
        self.ode_.Jim_.mult(x, y)

    def multTranspose(self, A, x, y):
        "y <- A' * x"
        self.ode_.Jim_.multTranspose(x, y)

    # MatProduct for KSPMatSolve
    def productSetFromOptions(self, mat, producttype, A, B, C):
        return True

    def productSymbolic(self, mat, product, producttype, A, B, C):
        product.setType(B.getType())
        product.setSizes(B.getSizes())
        product.setUp()
        product.assemble()

    def productNumeric(self, mat, product, producttype, A, B, C):
        D = PETSc.Mat()
        if producttype == "AB":
            self.ode_.Jim_inner_.matMult(B, D)
        if producttype == "AtB":
            self.ode_.Jim_inner_.transposeMatMult(B, D)
        D.copy(product)


class IJacPShell:
    def __init__(self, ode):
        self.ode_ = ode

    def multTranspose(self, A, x, y):
        "y <- A' * x"
        self.ode_.JimP_.multTranspose(x, y)


class PCShell:
    def __init__(self, A, m, n):
        self._m = m
        self._n = n
        self._ksp = PETSc.KSP()
        self._ksp.create(PETSc.COMM_WORLD)
        self._ksp.setType("hpddm")
        self._ksp.setOperators(A)
        self._ksp.setInitialGuessNonzero(True)
        self._ksp.setOptionsPrefix("inner_")
        self._ksp.setFromOptions()
        self._random = PETSc.Random()
        self._random.create(PETSc.COMM_WORLD)
        self._random.setInterval([0, 0.001])

    def apply(self, pc, x, y):
        y.setRandom(self._random)
        X = PETSc.Mat().createDense([self._n, self._m], array=x.array_r)
        Y = PETSc.Mat().createDense([self._n, self._m], array=y.array)
        self._ksp.matSolve(X, Y)
        X.destroy()
        Y.destroy()

    def applyTranspose(self, pc, x, y):
        y.setRandom(self._random)
        X = PETSc.Mat().createDense([self._n, self._m], array=x.array_r)
        Y = PETSc.Mat().createDense([self._n, self._m], array=y.array)
        self._ksp.matSolveTranspose(X, Y)
        X.destroy()
        Y.destroy()

    def getKSP(self):
        return self._ksp


OptDB = PETSc.Options()

mu_ = OptDB.getScalar("mu", 1.0e3)
mf_ = OptDB.getBool("mf", False)

imexform_ = OptDB.getBool("imexform", False)

ode = VDP(mu_, mf_, imexform_)

if not mf_:
    Jim = PETSc.Mat().createDense([ode.n, ode.n], comm=ode.comm)
    Jim.setUp()
    JimP = PETSc.Mat().createDense([ode.n, 1], comm=ode.comm)
    JimP.setUp()
    Jex = PETSc.Mat().createDense([ode.n, ode.n], comm=ode.comm)
    Jex.setUp()
    JexP = PETSc.Mat().createDense([ode.n, 1], comm=ode.comm)
    JexP.setUp()
else:
    Jim = PETSc.Mat().create()
    Jim.setSizes([ode.n, ode.n])
    Jim.setType("python")
    shell = IJacShell(ode)
    Jim.setPythonContext(shell)
    Jim.setUp()
    Jim.assemble()
    innerkspmat = PETSc.Mat().createPython([ode.n // 2, ode.n // 2], shell)
    JimP = PETSc.Mat().create()
    JimP.setSizes([ode.n, 1])
    JimP.setType("python")
    shell = IJacPShell(ode)
    JimP.setPythonContext(shell)
    JimP.setUp()
    JimP.assemble()
    Jex = PETSc.Mat().create()
    Jex.setSizes([ode.n, ode.n])
    Jex.setType("python")
    shell = JacShell(ode)
    Jex.setPythonContext(shell)
    Jex.setUp()
    Jex.assemble()
    JexP = PETSc.Mat().create()
    JexP.setSizes([ode.n, 1])
    JexP.setType("python")
    shell = JacPShell(ode)
    JexP.setPythonContext(shell)
    JexP.setUp()
    # JexP.zeroEntries()
    JexP.assemble()

u = PETSc.Vec().createSeq(ode.n, comm=ode.comm)
f = u.duplicate()
adj_u = []
adj_u.append(PETSc.Vec().createSeq(ode.n, comm=ode.comm))
adj_p = []
adj_p.append(PETSc.Vec().createSeq(1, comm=ode.comm))

adj_u[0][0] = 1.0
adj_u[0][1] = 0.0
adj_u[0][2] = 0.0
adj_u[0][3] = 1.0
adj_u[0].assemble()
adj_p[0][0] = 0
adj_p[0].assemble()

ts = PETSc.TS().create(comm=ode.comm)
ts.setProblemType(ts.ProblemType.NONLINEAR)

if imexform_:
    ts.setType(ts.Type.ARKIMEX)
    ts.setIFunction(ode.evalIFunction, f)
    ts.setIJacobian(ode.evalIJacobian, Jim)
    ts.setIJacobianP(ode.evalIJacobianP, JimP)
    ts.setRHSFunction(ode.evalRHSFunction, f)
    ts.setRHSJacobian(ode.evalRHSJacobian, Jex)
    ts.setRHSJacobianP(ode.evalRHSJacobianP, JexP)
else:
    ts.setType(ts.Type.CN)
    ts.setIFunction(ode.evalIFunction, f)
    ts.setIJacobian(ode.evalIJacobian, Jim)
    ts.setIJacobianP(ode.evalIJacobianP, JimP)

if mf_:
    snes = ts.getSNES()
    ksp = snes.getKSP()
    pc = PETSc.PC()
    pcshell = PCShell(innerkspmat, 2, ode.n // 2)
    pc.createPython(pcshell, PETSc.COMM_SELF)
    kmat, _ = ksp.getOperators()
    pc.setOperators(kmat)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.setPC(pc)

ts.setSaveTrajectory()
ts.setTime(0.0)
ts.setTimeStep(0.001)
ts.setMaxTime(0.5)
ts.setMaxSteps(2)
ts.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)

ts.setFromOptions()
ode.initialCondition(u)
ts.solve(u)

ts.setCostGradients(adj_u, adj_p)
ts.adjointSolve()
adj_u[0].view()
adj_p[0].view()

del ode, Jim, JimP, Jex, JexP, u, f, ts, adj_u, adj_p
