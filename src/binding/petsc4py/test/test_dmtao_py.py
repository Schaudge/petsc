# --------------------------------------------------------------------

import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
from sys import getrefcount
import unittest
import numpy as np


# --------------------------------------------------------------------


class MyDMTao:
    def __init__(self):
        self.log = {}

    def _log(self, method):
        self.log.setdefault(method, 0)
        self.log[method] += 1

    def create(self, dm):
        self._log('create')
        #self.testvec = PETSc.Vec()

    def destroy(self, dm):
        self._log('destroy')
        #self.testvec.destroy()

    def setFromOptions(self, dm):
        self._log('setFromOptions')

    def setUp(self, dm):
        self._log('setUp')
        #self.testvec = dm.getSolution().duplicate()#TODO does this work?

    def applyproximalmap(self, dm0, dm1, thresh, vecin, vecout, flg):
        self._log('applyproximalmap')
        #start, end = vecin.getOwnershipRange()
        #for i in range(start,end):
        #    vecout[i] = (vecin[i]/np.abs(vecin[i])) * np.maximum(np.abs(vecin[i]) - thresh, 0)
        #vecin.assemble()
        #vecout.assemble()

class TestDMTaoPython(unittest.TestCase):
    def setUp(self):
        print("set up")
        self.dm = PETSc.DM()
        self.dm.createTAOPython(MyDMTao(), comm=PETSc.COMM_SELF)
        print("set up end")
        #ctx = self.dm.getTAOPythonContext()
        #self.assertEqual(getrefcount(ctx), 3)
        #self.assertEqual(ctx.log['create'], 1)

    def tearDown(self):
        print("tear down 1")
        #ctx = self.dm.getTAOPythonContext()
        print("tear down 2")
        #self.assertEqual(getrefcount(ctx), 3)
        print("tear down 3")
        #self.assertTrue('destroy' not in ctx.log)
        print("tear down 4")
        #self.dm.setTAOFromOptions()
        print("tear down 5")
        self.dm.destroy()
        print("tear down 6")
        self.dm = None
        print("tear down 7")
        PETSc.garbage_cleanup()
        #self.assertEqual(ctx.log['destroy'], 1)
        #self.assertEqual(getrefcount(ctx), 2)

    def testGetType(self):
        print("get type")
        ctx = self.dm.getTAOPythonContext()
        pytype = f'{ctx.__module__}.{type(ctx).__name__}'
        self.assertTrue(self.dm.getTAOPythonType() == pytype)

    def testProx(self):
        print("test prox begin")
        thresh = 0.1
        dm     = self.dm
        ctx    = dm.getTAOPythonContext()
        x      = PETSc.Vec().create(dm.getComm())

        print('3')
        x.setType('standard')
        x.setSizes(10)
        x.set(0)
        y1 = x.duplicate()
        y1.setRandom()
        y2 = y1.duplicate()
        y1.copy(y2)

        x_l1_test = x.duplicate()
        x.copy(x_l1_test)

        dml1= PETSc.DM()
        dml1.create(PETSc.COMM_WORLD)
        dml1.setTAOType(PETSc.DM.TAOType.L1)

        dml2 = PETSc.DM()
        dml2.create(PETSc.COMM_WORLD)
        dml2.setTAOType(PETSc.DM.TAOType.L2)

        # Solve built-in version
        dml1.applyTAOproximalmap(dml2, thresh, y2, x_l1_test, False)

        # Solve MyDMTao version
        dm.applyTAOproximalmap(dml2, thresh, y1, x, False)

        x.view()
        x_l1_test.view()

        self.assertTrue(x.equal(x_l1_test))
        #self.assertTrue(ctx.log['solve'] == 1)
        #self.assertTrue(ctx.log['setUp'] == 1)
        #self.assertTrue(ctx.log['setFromOptions'] == 1)
        dml1.destroy()
        dml2.destroy()
        x_l1_test.destroy()
        x.destroy()
        y1.destroy()
        y2.destroy()
        print("test prox end")



#Build-in DMTAO
#OptDB = PETSc.Options()
#
#n = OptDB.getInt('n', 10)
#thresh  = OptDB.getReal('lambda', 0.1)
#
#dm0 = PETSc.DM()
#dm0.create(PETSc.COMM_WORLD)
#dm0.setTAOType(PETSc.DM.TAOType.L1)
#
#dm1 = PETSc.DM()
#dm1.create(PETSc.COMM_WORLD)
#dm1.setTAOType(PETSc.DM.TAOType.L2)
#
#y1 = PETSc.Vec()
#y1.create(PETSc.COMM_WORLD)
#y1.setSizes(n)
#y1.setType(PETSc.Vec.Type.SEQ)
#y1.setRandom()
#y2 = y1.duplicate()
#y1.copy(y2)
#
#x1 = PETSc.Vec()
#x1.create(PETSc.COMM_WORLD)
#x1.setSizes(n)
#x1.setType(PETSc.Vec.Type.SEQ)
#x1.set(0)
#x2 = x1.duplicate()
#x1.copy(x2)
#
#dm0.applyTAOproximalmap(dm1, thresh, y1, x1, False)
#
## DMTAOPYTHON
#dmpy0 = PETSc.DM()
#dmpy0.create(PETSc.COMM_WORLD)
#dmpy0.createTAOPython(MyDMTao(), comm=PETSc.COMM_WORLD)
#dmpy0.applyTAOproximalmap(dm1, thresh,  y2, x2, False)
#
#y1.viewFromOptions('-view_input_builtin')
#y2.viewFromOptions('-view_input_py')
#x1.viewFromOptions('-view_sol_builtin')
#x2.viewFromOptions('-view_sol_py')
#
#test = x1.equal(x2)
#print(test)


# --------------------------------------------------------------------

# --------------------------------------------------------------------

if np.iscomplexobj(PETSc.ScalarType()):
    del TestDMTaoPython

if __name__ == '__main__':
    unittest.main()

# --------------------------------------------------------------------
