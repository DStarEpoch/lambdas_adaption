/*main file for this module*/
#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif /* PY_SSIZE_T_CLEAN */

#include "Python.h"
#ifndef Py_PYTHON_H
    #error Python headers needed to compile C extensions, please install development version of Python.
#elif PY_VERSION_HEX < 0x02060000 || (0x03000000 <= PY_VERSION_HEX && PY_VERSION_HEX < 0x03030000)
    #error Cython requires Python 2.6+ or Python 3.3+.
#endif

#include "dp_info.h"
#include "dp_shortest_path_optimizer.h"

static PyModuleDef DPOptimizerModule = {
    PyModuleDef_HEAD_INIT,
    "dp_optimizer",
    "Dynamic Programming optimization module",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_dp_optimizer(void)
{
    PyObject *m = PyModule_Create(&DPOptimizerModule);
    if (m == NULL)
        return NULL;

    if (PyType_Ready(&DPShortestPathOptimizerType) < 0)
        return NULL;

    if (PyType_Ready(&DPInfoType) < 0)
        return NULL;

    Py_INCREF(&DPShortestPathOptimizerType);
    PyModule_AddObject(m, "DPShortestPathOptimizer", (PyObject *) &DPShortestPathOptimizerType);
    Py_INCREF(&DPInfoType);
    PyModule_AddObject(m, "DPInfo", (PyObject *) &DPInfoType);
    return m;
}
