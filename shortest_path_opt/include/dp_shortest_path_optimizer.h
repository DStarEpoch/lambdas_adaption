#ifndef DP_SHORTEST_PATH_OPTIMIZER_H
#define DP_SHORTEST_PATH_OPTIMIZER_H

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif /* PY_SSIZE_T_CLEAN */

#include "Python.h"
#ifndef Py_PYTHON_H
    #error Python headers needed to compile C extensions, please install development version of Python.
#elif PY_VERSION_HEX < 0x02060000 || (0x03000000 <= PY_VERSION_HEX && PY_VERSION_HEX < 0x03030000)
    #error Cython requires Python 2.6+ or Python 3.3+.
#endif

#include "structmember.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <stddef.h>
#ifndef offsetof
  #define offsetof(type, member) ( (size_t) & ((type*)0) -> member )
#endif

typedef struct _DPShortestPathOptimizer {
    PyObject_HEAD
    double **distance_matrix;
    int distance_matrix_size;
    int *retain_lambda_idx;
    int retain_lambda_idx_size;
}DPShortestPathOptimizer;


extern PyTypeObject DPShortestPathOptimizerType;

#endif