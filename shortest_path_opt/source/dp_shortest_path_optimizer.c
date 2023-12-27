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

#define ITEM_IN_LIST(list, item, size) ({ \
    bool _ret = false; \
    for (int _i = 0; _i < size; _i++) { \
        if (list[_i] == item) { \
            _ret = true; \
            break; \
        } \
    } \
    _ret; \
})

#include "dp_info.h"

typedef struct _DPShortestPathOptimizer {
    PyObject_HEAD
    double **distance_matrix;
    int distance_matrix_size;
    int *retain_lambda_idx;
    int retain_lambda_idx_size;
}DPShortestPathOptimizer;

static PyTypeObject DPShortestPathOptimizerType;

static void
DPShortestPathOptimizer_dealloc(DPShortestPathOptimizer* self) {
    for (int i = 0; i < self->distance_matrix_size; i++) {
        free(self->distance_matrix[i]);
    }
    free(self->distance_matrix);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
DPShortestPathOptimizer_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    DPShortestPathOptimizer *self;
    self = (DPShortestPathOptimizer *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->distance_matrix = NULL;
    }
    return (PyObject *)self;
}

static int
DPShortestPathOptimizer_init(DPShortestPathOptimizer *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"distance_matrix", "retain_lambda_idx", NULL};
    PyObject *distance_matrix_obj;
    PyObject *retain_lambda_idx_obj;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, &distance_matrix_obj, &retain_lambda_idx_obj)) {
        return -1;
    }

    if (!PyList_Check(distance_matrix_obj)) {
        PyErr_SetString(PyExc_TypeError, "distance_matrix must be list");
        return -1;
    }

    if (!PyList_Check(retain_lambda_idx_obj)) {
        PyErr_SetString(PyExc_TypeError, "retain_lambda_idx must be list");
        return -1;
    }

    self->distance_matrix_size = PyList_Size(distance_matrix_obj);
    self->distance_matrix = (double **)malloc(sizeof(double *) * self->distance_matrix_size);
    for (int i = 0; i < self->distance_matrix_size; i++) {
        PyObject *row_obj = PyList_GetItem(distance_matrix_obj, i);
        if (!PyList_Check(row_obj)) {
            PyErr_SetString(PyExc_TypeError, "distance_matrix must be list of list");
            return -1;
        }
        int row_size = PyList_Size(row_obj);
        if (row_size != self->distance_matrix_size) {
            PyErr_SetString(PyExc_TypeError, "distance_matrix must be square matrix");
            return -1;
        }
        self->distance_matrix[i] = (double *)malloc(sizeof(double) * self->distance_matrix_size);
        for (int j = 0; j < self->distance_matrix_size; j++) {
            PyObject *distance_obj = PyList_GetItem(row_obj, j);
            if (!PyFloat_Check(distance_obj)) {
                PyErr_SetString(PyExc_TypeError, "distance_matrix must be list of list of float");
                return -1;
            }
            self->distance_matrix[i][j] = PyFloat_AsDouble(distance_obj);
        }
    }

    self->retain_lambda_idx_size = PyList_Size(retain_lambda_idx_obj);
    self->retain_lambda_idx = (int *)malloc(sizeof(int) * self->retain_lambda_idx_size);
    for (int i = 0; i < self->retain_lambda_idx_size; i++) {
        PyObject *idx_obj = PyList_GetItem(retain_lambda_idx_obj, i);
        if (!PyLong_Check(idx_obj)) {
            PyErr_SetString(PyExc_TypeError, "retain_lambda_idx must be list of int");
            return -1;
        }
        self->retain_lambda_idx[i] = PyLong_AsLong(idx_obj);
    }

    return 0;
}

static PyObject *
DPShortestPathOptimizer_get_distance_matrix(DPShortestPathOptimizer *self, void *closure) {
    Py_INCREF(self->distance_matrix);
    return self->distance_matrix;
}

static PyGetSetDef DPShortestPathOptimizer_getsetters[] = {
    {"distance_matrix", (getter)DPShortestPathOptimizer_get_distance_matrix, NULL, "distance_matrix", NULL},
    {NULL}  /* Sentinel */
};

static PyMemberDef DPShortestPathOptimizer_members[] = {
    {NULL}  /* Sentinel */
};

static PyObject *
DPShortestPathOptimizer_optimize(DPShortestPathOptimizer *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"target_lambda_num", NULL};
    int target_lambda_num;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", kwlist, &target_lambda_num)) {
        return NULL;
    }

    if (target_lambda_num <= self->retain_lambda_idx_size) {
        PyObject *list = PyList_New(self->retain_lambda_idx_size);
        for (int i = 0; i < self->retain_lambda_idx_size; i++) {
            PyList_SetItem(list, i, PyLong_FromLong(self->retain_lambda_idx[i]));
        }
        return Py_BuildValue("fO", 0.0, list);
    }

    int n = self->distance_matrix_size;
    int m = target_lambda_num - self->retain_lambda_idx_size;

    // initialize DPInfo
    // dp[i][j][k]: optimal path from lambda_0 to lambda_i with j selected points and try to insert lambda_k
    DPInfo* ***dp = (DPInfo* ***)malloc(sizeof(DPInfo* **) * n);
    for (int i = 0; i < n; i++) {
        dp[i] = (DPInfo* **)malloc(sizeof(DPInfo* *) * m);
        for (int j = 0; j < m; j++) {
            dp[i][j] = (DPInfo* *)malloc(sizeof(DPInfo*) * (i + 1));
            for (int k = 0; k < i + 1; k++) {
                // new
                dp[i][j][k] = newDPInfo(k);
            }
        }
        // init dp[i][0][k] for no selected points
        for (int k = 0; k < i + 1; k++) {
            dp[i][0][k]->cost = self->distance_matrix[0][k] + self->distance_matrix[k][i];
        }
    }

    // DP
    for (int i = 0; i < n; i++) {
        for (int j = 1; j < m; j++) {
            for (int k = 0; k < i + 1; k++) {
                if (ITEM_IN_LIST(self->retain_lambda_idx, k, self->retain_lambda_idx_size)) {
                    // skip retained lambdas
                    continue;
                }

                // Try selecting index k
                for (int k_prime=0; k_prime < i; k_prime++) {
                    if (ITEM_IN_LIST(self->retain_lambda_idx, k_prime, self->retain_lambda_idx_size)) {
                        // skip retained lambdas
                        continue;
                    }

                    double cost = dp[k_prime][j-1][k]->cost + self->distance_matrix[k_prime][k] + self->distance_matrix[k][i];
                    if (cost < dp[i][j][k]->cost) {
                        dp[i][j][k]->cost = cost;
                        dp[i][j][k]->setParent(dp[k_prime][j-1][k]);
                    }
                }
            }
        }
    }

    // find optimal path
    double min_cost = dp[n - 1][m - 1][0]->cost
    int min_cost_idx = 0;
    for (int k = 1; k < n; k++) {
        if (dp[n - 1][m - 1][k]->cost < min_cost) {
            min_cost = dp[n - 1][m - 1][k]->cost;
            min_cost_idx = k;
        }
    }

    // build optimal path
    PyObject *list = PyList_New(0);
    int* select_seq = dp[n-1][m-1][min_cost_idx]->getSequence();
    int select_seq_length = getDPInfoSequenceLength(dp[n-1][m-1][min_cost_idx]);
    for (int i = 0; i < n; i++) {
        if (ITEM_IN_LIST(self->retain_lambda_idx, i, self->retain_lambda_idx_size) ||
        ITEM_IN_LIST(select_seq, i, select_seq_length)) {
            PyList_Append(list, PyLong_FromLong(i));
        }
    }

    // free DPInfo
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < i + 1; k++) {
                freeDPInfo(dp[i][j][k]);
            }
            free(dp[i][j]);
        }
        free(dp[i]);
    }

    return Py_BuildValue("fO", min_cost, list);
}
