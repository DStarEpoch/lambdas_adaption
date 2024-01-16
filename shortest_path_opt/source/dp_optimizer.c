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

int ITEM_IN_LIST(long* list, long item, long size) {
    int _ret = 0;
    for (long _i = 0; _i < size; _i++) {
        if (list[_i] == item) {
            _ret = 1;
            break;
        }
    }
    return _ret;
}

// module dp_optimizer
static PyObject *
DPOptimizer_optimize(PyObject *self, PyObject *args, PyObject *kwds)
{

    static char *kwlist[] = {"distance_matrix", "retain_lambda_idx", "target_lambda_num", NULL};
    PyObject *distance_matrix_obj;
    PyObject *retain_lambda_idx_obj;
    long target_lambda_num;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOl", kwlist,
                                     &distance_matrix_obj, &retain_lambda_idx_obj,
                                     &target_lambda_num)) {
        return NULL;
    }

    if (!PyList_Check(distance_matrix_obj)) {
        PyErr_SetString(PyExc_TypeError, "distance_matrix must be list");
        return NULL;
    }

    if (!PyList_Check(retain_lambda_idx_obj)) {
        PyErr_SetString(PyExc_TypeError, "retain_lambda_idx must be list");
        return NULL;
    }

    long distance_matrix_size = PyList_Size(distance_matrix_obj);
    double **distance_matrix = (double **)malloc(sizeof(double *) * distance_matrix_size);
    for (long i = 0; i < distance_matrix_size; i++) {
        PyObject *row_obj = PyList_GetItem(distance_matrix_obj, i);
        if (!PyList_Check(row_obj)) {
            PyErr_SetString(PyExc_TypeError, "distance_matrix must be list of list");
            return NULL;
        }
        long row_size = PyList_Size(row_obj);
        if (row_size != distance_matrix_size) {
            PyErr_SetString(PyExc_TypeError, "distance_matrix must be square matrix");
            return NULL;
        }
        distance_matrix[i] = (double *)malloc(sizeof(double) * distance_matrix_size);
        for (long j = 0; j < distance_matrix_size; j++) {
            PyObject *distance_obj = PyList_GetItem(row_obj, j);
            if (!PyFloat_Check(distance_obj)) {
                PyErr_SetString(PyExc_TypeError, "distance_matrix must be list of list of float");
                return NULL;
            }
            distance_matrix[i][j] = PyFloat_AsDouble(distance_obj);
        }
    }

    long retain_lambda_idx_size = PyList_Size(retain_lambda_idx_obj);
    long* retain_lambda_idx = (long *)malloc(sizeof(long) * retain_lambda_idx_size);
    for (long i = 0; i < retain_lambda_idx_size; i++) {
        PyObject *idx_obj = PyList_GetItem(retain_lambda_idx_obj, i);
        if (!PyLong_Check(idx_obj)) {
            PyErr_SetString(PyExc_TypeError, "retain_lambda_idx must be list of int");
            return NULL;
        }
        retain_lambda_idx[i] = PyLong_AsLong(idx_obj);
    }

    // optimization
    if (target_lambda_num <= retain_lambda_idx_size)
    {
        double min_cost = 0.0;
        PyObject *list = PyList_New(retain_lambda_idx_size);
        for (long i = 0; i < retain_lambda_idx_size; i++)
        {
            PyList_SetItem(list, i, PyLong_FromLong(retain_lambda_idx[i]));
            if (i > 0) {
                min_cost += distance_matrix[retain_lambda_idx[i - 1]][retain_lambda_idx[i]];
            }
        }


        // free
        for (long i = 0; i < distance_matrix_size; i++) {
                free(distance_matrix[i]);
                distance_matrix[i] = NULL;
        }
        free(distance_matrix);
        distance_matrix = NULL;
        free(retain_lambda_idx);
        retain_lambda_idx = NULL;
        return Py_BuildValue("fO", min_cost, list);
    }

    long n = distance_matrix_size;
    long m = target_lambda_num - retain_lambda_idx_size;

    // initialize DPInfo
    // dp[i][j][k]: optimal path from lambda_0 to lambda_i with j selected points and try to insert lambda_k
    DPInfo* ***dp = (DPInfo* ***)malloc(sizeof(DPInfo* **) * n);
    for (long i = 0; i < n; i++)
    {
        dp[i] = (DPInfo* **)malloc(sizeof(DPInfo* *) * m);
        for (long j = 0; j < m; j++)
        {
            dp[i][j] = (DPInfo* *)malloc(sizeof(DPInfo*) * (i + 1));
            for (long k = 0; k < i + 1; k++)
            {
                // new
                dp[i][j][k] = newDPInfo(k);
            }
        }
        // init dp[i][0][k] for no selected points
        for (long k = 0; k < i + 1; k++)
        {
            dp[i][0][k]->cost = distance_matrix[0][k] + distance_matrix[k][i];
        }
    }

    // DP
    for (long i = 0; i < n; i++) {
        for (long j = 1; j < m; j++) {
            for (long k = 0; k < i + 1; k++) {
                if (ITEM_IN_LIST(retain_lambda_idx, k, retain_lambda_idx_size))
                {
                    // skip retained lambdas
                    continue;
                }

                // Try selecting index k
                for (long k_prime=0; k_prime < k; k_prime++) {
                    if (ITEM_IN_LIST(retain_lambda_idx, k_prime, retain_lambda_idx_size))
                    {
                        // skip retained lambdas
                        continue;
                    }

                    double cost = dp[k_prime][j-1][k_prime]->cost + distance_matrix[k_prime][k] + distance_matrix[k][i];
                    if (cost < dp[i][j][k]->cost)
                    {
                        dp[i][j][k]->cost = cost;
                        dp[i][j][k]->setParent(dp[i][j][k], dp[k_prime][j-1][k_prime]);
                    }
                }
            }
        }
    }

    // find optimal path
    double min_cost = dp[n - 1][m - 1][0]->cost;
    long min_cost_idx = 0;
    for (long k = 1; k < n; k++) {
        if (dp[n - 1][m - 1][k]->cost < min_cost) {
            min_cost = dp[n - 1][m - 1][k]->cost;
            min_cost_idx = k;
        }
    }

    // build optimal path
    PyObject *list = PyList_New(0);
    long* select_seq = dp[n-1][m-1][min_cost_idx]->getSequence(dp[n-1][m-1][min_cost_idx]);
    long select_seq_length = getDPInfoSequenceLength(dp[n-1][m-1][min_cost_idx]);
    for (long i = 0; i < n; i++) {
        if (ITEM_IN_LIST(retain_lambda_idx, i, retain_lambda_idx_size) ||
        ITEM_IN_LIST(select_seq, i, select_seq_length)) {
            PyList_Append(list, PyLong_FromLong(i));
        }
    }

    // free DPInfo
    if (dp != NULL) {
        for (long i = 0; i < n; i++) {
            if (dp[i] != NULL) {
                for (long j = 0; j < m; j++) {
                    if (dp[i][j] != NULL) {
                        for (long k = 0; k < i + 1; k++) {
                            if (dp[i][j][k]) {
                                freeDPInfo(dp[i][j][k]);
                            }
                        }
                        free(dp[i][j]);
                        dp[i][j] = NULL;
                    }
                }
                free(dp[i]);
                dp[i] = NULL;
            }
        }
        free(dp);
        dp = NULL;
    }

    // free
    for (long i = 0; i < distance_matrix_size; i++) {
            free(distance_matrix[i]);
            distance_matrix[i] = NULL;
    }
    free(distance_matrix);
    distance_matrix = NULL;
    free(retain_lambda_idx);
    retain_lambda_idx = NULL;

    // return
    return Py_BuildValue("fO", min_cost, list);
}

static PyMethodDef DPOptimizer_methods[] = {
    {"optimize", (PyCFunction)DPOptimizer_optimize, METH_VARARGS | METH_KEYWORDS,
    "function of optimizing lambda selection\n"
    "optimize(distance_matrix: List[List[float]], target_lambda_num: int, retain_lambdas_idx: List[int])\n"
        "--\n"
        "\n"
        "find shortest pathway with certain points number as target_lambda_num by dynamic programming.\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "distance_matrix: List[List[float]], required\n"
        "    matrix of [-ΔλΔU]\n"
        "target_lambda_num : int, required\n"
        "    Total number of steps to perform\n"
        "retain_lambdas_idx : List[int], required"
        "    lambda indexes that forced to be retained\n"
        "Returns\n"
        "-------\n"
        "min_cost : float\n"
        "    minimal cost for optimal sequence\n"
        "optimal_sequence : list\n"
        "    optimal selected sequence of nodes for pathway\n"},
    {NULL}  /* Sentinel */
};

static PyModuleDef DPOptimizerModule = {
    PyModuleDef_HEAD_INIT,
    "dp_optimizer",
    "Dynamic Programming optimization module",
    -1,
    DPOptimizer_methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_dp_optimizer(void)
{
    PyObject *m = PyModule_Create(&DPOptimizerModule);
    if (m == NULL)
        return NULL;
    return m;
}
