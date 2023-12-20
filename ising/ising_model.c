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

typedef struct _IsingModel
{
    PyObject_HEAD
    int N;
    double beta;
    long *spins;
}IsingModel;

static PyTypeObject IsingModelType;

static void
IsingModel_dealloc(IsingModel* self)
{
    free(self->spins);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject *
IsingModel_repr(IsingModel *self)
{
    char repr_str[128];
    sprintf(repr_str, "IsingModel(N=%d, beta=%f)", self->N, self->beta);
    PyObject *repr = PyUnicode_FromFormat(repr_str);
    if (repr == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for repr");
        return NULL;
    }
    return repr;
}

static PyObject *
IsingModel_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    IsingModel *self;

    self = (IsingModel *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->N = 50;
        self->beta = 0.4;
        self->spins = NULL;
    }

    return (PyObject *)self;
}

static int
IsingModel_init(IsingModel *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"N", "beta", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|if", kwlist,
                                      &self->N, &self->beta))
        return -1;

    time_t t;
    srand((unsigned) time(&t));

    self->spins = malloc(self->N * self->N * sizeof(long));
    if (self->spins == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for spins");
        return -1;
    }

    for (long i = 0; i < self->N * self->N; i++) {
        self->spins[i] = (rand() % 2) * 2 - 1;
    }

    return 0;
}

static PyObject *
IsingModel_get_spins(IsingModel *self, void *closure)
{
    PyObject *list = PyList_New(self->N * self->N);
    if (list == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for spins");
        return NULL;
    }

    for (long i = 0; i < self->N * self->N; i++) {
        PyList_SET_ITEM(list, i, PyLong_FromLong(self->spins[i]));
    }

    return list;
}

static PyObject *
IsingModel_set_spins(IsingModel *self, PyObject *value, void *closure)
{
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the spins attribute");
        return NULL;
    }

    if (! PyList_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "The spins attribute value must be a list");
        return NULL;
    }

    if (PyList_Size(value) != self->N * self->N) {
        PyErr_SetString(PyExc_ValueError, "The spins attribute value must be a list of length N*N");
        return NULL;
    }

    for (long i = 0; i < self->N * self->N; i++) {
        self->spins[i] = PyLong_AsLong(PyList_GetItem(value, i));
    }

    return 0;
}

static PyObject *
IsingModel_properties(IsingModel *self, void *closure)
{
    PyObject *list = PyList_New(0);
    if (list == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for properties");
        return NULL;
    }
    PyList_Append(list, PyLong_FromLong(self->N));
    PyList_Append(list, PyFloat_FromDouble(self->beta));
    PyList_Append(list, IsingModel_get_spins(self, NULL));
    return list;
}

static double
get_energy(IsingModel *self)
{
    double energy = 0;

    for (int i = 0; i < self->N; i++) {
        for (int j = 0; j < self->N; j++) {
            energy += -self->spins[i * self->N + j] * (
                self->spins[((i - 1 + self->N) % self->N) * self->N + j] +
                self->spins[((i + 1) % self->N) * self->N + j] +
                self->spins[i * self->N + (j - 1 + self->N) % self->N] +
                self->spins[i * self->N + (j + 1) % self->N]
            );
        }
    }

    return energy / 4;
}

static PyObject *
IsingModel_get_energy(IsingModel *self, void *closure)
{
    return PyFloat_FromDouble(get_energy(self));
}

static PyObject *
IsingModel_get_dimless_energy(IsingModel *self, void *closure)
{
    return PyFloat_FromDouble(get_energy(self) * self->beta);
}

static PyObject *
IsingModel_mcMove(IsingModel *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"total_steps", "sample_steps", NULL};
    long *total_steps = malloc(sizeof(long));
    long *sample_steps = malloc(sizeof(long));
    *total_steps = 1;
    *sample_steps = 1;

    // parse arguments for total_steps and sample_steps, otherwise use default values
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|ll", kwlist,
                                      total_steps, sample_steps)) {
        free(total_steps);
        free(sample_steps);
        return NULL;
    }

    PyObject *energy_samples = PyList_New(*total_steps);
    if (energy_samples == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for energy trajectory");
        return NULL;
    }

    PyObject *spins_samples = PyList_New((*total_steps) / (*sample_steps));
    if (spins_samples == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for spins trajectory");
        return NULL;
    }

    for (long step=0; step < (*total_steps); step++) {
        for (long k = 0; k < self->N * self->N; k++) {
            int i = rand() % self->N;
            int j = rand() % self->N;

            int deltaE = 2 * self->spins[i * self->N + j] * (
                self->spins[((i - 1 + self->N) % self->N) * self->N + j] +
                self->spins[((i + 1) % self->N) * self->N + j] +
                self->spins[i * self->N + (j - 1 + self->N) % self->N] +
                self->spins[i * self->N + (j + 1) % self->N]
            );

            if (deltaE <= 0 || (float)rand() / RAND_MAX < exp(-self->beta * deltaE)) {
                self->spins[i * self->N + j] *= -1;
            }
        }

        if (step % (*sample_steps) == (*sample_steps) - 1) {
            PyList_SET_ITEM(spins_samples, step / (*sample_steps), IsingModel_get_spins(self, NULL));
        }
        PyList_SET_ITEM(energy_samples, step, IsingModel_get_energy(self, NULL));
    }

    return Py_BuildValue("OO", energy_samples, spins_samples);
}

static PyObject *
IsingModel_dimlessEnergyOnSpins(IsingModel *self, PyObject *args)
{
    PyObject *spins;
    if (! PyArg_ParseTuple(args, "O", &spins))
        return NULL;

    if (! PyList_Check(spins)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list");
        return NULL;
    }

    if (PyList_Size(spins) != self->N * self->N) {
        PyErr_SetString(PyExc_ValueError, "Argument must be a list of length N*N");
        return NULL;
    }

    double energy = 0;

    for (int i = 0; i < self->N; i++) {
        for (int j = 0; j < self->N; j++) {
            energy += -PyLong_AsLong(PyList_GetItem(spins, i * self->N + j)) * (
                PyLong_AsLong(PyList_GetItem(spins, ((i - 1 + self->N) % self->N) * self->N + j)) +
                PyLong_AsLong(PyList_GetItem(spins, ((i + 1) % self->N) * self->N + j)) +
                PyLong_AsLong(PyList_GetItem(spins, i * self->N + (j - 1 + self->N) % self->N)) +
                PyLong_AsLong(PyList_GetItem(spins, i * self->N + (j + 1) % self->N))
            );
        }
    }

    return PyFloat_FromDouble(energy * self->beta / 4);
}

static PyObject *
IsingModel_createFromProperties(PyObject *cls, PyObject *args)
{
    // classmethod
    PyObject *properties;
    if (! PyArg_ParseTuple(args, "O", &properties))
        return NULL;

    if (! PyList_Check(properties)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a list");
        return NULL;
    }

    if (PyList_Size(properties) != 3) {
        PyErr_SetString(PyExc_ValueError, "Argument must be a list of length 3");
        return NULL;
    }

    PyObject *N = PyList_GetItem(properties, 0);
    PyObject *beta = PyList_GetItem(properties, 1);
    PyObject *spins = PyList_GetItem(properties, 2);

    if (! PyLong_Check(N)) {
        PyErr_SetString(PyExc_TypeError, "N must be an integer");
        return NULL;
    }

    if (! PyFloat_Check(beta)) {
        PyErr_SetString(PyExc_TypeError, "beta must be a float");
        return NULL;
    }

    if (! PyList_Check(spins)) {
        PyErr_SetString(PyExc_TypeError, "spins must be a list");
        return NULL;
    }

    if (PyList_Size(spins) != PyLong_AsLong(N) * PyLong_AsLong(N)) {
        PyErr_SetString(PyExc_ValueError, "spins must be a list of length N*N");
        return NULL;
    }

    IsingModel *self = (IsingModel *)PyObject_CallObject((PyObject *)&IsingModelType, NULL);
    if (self == NULL) {
        return NULL;
    }

    self->N = PyLong_AsLong(N);
    self->beta = PyFloat_AsDouble(beta);

    for (long i = 0; i < self->N * self->N; i++) {
        self->spins[i] = PyLong_AsLong(PyList_GetItem(spins, i));
    }

    return (PyObject *)self;
}

static PyMemberDef IsingModel_members[] = {
    {"N", T_INT, offsetof(IsingModel, N), 0,
     "Size of the lattice"},
    {"beta", T_FLOAT, offsetof(IsingModel, beta), 0,
     "Inverse temperature"},
    {NULL}  /* Sentinel */
};

static PyGetSetDef IsingModel_getters[] = {
    {"spins", (getter)IsingModel_get_spins, (setter)IsingModel_set_spins,
     "Spins", NULL},
    {"energy", (getter)IsingModel_get_energy, NULL,
     "Energy", NULL},
    {"dimless_energy", (getter)IsingModel_get_dimless_energy, NULL,
     "Dimensionless energy", NULL},
    {"properties", (getter)IsingModel_properties, NULL, "Properties", NULL},
    {NULL}  /* Sentinel */
};

static PyMethodDef IsingModel_methods[] = {
    {"mcMove", (PyCFunction)IsingModel_mcMove, METH_VARARGS | METH_KEYWORDS,
        "mcMove(total_steps=1, sample_steps=1)\n"
        "--\n"
        "\n"
        "Monte Carlo move\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "total_steps : int, optional\n"
        "    Total number of steps to perform\n"
        "sample_steps : int, optional\n"
        "    Number of steps between each sample\n"
        "\n"
        "Returns\n"
        "-------\n"
        "energy_samples : list\n"
        "    List of energy samples\n"
        "spins_samples : list\n"
        "    List of spin configurations\n"
    },
    {"dimlessEnergyOnSpins", (PyCFunction)IsingModel_dimlessEnergyOnSpins, METH_VARARGS,
     "Calculate the dimensionless energy on a given spin configuration at the current temperature"},
    {"createFromProperties", (PyCFunction)IsingModel_createFromProperties, METH_CLASS | METH_VARARGS,
     "Create an IsingModel object from properties"},
    {NULL}  /* Sentinel */
};

static PyTypeObject IsingModelType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name="ising.IsingModel",
    .tp_repr=(reprfunc)IsingModel_repr,
    .tp_basicsize=sizeof(IsingModel),
    .tp_itemsize=0,
    .tp_flags=Py_TPFLAGS_DEFAULT,
    .tp_doc="IsingModel objects",
    .tp_dealloc=(destructor)IsingModel_dealloc,
    .tp_new=IsingModel_new,
    .tp_init=(initproc)IsingModel_init,
    .tp_members=IsingModel_members,
    .tp_getset=IsingModel_getters,
    .tp_methods=IsingModel_methods,
};

static PyModuleDef IsingModelModule = {
    PyModuleDef_HEAD_INIT,
    "ising",
    "Ising Model",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_ising(void)
{
    PyObject *m;

    if (PyType_Ready(&IsingModelType) < 0)
        return NULL;

    m = PyModule_Create(&IsingModelModule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&IsingModelType);
    PyModule_AddObject(m, "IsingModel", (PyObject *)&IsingModelType);
    return m;
}

