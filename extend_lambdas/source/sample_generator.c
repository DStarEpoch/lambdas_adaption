#include "sample_generator.h"

static PyObject *
SampleGeneratorModule_testFunc(PyObject *self, PyObject *args, PyObject *kwds)
{

    PyObject *op;
    op = LambdaInfoContext_createFromProperties(&LambdaInfoContextObjectType, args);
    if (op) {
        printf("SampleGeneratorModule_testFunc 111\n");
        return (PyObject *) op;
    }
    printf("SampleGeneratorModule_testFunc 222\n");
    Py_RETURN_NONE;
}

static PyMethodDef SampleGeneratorModule_methods[] = {
    {"test", (PyCFunction)SampleGeneratorModule_testFunc, METH_VARARGS | METH_KEYWORDS,
    "test function\n"},
    {NULL}  /* Sentinel */
};

static PyModuleDef SampleGeneratorModule = {
    PyModuleDef_HEAD_INIT,
    "sample_generator",
    "Dynamic Programming optimization module",
    -1,
    SampleGeneratorModule_methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_sample_generator(void)
{
    PyObject *m = PyModule_Create(&SampleGeneratorModule);
    if (m == NULL)
        return NULL;

    if (PyType_Ready(&LambdaInfoContextObjectType) < 0)
        return NULL;

    Py_INCREF(&LambdaInfoContextObjectType);
    PyModule_AddObject(m, "LambdaInfoContext", (PyObject *) &LambdaInfoContextObjectType);
    return m;
}
