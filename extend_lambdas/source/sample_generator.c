#include "sample_generator.h"

// class SampleGenerator
static void
SampleGeneratorObject_dealloc(SampleGeneratorObject *self)
{
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static PyObject *
SampleGeneratorObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    SampleGeneratorObject *self;
    self = (SampleGeneratorObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->lambda_num = 0;
        self->samples_per_lambda = 0;
        self->org_u_nks = NULL;
    }
    return (PyObject *) self;
}

static int
SampleGeneratorObject_init(SampleGeneratorObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"lambda_num", "samples_per_lambda", "org_u_nks", NULL};

    PyObject *org_u_nks;

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "llO", kwlist,
                                      &self->lambda_num, &self->samples_per_lambda, &org_u_nks))
        return -1;

    if (!PyList_Check(org_u_nks)) {
        PyErr_SetString(PyExc_TypeError, "org_u_nks must be list");
        return -1;
    }

    self->org_u_nks = malloc(self->lambda_num * self->lambda_num * self->samples_per_lambda * sizeof(double));
    for (long i = 0; i < self->lambda_num; i++) {
        PyObject *sample_from_lambda = PyList_GetItem(org_u_nks, i);
        for (long j = 0; j < self->lambda_num; j++) {
            PyObject *eval_potential_at_lambda = PyList_GetItem(sample_from_lambda, j);
            for (long k=0; k < self->samples_per_lambda; k++) {
                double potential = PyFloat_AsDouble(PyList_GetItem(eval_potential_at_lambda, k));
                self->org_u_nks[i * self->lambda_num + j * self->lambda_num + k] = potential;
            }
        }
    }

    return 0;
}

static PyObject *
SampleGeneratorObject_get_org_u_nks(SampleGeneratorObject *self, void *closure)
{
    PyObject *list = PyList_New(self->lambda_num);
    if (list == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for distance_matrix");
        return NULL;
    }

    for (long i = 0; i < self->lambda_num; i++) {
        PyObject *sample_from_lambda = PyList_New(self->lambda_num);
        for (long j = 0; j < self->lambda_num; j++) {
            PyObject *eval_potential_at_lambda = PyList_New(self->samples_per_lambda);
            for (long k = 0; k < self->samples_per_lambda; k++) {
                double potential = self->org_u_nks[i * self->lambda_num + j * self->lambda_num + k];
                PyList_SET_ITEM(eval_potential_at_lambda, k, PyFloat_FromDouble(potential));
            }
            PyList_SET_ITEM(sample_from_lambda, j, eval_potential_at_lambda);
        }
        PyList_SET_ITEM(list, i, sample_from_lambda);
    }

    return Py_BuildValue("O", list);
}

static PyObject *
SampleGeneratorObject_genSamplesWithInsertLambda(SampleGeneratorObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"insert_lambdas_info", NULL};

    PyObject *insert_lambdas_info;

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &insert_lambdas_info)) {
        return NULL;
    }

    if (!PyList_Check(insert_lambdas_info)) {
        PyErr_SetString(PyExc_TypeError, "insert_lambdas_info must be list");
        return NULL;
    }

    time_t t;
    srand((unsigned) time(&t));

    Py_ssize_t insert_lambdas_size = PyList_Size(insert_lambdas_info);
    printf("insert_lambdas_size: %ld\n rand: %d\n rand_max: %d\n",
    insert_lambdas_size, rand(), RAND_MAX);

    // pre-compute free energy of inserted lambdas and picking list


    Py_RETURN_NONE;
}

static PyGetSetDef SampleGeneratorObject_getsetters[] = {
    {"org_u_nks", (getter)SampleGeneratorObject_get_org_u_nks, NULL, "org_u_nks", NULL},
    {NULL}  /* Sentinel */
};

static PyMemberDef SampleGeneratorObject_members[] = {
    {"lambda_num", T_INT, offsetof(SampleGeneratorObject, lambda_num), 0,
     "number of lambdas"},
    {"samples_per_lambda", T_INT, offsetof(SampleGeneratorObject, samples_per_lambda), 0,
     "number of potential samples at each lambda"},
    {NULL}  /* Sentinel */
};

static PyMethodDef SampleGeneratorObject_methods[] = {
    {"genSamplesWithInsertLambda",
    (PyCFunction)SampleGeneratorObject_genSamplesWithInsertLambda,
    METH_VARARGS | METH_KEYWORDS,
     "genSamplesWithInsertLambda(insert_lambdas_info: List[Tuple[int, int, float]]) -> Tuple[bp_u_nks, all_lambdas_info]"},
    {NULL}  /* Sentinel */
};

PyTypeObject SampleGeneratorObjectType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "sample_generator.SampleGenerator",
    .tp_basicsize = sizeof(SampleGeneratorObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)SampleGeneratorObject_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = "SampleGenerator objects\n"
              "SampleGenerator(lambda_num: int, samples_per_lambda: int, org_u_nks: List[List[List[float]]])\n\n",
    .tp_methods = SampleGeneratorObject_methods,
    .tp_members = SampleGeneratorObject_members,
    .tp_getset = SampleGeneratorObject_getsetters,
    .tp_init = (initproc)SampleGeneratorObject_init,
    .tp_new = SampleGeneratorObject_new,
};

// module sample_generator
static PyObject *
SampleGeneratorModule_copy_context(PyObject *self, PyObject *args, PyObject *kwds)
{

    PyObject *op;
    op = LambdaInfoContext_createFromProperties(&LambdaInfoContextObjectType, args);
    if (op) {
        return (PyObject *) op;
    }
    printf("failed to create LambdaInfoContext from properties");
    Py_RETURN_NONE;
}

static PyMethodDef SampleGeneratorModule_methods[] = {
    {"copyLambdaInfoContextFromProperties", (PyCFunction)SampleGeneratorModule_copy_context, METH_VARARGS | METH_KEYWORDS,
    "create a new LambdaInfoContext object from properties of an exist object\n"},
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

    if (PyType_Ready(&SampleGeneratorObjectType) < 0)
        return NULL;

    Py_INCREF(&LambdaInfoContextObjectType);
    PyModule_AddObject(m, "LambdaInfoContext", (PyObject *) &LambdaInfoContextObjectType);
    Py_INCREF(&SampleGeneratorObjectType);
    PyModule_AddObject(m, "SampleGenerator", (PyObject *) &SampleGeneratorObjectType);
    return m;
}
