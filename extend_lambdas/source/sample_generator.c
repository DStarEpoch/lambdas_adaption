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
        self->f_k = NULL;
    }
    return (PyObject *) self;
}

static int
SampleGeneratorObject_init(SampleGeneratorObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"lambda_num", "samples_per_lambda", "org_u_nks", "f_k", NULL};

    PyObject *org_u_nks;
    PyObject *f_k;

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "llOO", kwlist,
                                      &self->lambda_num, &self->samples_per_lambda,
                                      &org_u_nks, &f_k))
        return -1;

    if (!PyList_Check(org_u_nks)) {
        PyErr_SetString(PyExc_TypeError, "org_u_nks must be list");
        return -1;
    }

    if (!PyList_Check(f_k)) {
        PyErr_SetString(PyExc_TypeError, "f_k must be list");
        return -1;
    }

    self->org_u_nks = malloc(self->lambda_num * self->lambda_num * self->samples_per_lambda * sizeof(double));
    for (long i = 0; i < self->lambda_num; i++) {
        PyObject *sample_from_lambda = PyList_GetItem(org_u_nks, i);
        for (long j = 0; j < self->lambda_num; j++) {
            PyObject *eval_potential_at_lambda = PyList_GetItem(sample_from_lambda, j);
            for (long k=0; k < self->samples_per_lambda; k++) {
                double potential = PyFloat_AsDouble(PyList_GetItem(eval_potential_at_lambda, k));
                self->org_u_nks[i * (self->lambda_num * self->samples_per_lambda) + j * self->samples_per_lambda + k] = potential;
            }
        }
    }

    self->f_k = malloc(self->lambda_num * sizeof(double));
    for (Py_ssize_t i = 0; i < self->lambda_num; i++) {
        double f = PyFloat_AsDouble(PyList_GetItem(f_k, i));
        self->f_k[i] = f;
    }

    return 0;
}

static PyObject *
SampleGeneratorObject_get_org_u_nks(SampleGeneratorObject *self, void *closure)
{
    PyObject *list = PyList_New(self->lambda_num);
    if (list == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for org_u_nks");
        return NULL;
    }

    for (long i = 0; i < self->lambda_num; i++) {
        PyObject *sample_from_lambda = PyList_New(self->lambda_num);
        for (long j = 0; j < self->lambda_num; j++) {
            PyObject *eval_potential_at_lambda = PyList_New(self->samples_per_lambda);
            for (long k = 0; k < self->samples_per_lambda; k++) {
                double potential = self->org_u_nks[i * (self->lambda_num * self->samples_per_lambda) + j * self->samples_per_lambda + k];
                PyList_SET_ITEM(eval_potential_at_lambda, k, PyFloat_FromDouble(potential));
            }
            PyList_SET_ITEM(sample_from_lambda, j, eval_potential_at_lambda);
        }
        PyList_SET_ITEM(list, i, sample_from_lambda);
    }

    return Py_BuildValue("O", list);
}

static PyObject *
SampleGeneratorObject_get_f_k(SampleGeneratorObject *self, void *closure)
{
    PyObject *list = PyList_New(self->lambda_num);
    if (list == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for f_k");
        return NULL;
    }

    for (long k = 0; k < self->lambda_num; k++) {
        double f = self->f_k[k];
        PyList_SET_ITEM(list, k, PyFloat_FromDouble(f));
    }

    return Py_BuildValue("O", list);
}

typedef struct _PickTag {
    long lambda_idx;
    long sample_idx_at_lambda;
    double weight;
    double stop_weight;
}PickTag;

static PyObject *
SampleGeneratorObject_genSamplesForInsertLambda(SampleGeneratorObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"insert_lambdas_pos", NULL};

    PyObject *insert_lambdas_pos;

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &insert_lambdas_pos)) {
        return NULL;
    }

    if (!PyList_Check(insert_lambdas_pos)) {
        PyErr_SetString(PyExc_TypeError, "insert_lambdas_pos must be list");
        return NULL;
    }

    time_t t;
    srand((unsigned) time(&t));

    Py_ssize_t insert_lambdas_size = PyList_Size(insert_lambdas_pos);
    PyObject *insert_lambdas_info = PyList_New(insert_lambdas_size);
    PickTag* *pick_tags_list = (PickTag* *)malloc(sizeof(PickTag*) * insert_lambdas_size);

    // pre-compute free energy of inserted lambdas and picking list
    for (Py_ssize_t i = 0; i < insert_lambdas_size; i++) {
        long start_lambda_idx;
        long end_lambda_idx;
        double ratio;

        // check value of insert_lambdas_pos
        if (! PyArg_ParseTuple(PyList_GetItem(insert_lambdas_pos, i), "lld",
        &start_lambda_idx, &end_lambda_idx, &ratio)) {
            char err_msg[128];
            sprintf(err_msg, "insert_lambdas_pos[%ld] cannot parsed as Tuple[start_lambda_idx:int, end_lambda_idx:int, ratio:float]\n", i);
            PyErr_SetString(PyExc_TypeError, err_msg);
            return NULL;
        }

        if (start_lambda_idx < 0 || start_lambda_idx >= self->lambda_num) {
            char err_msg[128];
            sprintf(err_msg, "start_lambda_idx=%ld at insert_lambdas_pos[%ld] is out of range [0, %ld] of org_u_nks\n",
            start_lambda_idx, i, self->lambda_num);
            PyErr_SetString(PyExc_TypeError, err_msg);
            return NULL;
        }

        if (end_lambda_idx < 0 || end_lambda_idx >= self->lambda_num) {
            char err_msg[128];
            sprintf(err_msg, "end_lambda_idx=%ld at insert_lambdas_pos[%ld] is out of range [0, %ld] of org_u_nks\n",
            end_lambda_idx, i, self->lambda_num);
            PyErr_SetString(PyExc_TypeError, err_msg);
            return NULL;
        }
        // check end

        // compute probability of all samples in current inserted lambda
        // f_λ = -lnΣ_jΣ_n{ exp[-u_λ(x_jn)] / Σ_k{exp[f_k-u_k(x_jn)]} }
        double sum_p = 0.0;
        double **all_u_samples = (double **)malloc(sizeof(double *) * self->lambda_num);

        for (Py_ssize_t j = 0; j < self->lambda_num; j++) {
            double *p_over_bias_list = (double *)malloc(sizeof(double) * self->samples_per_lambda);
            for (Py_ssize_t k = 0; k < self->samples_per_lambda; k ++) {
                double bias = 0.0;
                for (Py_ssize_t l = 0; l < self->lambda_num; l ++) {
                    bias += exp(self->f_k[l] - self->org_u_nks[j*(self->lambda_num*self->samples_per_lambda) + l*self->samples_per_lambda + k]);
                }
                double p_over_bias = exp(-(self->org_u_nks[j*(self->lambda_num*self->samples_per_lambda) + start_lambda_idx*self->samples_per_lambda + k] * (1 - ratio)
                + self->org_u_nks[j*(self->lambda_num*self->samples_per_lambda) + end_lambda_idx*self->samples_per_lambda + k] * ratio));
                p_over_bias_list[k] = p_over_bias / (bias * self->samples_per_lambda);
                sum_p += p_over_bias_list[k];
            }
            all_u_samples[j] = p_over_bias_list;
        }
        double f_insert = -log(sum_p);

        LambdaInfoContextObject *op;
        op = (LambdaInfoContextObject *)createLambdaInfoContextObjFromArgs(start_lambda_idx, end_lambda_idx, ratio, 1, -1, f_insert);
        printf("lambda_idx: %ld, rank: %lf, str: %s, f_k: %lf\n", i, getRank(op->context), getStr(op->context), op->context->f_k);
        PyList_SET_ITEM(insert_lambdas_info, i, (PyObject *)op);

        // pick samples for inserted lambda by weight
        // p_λ(x) = exp[f_λ-u_λ(x)] / Σ_k{exp[f_k-u_k(x)]}
        PickTag *candidate_tags = (PickTag*)malloc(sizeof(PickTag) * self->samples_per_lambda * self->lambda_num);
        double stop_weight = 0.0;
        for (Py_ssize_t j = 0; j < self->lambda_num; j++) {
            for (Py_ssize_t k = 0; k < self->samples_per_lambda; k ++) {
                Py_ssize_t ptr = j * self->lambda_num + k;
                candidate_tags[ptr].stop_weight = stop_weight;
                candidate_tags[ptr].weight = all_u_samples[j][k] / sum_p;
                candidate_tags[ptr].lambda_idx = j;
                candidate_tags[ptr].sample_idx_at_lambda = k;
            }
        }
        pick_tags_list[i] = (PickTag*)malloc(sizeof(PickTag) * self->samples_per_lambda);
        for (Py_ssize_t j = 0; j < self->samples_per_lambda; j++) {
            double rand_choice_weight_at = rand() / RAND_MAX;
            long select_tag_idx = self->samples_per_lambda * self->lambda_num - 1;
            for (Py_ssize_t k = 0; k < self->samples_per_lambda * self->lambda_num; k++) {
                if (candidate_tags[k].stop_weight > rand_choice_weight_at) {
                    select_tag_idx = k;
                    break;
                }
            }
            pick_tags_list[i][j].stop_weight = candidate_tags[select_tag_idx].stop_weight;
            pick_tags_list[i][j].weight = candidate_tags[select_tag_idx].weight;
            pick_tags_list[i][j].lambda_idx = candidate_tags[select_tag_idx].lambda_idx;
            pick_tags_list[i][j].sample_idx_at_lambda = candidate_tags[select_tag_idx].sample_idx_at_lambda;
        }

        // free candidate_tags
        free(candidate_tags);
        // free all_u_samples
        for (Py_ssize_t j = 0; j < self->lambda_num; j++) {
            free(all_u_samples[j]);
        }
        free(all_u_samples);
        all_u_samples = NULL;
    }

    // generate fake sampling for inserted lambda and reorder u_nks for all lambdas


    // free pick_tags_list
    for (Py_ssize_t i = 0; i < insert_lambdas_size; i++) {
        free(pick_tags_list[i]);
    }
    free(pick_tags_list);
    pick_tags_list = NULL;

    return Py_BuildValue("O", insert_lambdas_info);
}

static PyGetSetDef SampleGeneratorObject_getsetters[] = {
    {"org_u_nks", (getter)SampleGeneratorObject_get_org_u_nks, NULL, "org_u_nks", NULL},
    {"f_k", (getter)SampleGeneratorObject_get_f_k, NULL, "f_k", NULL},
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
    {"genSamplesForInsertLambda",
    (PyCFunction)SampleGeneratorObject_genSamplesForInsertLambda,
    METH_VARARGS | METH_KEYWORDS,
     "genSamplesForInsertLambda(insert_lambdas_pos: List[Tuple[int, int, float]]) \n"
     "-> Tuple[bp_u_nks: List[List[List[float]]], insert_lambdas_info: List[LambdaInfoContext]]"},
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
    printf("failed to create LambdaInfoContext from properties\n");
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
