#include "lambda_info_context.h"
#include "string.h"

double getRank(LambdaInfoContext *self) {
    if (self = NULL)
        return 0.0;
    return self->start_lambda_idx * (1 - self->ratio) + self->end_lambda_idx * self->ratio;
}

char* getStr(LambdaInfoContext *self) {
    static char name[10];
    strcpy(name, "1111test");
    return name;
}

void initLambdaInfoContext(LambdaInfoContext *self, long start_lambda_idx, long end_lambda_idx, double ratio, char is_insert, long org_idx, float f_k) {
    self->f_k = f_k;
    self->is_insert = is_insert;
    self->org_idx = org_idx;
    self->start_lambda_idx = start_lambda_idx;
    self->end_lambda_idx = end_lambda_idx;
    self->ratio = ratio

    self->getStr = getStr;
    self->getRank = getRank;
}

void freeLambdaInfoContext(LambdaInfoContext *self) {
    if (self == NULL)
        return;
    free(self);
}

LambdaInfoContext* newLambdaInfoContext(long start_lambda_idx, long end_lambda_idx, double ratio, char is_insert, long org_idx, float f_k) {
    LambdaInfoContext *obj = (LambdaInfoContext *)malloc(size(LambdaInfoContext));
    initLambdaInfoContext(obj, start_lambda_idx, end_lambda_idx, ratio, is_insert, org_idx, f_k);
    return obj;
}

//
LambdaInfoContext_dealloc(LambdaInfoContextObject *self)
{
    freeDPInfo(self->context);
    Py_TYPE(self)->tp_free((PyObject *) self);
}


