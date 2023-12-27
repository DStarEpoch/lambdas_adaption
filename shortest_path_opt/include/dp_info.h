#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "math.h"

typedef struct _DPInfo
{
    struct _DPInfo *parent;
    int latest_insert_idx;
    double cost;

    int (*getSequenceLength)(struct _DPInfo *self);
    int* (*getSequence)(struct _DPInfo *self);
    void (*setParent)(struct _DPInfo *self, struct _DPInfo *parent);
}DPInfo;

int getDPInfoSequenceLength(DPInfo *self);
int *getDPInfoSequence(DPInfo *self);
void setDPInfoParent(DPInfo *self, DPInfo *parent);

void initDPInfo(DPInfo *self, int latest_insert_idx);
void freeDPInfo(DPInfo *self);

DPInfo *newDPInfo(int latest_insert_idx);

static PyTypeObject DPInfoType;
