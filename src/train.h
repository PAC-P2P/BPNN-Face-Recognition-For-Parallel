//
// Created by xx on 17/7/23.
//

#ifndef BPNN_TRAIN_H
#define BPNN_TRAIN_H

#include <mpi.h>
#include "pgmimage.h"
#include "backprop.h"
#include "evaluate.h"
#include "config.h"

int backprop_face(IMAGELIST *, IMAGELIST *, int, int, char *, int, map_t *);

void backprop_face_choose(IMAGELIST *,int, int , BPNN *, int, int, int, double*, map_t *);

void backprop_face_parallel(BPNN *, double ***, double ***, int , double , double ,
                            double ***, double ***, IMAGELIST *, IMAGELIST *, int , map_t *);
#endif //BPNN_TRAIN_H
