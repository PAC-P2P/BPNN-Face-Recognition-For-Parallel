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
void backprop_face_choose(IMAGELIST *,int , BPNN *, int, int, double [], map_t *);
void backprop_face_parallel(BPNN *net, double **input_grad, double **hidden_grad, int epochs, double learning_rate, double momentum,
                            double **input_gobal_grad, double **hidden_gobal_grad, IMAGELIST *trainlist, IMAGELIST *testlist, int id, map_t *map_user);
#endif //BPNN_TRAIN_H
