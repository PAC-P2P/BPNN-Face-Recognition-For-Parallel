//
// Created by xx on 17/7/23.
//

#ifndef BPNN_EVALUATE_H
#define BPNN_EVALUATE_H

#include <cstl/cmap.h>
#include "backprop.h"
#include "imagenet.h"

#define false 0
#define true 1

int evaluate_performance(BPNN *, double *);
int performance_on_imagelist(BPNN *, IMAGELIST *, int , map_t *, double*);
void result_on_imagelist(BPNN *, IMAGELIST *, int , map_t *, double *, double *);


#endif //BPNN_EVALUATE_H
