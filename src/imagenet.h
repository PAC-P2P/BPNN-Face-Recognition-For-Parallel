//
// Created by xx on 17/7/13.
//

#ifndef FACEREC_IMAGENET_H
#define FACEREC_IMAGENET_H

#include <stdio.h>
#include <string.h>
#include <cstl/cmap.h>
#include "pgmimage.h"
#include "backprop.h"

#define TARGET_HIGH 0.9
#define TARGET_LOW 0.1

int load_target(IMAGE *img,BPNN *net, map_t *map_user);
int load_input_with_image(IMAGE *img,BPNN *net);


#endif //FACEREC_IMAGENET_H
