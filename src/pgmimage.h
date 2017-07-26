/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *      Prepared for 15-681, Fall 1994.
 *
 ******************************************************************
 */

#ifndef BPNN_PGMIMAGE_H

#define BPNN_PGMIMAGE_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cstl/cmap.h>

typedef struct {
  char *name;
  int rows, cols;
  int *data;
} IMAGE;

typedef struct {
  int n;
  IMAGE **list;
} IMAGELIST;

/*** User accessible macros ***/

#define ROWS(img)  ((img)->rows)
#define COLS(img)  ((img)->cols)
#define NAME(img)   ((img)->name)

/*** User accessible functions ***/

IMAGE *img_open(char *);
IMAGE *img_creat(char *, int, int);
void img_setpixel(IMAGE *, int, int, int);
int img_getpixel(IMAGE *, int, int);
int img_write(IMAGE *, char *);
void img_free(IMAGE *);

void imgl_load_images_from_textfile(IMAGELIST *,char *, int, int);
void imgl_load_images_from_textfile_map(int *, char *, map_t *);

IMAGELIST *imgl_alloc();
void imgl_add(IMAGELIST *, IMAGE *);
void imgl_free(IMAGELIST *);
int imgl_munge_name(char *);

#endif
