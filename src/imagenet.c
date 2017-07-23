/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *      Prepared for 15-681, Fall 1994.
 *
 ******************************************************************
 */

#include "imagenet.h"

extern void exit();

/* 避免使用0和1作为目标值的原因是sigmoid单元对于有限权值不能产生这样的输出,
   如果我们企图用训练网络来准确匹配目标值0和1，梯度下降将会迫使权值无限增长。
   值0.1和0.9是sigmoid函数在有限权值情况下可以完成的。
*/

/*** This is the target output encoding for a network with one output unit.
     It scans the image name, and if it's an image of me (js) then
     it sets the target unit to HIGH; otherwise it sets it to LOW.
     Remember, units are indexed starting at 1, so target unit 1
     is the one to change....  ***/

/***
    目标输出编码
***/

int load_target(IMAGE *img,BPNN *net, map_t *map_user)
{
    int scale;
    char userid[40];

    userid[0] = '\0';

    int userNum = map_size(map_user);

    /*** scan in the image features ***/
    sscanf(NAME(img), "%[^_]", userid);

    // 对 net->target 数组中每个元素初始化为 TARGET_LOW
    for(int i = 1; i <= userNum; ++i)
    {
        net->target[i] = TARGET_LOW;
    }

    // map迭代器
    map_iterator_t iterator;

    // 遍历map
    for (iterator = map_begin(map_user); !iterator_equal(iterator, map_end(map_user)); iterator = iterator_next(iterator)) {

        // userid等于map里的某个用户
        if(!strcmp(userid, (char *) pair_first(iterator_get_pointer(iterator))))
        {
            // 则此用户对应的输出结点标记为 TARGET_HIGH
            net->target[*(int *) pair_second(iterator_get_pointer(iterator))] = TARGET_HIGH;

            //printf("target[%d]= %f, <%s, %d>\n",*(int *) pair_second(iterator_get_pointer(iterator)),net->target[*(int *) pair_second(iterator_get_pointer(iterator))], (char *) pair_first(iterator_get_pointer(iterator)), *(int *) pair_second(iterator_get_pointer(iterator)) );
          }
    }
 
  return 0;
}


/***********************************************************************/
/********* You shouldn't need to change any of the code below.   *******/
/***********************************************************************/

int load_input_with_image(IMAGE *img,BPNN *net)
{
  double *units;
  int nr, nc, imgsize, i, j, k;

  nr = ROWS(img);
  nc = COLS(img);
  imgsize = nr * nc;;
  if (imgsize != net->input_n) {
    printf("LOAD_INPUT_WITH_IMAGE: This image has %d pixels,\n", imgsize);
    printf("   but your net has %d input units.  I give up.\n", net->input_n);
    exit (-1);
  }

  units = net->input_units;
  k = 1;
  for (i = 0; i < nr; i++) {
    for (j = 0; j < nc; j++) {
      units[k] = ((double) img_getpixel(img, i, j)) / 255.0;
      k++;
    }
  }
  return 0;
}
