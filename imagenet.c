/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *      Prepared for 15-681, Fall 1994.
 *
 ******************************************************************
 */

#include <stdio.h>
#include <string.h>
#include "pgmimage.h"
#include "backprop.h"

extern void exit();

/* 避免使用0和1作为目标值的原因是sigmoid单元对于有限权值不能产生这样的输出,
   如果我们企图用训练网络来准确匹配目标值0和1，梯度下降将会迫使权值无限增长。
   值0.1和0.9是sigmoid函数在有限权值情况下可以完成的。
*/
#define TARGET_HIGH 0.9 
#define TARGET_LOW 0.1


/*** This is the target output encoding for a network with one output unit.
     It scans the image name, and if it's an image of me (js) then
     it sets the target unit to HIGH; otherwise it sets it to LOW.
     Remember, units are indexed starting at 1, so target unit 1
     is the one to change....  ***/
/***
    目标输出编码
***/

int load_target(IMAGE *img,BPNN *net)
{
  int scale;
  char userid[40], head[40], expression[40], eyes[40], photo[40];

  userid[0] = head[0] = expression[0] = eyes[0] = photo[0] = '\0';

  /*** scan in the image features ***/
  sscanf(NAME(img), "%[^_]_%[^_]_%[^_]_%[^_]_%d.%[^_]",
    userid, head, expression, eyes, &scale, photo);
  //姓名     朝向     表情      眼镜    大小

  // 1 an2i      <.9, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1>
  // 2 at33      <.1, .9, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1>
  // 3 boland    <.1, .1, .9, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1>
  // 4 bpm       <.1, .1, .1, .9, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1>
  // 5 ch4f      <.1, .1, .1, .1, .9, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1>
  // 6 cheyer    <.1, .1, .1, .1, .1, .9, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1>
  // 7 choon     <.1, .1, .1, .1, .1, .1, .9, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1>
  // 8 danieln   <.1, .1, .1, .1, .1, .1, .1, .9, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1>
  // 9 glickman  <.1, .1, .1, .1, .1, .1, .1, .1, .9, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1>
  // 10 karyadi  <.1, .1, .1, .1, .1, .1, .1, .1, .1, .9, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1>
  // 11 kawamura <.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .9, .1, .1, .1, .1, .1, .1, .1, .1, .1>
  // 12 kk49     <.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .9, .1, .1, .1, .1, .1, .1, .1, .1>
  // 13 megak    <.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .9, .1, .1, .1, .1, .1, .1, .1>
  // 14 mitchell <.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .9, .1, .1, .1, .1, .1, .1>
  // 15 night    <.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .9, .1, .1, .1, .1, .1>
  // 16 phoebe   <.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .9, .1, .1, .1, .1>
  // 17 saavik   <.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .9, .1, .1, .1>
  // 18 steffi   <.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .9, .1, .1>
  // 19 sz24     <.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .9, .1>
  // 20 tammo    <.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .9>
  if (!strcmp(userid, "an2i")) 
  {   
      // 1
      net->target[1] = TARGET_HIGH;
      net->target[2] = TARGET_LOW;
      net->target[3] = TARGET_LOW;
      net->target[4] = TARGET_LOW;
      net->target[5] = TARGET_LOW;
      net->target[6] = TARGET_LOW;
      net->target[7] = TARGET_LOW;
      net->target[8] = TARGET_LOW;
      net->target[9] = TARGET_LOW;
      net->target[10] = TARGET_LOW;
      net->target[11] = TARGET_LOW;
      net->target[12] = TARGET_LOW;
      net->target[13] = TARGET_LOW;
      net->target[14] = TARGET_LOW;
      net->target[15] = TARGET_LOW;
      net->target[16] = TARGET_LOW;
      net->target[17] = TARGET_LOW;
      net->target[18] = TARGET_LOW;
      net->target[19] = TARGET_LOW;
      net->target[20] = TARGET_LOW;
  } 
  else if (!strcmp(userid, "at33")) 
  {
      // 2
      net->target[1] = TARGET_LOW;
      net->target[2] = TARGET_HIGH;
      net->target[3] = TARGET_LOW;
      net->target[4] = TARGET_LOW;
      net->target[5] = TARGET_LOW;
      net->target[6] = TARGET_LOW;
      net->target[7] = TARGET_LOW;
      net->target[8] = TARGET_LOW;
      net->target[9] = TARGET_LOW;
      net->target[10] = TARGET_LOW;
      net->target[11] = TARGET_LOW;
      net->target[12] = TARGET_LOW;
      net->target[13] = TARGET_LOW;
      net->target[14] = TARGET_LOW;
      net->target[15] = TARGET_LOW;
      net->target[16] = TARGET_LOW;
      net->target[17] = TARGET_LOW;
      net->target[18] = TARGET_LOW;
      net->target[19] = TARGET_LOW;
      net->target[20] = TARGET_LOW;
  } 
  else if (!strcmp(userid, "boland")) 
  {
      // 3
      net->target[1] = TARGET_LOW;
      net->target[2] = TARGET_LOW;
      net->target[3] = TARGET_HIGH;
      net->target[4] = TARGET_LOW;
      net->target[5] = TARGET_LOW;
      net->target[6] = TARGET_LOW;
      net->target[7] = TARGET_LOW;
      net->target[8] = TARGET_LOW;
      net->target[9] = TARGET_LOW;
      net->target[10] = TARGET_LOW;
      net->target[11] = TARGET_LOW;
      net->target[12] = TARGET_LOW;
      net->target[13] = TARGET_LOW;
      net->target[14] = TARGET_LOW;
      net->target[15] = TARGET_LOW;
      net->target[16] = TARGET_LOW;
      net->target[17] = TARGET_LOW;
      net->target[18] = TARGET_LOW;
      net->target[19] = TARGET_LOW;
      net->target[20] = TARGET_LOW;
  }
  else if (!strcmp(userid, "bpm")) 
  {
      // 4
      net->target[1] = TARGET_LOW;
      net->target[2] = TARGET_LOW;
      net->target[3] = TARGET_LOW;
      net->target[4] = TARGET_HIGH;
      net->target[5] = TARGET_LOW;
      net->target[6] = TARGET_LOW;
      net->target[7] = TARGET_LOW;
      net->target[8] = TARGET_LOW;
      net->target[9] = TARGET_LOW;
      net->target[10] = TARGET_LOW;
      net->target[11] = TARGET_LOW;
      net->target[12] = TARGET_LOW;
      net->target[13] = TARGET_LOW;
      net->target[14] = TARGET_LOW;
      net->target[15] = TARGET_LOW;
      net->target[16] = TARGET_LOW;
      net->target[17] = TARGET_LOW;
      net->target[18] = TARGET_LOW;
      net->target[19] = TARGET_LOW;
      net->target[20] = TARGET_LOW;
  }
  else if (!strcmp(userid, "ch4f")) 
  {
      // 5
      net->target[1] = TARGET_LOW;
      net->target[2] = TARGET_LOW;
      net->target[3] = TARGET_LOW;
      net->target[4] = TARGET_LOW;
      net->target[5] = TARGET_HIGH;
      net->target[6] = TARGET_LOW;
      net->target[7] = TARGET_LOW;
      net->target[8] = TARGET_LOW;
      net->target[9] = TARGET_LOW;
      net->target[10] = TARGET_LOW;
      net->target[11] = TARGET_LOW;
      net->target[12] = TARGET_LOW;
      net->target[13] = TARGET_LOW;
      net->target[14] = TARGET_LOW;
      net->target[15] = TARGET_LOW;
      net->target[16] = TARGET_LOW;
      net->target[17] = TARGET_LOW;
      net->target[18] = TARGET_LOW;
      net->target[19] = TARGET_LOW;
      net->target[20] = TARGET_LOW;
  }
  else if (!strcmp(userid, "cheyer")) 
  {
      // 6
      net->target[1] = TARGET_LOW;
      net->target[2] = TARGET_LOW;
      net->target[3] = TARGET_LOW;
      net->target[4] = TARGET_LOW;
      net->target[5] = TARGET_LOW;
      net->target[6] = TARGET_HIGH;
      net->target[7] = TARGET_LOW;
      net->target[8] = TARGET_LOW;
      net->target[9] = TARGET_LOW;
      net->target[10] = TARGET_LOW;
      net->target[11] = TARGET_LOW;
      net->target[12] = TARGET_LOW;
      net->target[13] = TARGET_LOW;
      net->target[14] = TARGET_LOW;
      net->target[15] = TARGET_LOW;
      net->target[16] = TARGET_LOW;
      net->target[17] = TARGET_LOW;
      net->target[18] = TARGET_LOW;
      net->target[19] = TARGET_LOW;
      net->target[20] = TARGET_LOW;
  }
  else if (!strcmp(userid, "choon")) 
  {
      // 7
      net->target[1] = TARGET_LOW;
      net->target[2] = TARGET_LOW;
      net->target[3] = TARGET_LOW;
      net->target[4] = TARGET_LOW;
      net->target[5] = TARGET_LOW;
      net->target[6] = TARGET_LOW;
      net->target[7] = TARGET_HIGH;
      net->target[8] = TARGET_LOW;
      net->target[9] = TARGET_LOW;
      net->target[10] = TARGET_LOW;
      net->target[11] = TARGET_LOW;
      net->target[12] = TARGET_LOW;
      net->target[13] = TARGET_LOW;
      net->target[14] = TARGET_LOW;
      net->target[15] = TARGET_LOW;
      net->target[16] = TARGET_LOW;
      net->target[17] = TARGET_LOW;
      net->target[18] = TARGET_LOW;
      net->target[19] = TARGET_LOW;
      net->target[20] = TARGET_LOW;
  }
  else if (!strcmp(userid, "danieln")) 
  {
      // 8
      net->target[1] = TARGET_LOW;
      net->target[2] = TARGET_LOW;
      net->target[3] = TARGET_LOW;
      net->target[4] = TARGET_LOW;
      net->target[5] = TARGET_LOW;
      net->target[6] = TARGET_LOW;
      net->target[7] = TARGET_LOW;
      net->target[8] = TARGET_HIGH;
      net->target[9] = TARGET_LOW;
      net->target[10] = TARGET_LOW;
      net->target[11] = TARGET_LOW;
      net->target[12] = TARGET_LOW;
      net->target[13] = TARGET_LOW;
      net->target[14] = TARGET_LOW;
      net->target[15] = TARGET_LOW;
      net->target[16] = TARGET_LOW;
      net->target[17] = TARGET_LOW;
      net->target[18] = TARGET_LOW;
      net->target[19] = TARGET_LOW;
      net->target[20] = TARGET_LOW;
  }
  else if (!strcmp(userid, "glickman")) 
  {
      // 9
      net->target[1] = TARGET_LOW;
      net->target[2] = TARGET_LOW;
      net->target[3] = TARGET_LOW;
      net->target[4] = TARGET_LOW;
      net->target[5] = TARGET_LOW;
      net->target[6] = TARGET_LOW;
      net->target[7] = TARGET_LOW;
      net->target[8] = TARGET_LOW;
      net->target[9] = TARGET_HIGH;
      net->target[10] = TARGET_LOW;
      net->target[11] = TARGET_LOW;
      net->target[12] = TARGET_LOW;
      net->target[13] = TARGET_LOW;
      net->target[14] = TARGET_LOW;
      net->target[15] = TARGET_LOW;
      net->target[16] = TARGET_LOW;
      net->target[17] = TARGET_LOW;
      net->target[18] = TARGET_LOW;
      net->target[19] = TARGET_LOW;
      net->target[20] = TARGET_LOW;
  }
  else if (!strcmp(userid, "karyadi")) 
  {
      // 10
      net->target[1] = TARGET_LOW;
      net->target[2] = TARGET_LOW;
      net->target[3] = TARGET_LOW;
      net->target[4] = TARGET_LOW;
      net->target[5] = TARGET_LOW;
      net->target[6] = TARGET_LOW;
      net->target[7] = TARGET_LOW;
      net->target[8] = TARGET_LOW;
      net->target[9] = TARGET_LOW;
      net->target[10] = TARGET_HIGH;
      net->target[11] = TARGET_LOW;
      net->target[12] = TARGET_LOW;
      net->target[13] = TARGET_LOW;
      net->target[14] = TARGET_LOW;
      net->target[15] = TARGET_LOW;
      net->target[16] = TARGET_LOW;
      net->target[17] = TARGET_LOW;
      net->target[18] = TARGET_LOW;
      net->target[19] = TARGET_LOW;
      net->target[20] = TARGET_LOW;
  }
  else if (!strcmp(userid, "kawamura")) 
  {
      // 11
      net->target[1] = TARGET_LOW;
      net->target[2] = TARGET_LOW;
      net->target[3] = TARGET_LOW;
      net->target[4] = TARGET_LOW;
      net->target[5] = TARGET_LOW;
      net->target[6] = TARGET_LOW;
      net->target[7] = TARGET_LOW;
      net->target[8] = TARGET_LOW;
      net->target[9] = TARGET_LOW;
      net->target[10] = TARGET_LOW;
      net->target[11] = TARGET_HIGH;
      net->target[12] = TARGET_LOW;
      net->target[13] = TARGET_LOW;
      net->target[14] = TARGET_LOW;
      net->target[15] = TARGET_LOW;
      net->target[16] = TARGET_LOW;
      net->target[17] = TARGET_LOW;
      net->target[18] = TARGET_LOW;
      net->target[19] = TARGET_LOW;
      net->target[20] = TARGET_LOW;
  }
  else if (!strcmp(userid, "kk49")) 
  {
      // 12
      net->target[1] = TARGET_LOW;
      net->target[2] = TARGET_LOW;
      net->target[3] = TARGET_LOW;
      net->target[4] = TARGET_LOW;
      net->target[5] = TARGET_LOW;
      net->target[6] = TARGET_LOW;
      net->target[7] = TARGET_LOW;
      net->target[8] = TARGET_LOW;
      net->target[9] = TARGET_LOW;
      net->target[10] = TARGET_LOW;
      net->target[11] = TARGET_LOW;
      net->target[12] = TARGET_HIGH;
      net->target[13] = TARGET_LOW;
      net->target[14] = TARGET_LOW;
      net->target[15] = TARGET_LOW;
      net->target[16] = TARGET_LOW;
      net->target[17] = TARGET_LOW;
      net->target[18] = TARGET_LOW;
      net->target[19] = TARGET_LOW;
      net->target[20] = TARGET_LOW;
  }
  else if (!strcmp(userid, "megak")) 
  {
      // 13
      net->target[1] = TARGET_LOW;
      net->target[2] = TARGET_LOW;
      net->target[3] = TARGET_LOW;
      net->target[4] = TARGET_LOW;
      net->target[5] = TARGET_LOW;
      net->target[6] = TARGET_LOW;
      net->target[7] = TARGET_LOW;
      net->target[8] = TARGET_LOW;
      net->target[9] = TARGET_LOW;
      net->target[10] = TARGET_LOW;
      net->target[11] = TARGET_LOW;
      net->target[12] = TARGET_LOW;
      net->target[13] = TARGET_HIGH;
      net->target[14] = TARGET_LOW;
      net->target[15] = TARGET_LOW;
      net->target[16] = TARGET_LOW;
      net->target[17] = TARGET_LOW;
      net->target[18] = TARGET_LOW;
      net->target[19] = TARGET_LOW;
      net->target[20] = TARGET_LOW;
  }
  else if (!strcmp(userid, "mitchell")) 
  {
      // 14
      net->target[1] = TARGET_LOW;
      net->target[2] = TARGET_LOW;
      net->target[3] = TARGET_LOW;
      net->target[4] = TARGET_LOW;
      net->target[5] = TARGET_LOW;
      net->target[6] = TARGET_LOW;
      net->target[7] = TARGET_LOW;
      net->target[8] = TARGET_LOW;
      net->target[9] = TARGET_LOW;
      net->target[10] = TARGET_LOW;
      net->target[11] = TARGET_LOW;
      net->target[12] = TARGET_LOW;
      net->target[13] = TARGET_LOW;
      net->target[14] = TARGET_HIGH;
      net->target[15] = TARGET_LOW;
      net->target[16] = TARGET_LOW;
      net->target[17] = TARGET_LOW;
      net->target[18] = TARGET_LOW;
      net->target[19] = TARGET_LOW;
      net->target[20] = TARGET_LOW;
  }
  else if (!strcmp(userid, "night")) 
  {
      // 15
      net->target[1] = TARGET_LOW;
      net->target[2] = TARGET_LOW;
      net->target[3] = TARGET_LOW;
      net->target[4] = TARGET_LOW;
      net->target[5] = TARGET_LOW;
      net->target[6] = TARGET_LOW;
      net->target[7] = TARGET_LOW;
      net->target[8] = TARGET_LOW;
      net->target[9] = TARGET_LOW;
      net->target[10] = TARGET_LOW;
      net->target[11] = TARGET_LOW;
      net->target[12] = TARGET_LOW;
      net->target[13] = TARGET_LOW;
      net->target[14] = TARGET_LOW;
      net->target[15] = TARGET_HIGH;
      net->target[16] = TARGET_LOW;
      net->target[17] = TARGET_LOW;
      net->target[18] = TARGET_LOW;
      net->target[19] = TARGET_LOW;
      net->target[20] = TARGET_LOW;
  }
  else if (!strcmp(userid, "phoebe")) 
  {
      // 16
      net->target[1] = TARGET_LOW;
      net->target[2] = TARGET_LOW;
      net->target[3] = TARGET_LOW;
      net->target[4] = TARGET_LOW;
      net->target[5] = TARGET_LOW;
      net->target[6] = TARGET_LOW;
      net->target[7] = TARGET_LOW;
      net->target[8] = TARGET_LOW;
      net->target[9] = TARGET_LOW;
      net->target[10] = TARGET_LOW;
      net->target[11] = TARGET_LOW;
      net->target[12] = TARGET_LOW;
      net->target[13] = TARGET_LOW;
      net->target[14] = TARGET_LOW;
      net->target[15] = TARGET_LOW;
      net->target[16] = TARGET_HIGH;
      net->target[17] = TARGET_LOW;
      net->target[18] = TARGET_LOW;
      net->target[19] = TARGET_LOW;
      net->target[20] = TARGET_LOW;
  }
  else if (!strcmp(userid, "saavik")) 
  {
      // 17
      net->target[1] = TARGET_LOW;
      net->target[2] = TARGET_LOW;
      net->target[3] = TARGET_LOW;
      net->target[4] = TARGET_LOW;
      net->target[5] = TARGET_LOW;
      net->target[6] = TARGET_LOW;
      net->target[7] = TARGET_LOW;
      net->target[8] = TARGET_LOW;
      net->target[9] = TARGET_LOW;
      net->target[10] = TARGET_LOW;
      net->target[11] = TARGET_LOW;
      net->target[12] = TARGET_LOW;
      net->target[13] = TARGET_LOW;
      net->target[14] = TARGET_LOW;
      net->target[15] = TARGET_LOW;
      net->target[16] = TARGET_LOW;
      net->target[17] = TARGET_HIGH;
      net->target[18] = TARGET_LOW;
      net->target[19] = TARGET_LOW;
      net->target[20] = TARGET_LOW;
  }
  else if (!strcmp(userid, "steffi")) 
  {
      // 18
      net->target[1] = TARGET_LOW;
      net->target[2] = TARGET_LOW;
      net->target[3] = TARGET_LOW;
      net->target[4] = TARGET_LOW;
      net->target[5] = TARGET_LOW;
      net->target[6] = TARGET_LOW;
      net->target[7] = TARGET_LOW;
      net->target[8] = TARGET_LOW;
      net->target[9] = TARGET_LOW;
      net->target[10] = TARGET_LOW;
      net->target[11] = TARGET_LOW;
      net->target[12] = TARGET_LOW;
      net->target[13] = TARGET_LOW;
      net->target[14] = TARGET_LOW;
      net->target[15] = TARGET_LOW;
      net->target[16] = TARGET_LOW;
      net->target[17] = TARGET_LOW;
      net->target[18] = TARGET_HIGH;
      net->target[19] = TARGET_LOW;
      net->target[20] = TARGET_LOW;
  }
  else if (!strcmp(userid, "sz24")) 
  {
      // 19
      net->target[1] = TARGET_LOW;
      net->target[2] = TARGET_LOW;
      net->target[3] = TARGET_LOW;
      net->target[4] = TARGET_LOW;
      net->target[5] = TARGET_LOW;
      net->target[6] = TARGET_LOW;
      net->target[7] = TARGET_LOW;
      net->target[8] = TARGET_LOW;
      net->target[9] = TARGET_LOW;
      net->target[10] = TARGET_LOW;
      net->target[11] = TARGET_LOW;
      net->target[12] = TARGET_LOW;
      net->target[13] = TARGET_LOW;
      net->target[14] = TARGET_LOW;
      net->target[15] = TARGET_LOW;
      net->target[16] = TARGET_LOW;
      net->target[17] = TARGET_LOW;
      net->target[18] = TARGET_LOW;
      net->target[19] = TARGET_HIGH;
      net->target[20] = TARGET_LOW;
  }
  else if (!strcmp(userid, "tammo")) 
  {
      // 20
      net->target[1] = TARGET_LOW;
      net->target[2] = TARGET_LOW;
      net->target[3] = TARGET_LOW;
      net->target[4] = TARGET_LOW;
      net->target[5] = TARGET_LOW;
      net->target[6] = TARGET_LOW;
      net->target[7] = TARGET_LOW;
      net->target[8] = TARGET_LOW;
      net->target[9] = TARGET_LOW;
      net->target[10] = TARGET_LOW;
      net->target[11] = TARGET_LOW;
      net->target[12] = TARGET_LOW;
      net->target[13] = TARGET_LOW;
      net->target[14] = TARGET_LOW;
      net->target[15] = TARGET_LOW;
      net->target[16] = TARGET_LOW;
      net->target[17] = TARGET_LOW;
      net->target[18] = TARGET_LOW;
      net->target[19] = TARGET_LOW;
      net->target[20] = TARGET_HIGH;
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
