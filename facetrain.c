/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *      Prepared for 15-681, Fall 1994.
 * 1-20人物识别器
 * Tue Oct  7 08:12:06 EDT 1997, bthom, added a few comments,
 *       tagged w/bthom
 *
 * Modified by Shengjia Yan @2016-03-29
 *
 ******************************************************************
 */

/*
 ██       ██████   ██████      ███████  █████   ██████ ███████
███            ██ ██  ████     ██      ██   ██ ██      ██
 ██ █████  █████  ██ ██ ██     █████   ███████ ██      █████
 ██       ██      ████  ██     ██      ██   ██ ██      ██
 ██       ███████  ██████      ██      ██   ██  ██████ ███████
*/

#include "backprop.h"
#include "pgmimage.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern char *strcpy();
extern void exit();

typedef int bool;
#define false 0
#define true 1

// 评估表现
int evaluate_performance(net, err) BPNN *net;
double *err;
{
  bool flag = true; // 样例匹配成功为true
  
  *err = 0.0;
  double delta;
  
  // 计算输出层均方误差之和
  for (int j = 1; j <= net->output_n; j++) 
  {
    delta = net->target[j] - net->output_units[j];
    *err += (0.5 * delta * delta);
  }
  
  
  for (int j = 1; j <= net->output_n; j++) {
    /*** If the target unit is on... ***/
    if (net->target[j] > 0.5) {
      if (net->output_units[j] > 0.5) {
        /*** If the output unit is on, then we correctly recognized me! ***/
      } else /*** otherwise, we didn't think it was me... ***/
      {
        flag = false;
      }
    } else /*** Else, the target unit is off... ***/
    {
      if (net->output_units[j] > 0.5) {
        /*** If the output unit is on, then we mistakenly thought it was me ***/
        flag = false;
      } else {
        /*** else, we correctly realized that it wasn't me ***/
      }
    }
  }

  if (flag)
    return 1;
  else
    return 0;
}

/*** Computes the performance of a net on the images in the imagelist. ***/
/*** Prints out the percentage correct on the image set, and the
     average error between the target and the output units for the set. ***/
int performance_on_imagelist(net, il, list_errors) BPNN *net;
IMAGELIST *il;
int list_errors;
{
  double err, val;
  int i, n, j, correct;

  err = 0.0;
  correct = 0;
  n = il->n;  // n：图片集中图片张数
  if (n > 0) {
    // 遍历图片列表中每张图片
    for (i = 0; i < n; i++) {

      /*** Load the image into the input layer. **/
      load_input_with_image(il->list[i], net);

      /*** Run the net on this input. **/
      bpnn_feedforward(net);

      /*** Set up the target vector for this image. **/
      load_target(il->list[i], net);

      /*** See if it got it right. ***/
      if (evaluate_performance(net, &val)) {
        //匹配成功，计数器加1
        correct++;
      } 
      else if (list_errors) 
      {
        printf("%s", NAME(il->list[i]));

        // for (j = 1; j <= net->output_n; j++) 
        // {
        //   printf("%.3f ", net->output_units[j]);
        // }
        printf("\n");
      }
      err += val; // 列表中所有图片 输出层 均方误差之和
    }

    err = err / (double)n;  // 列表中所有图片 输出层 均方误差之和 的平均数

    if (!list_errors)
      /* bthom==================================
         this line prints part of the ouput line
         discussed in section 3.1.2 of homework
          */
      // 输出 匹配准确率 和 误差
      printf("%g%%  %g \n", ((double)correct / (double)n) * 100.0, err);
  } else {
    if (!list_errors)
      printf("0.0 0.0 ");
  }
  return 0;
}

// 评估图片集的匹配情况
int result_on_imagelist(net, il, list_errors) BPNN *net;
IMAGELIST *il;
int list_errors;
{
  double err, val;
  int i, n, j, correct;

  err = 0.0;
  correct = 0;

  n = il->n; // 图片集元素个数

  if (n > 0) {
    for (i = 0; i < n; i++) {
      /*** Load the image into the input layer. **/
      // 装载图片到输入层
      load_input_with_image(il->list[i], net);

      /*** Run the net on this input. **/
      // 在此输入的基础上运行这个网络
      bpnn_feedforward(net);

      /*** Set up the target vector for this image. **/
      // 设置目标向量
      load_target(il->list[i], net);

      // 输出图片的名称
      printf("图片名称：%s\n", NAME(il->list[i]));

      // // 打印目标值
      // printf("目标值：");
      // for (j = 1; j <= net->output_n; j++) {
      //   printf("%.3f ", net->target[j]);
      // }

      // // 输出目标人物
      // if (net->target[1] > 0.5) {
      //   printf("an2i");
      // } else if (net->target[2] > 0.5) {
      //   printf("at33");
      // } else if (net->target[3] > 0.5) {
      //   printf("boland");
      // } else if (net->target[4] > 0.5) {
      //   printf("bpm");
      // } else if (net->target[5] > 0.5) {
      //   printf("ch4f");
      // } else if (net->target[6] > 0.5) {
      //   printf("cheyer");
      // } else if (net->target[7] > 0.5) {
      //   printf("choon");
      // } else if (net->target[8] > 0.5) {
      //   printf("danieln");
      // } else if (net->target[9] > 0.5) {
      //   printf("glickman");
      // } else if (net->target[10] > 0.5) {
      //   printf("karyadi");
      // } else if (net->target[11] > 0.5) {
      //   printf("kawamura");
      // } else if (net->target[12] > 0.5) {
      //   printf("kk49");
      // } else if (net->target[13] > 0.5) {
      //   printf("megak");
      // } else if (net->target[14] > 0.5) {
      //   printf("mitchell");
      // } else if (net->target[15] > 0.5) {
      //   printf("night");
      // } else if (net->target[16] > 0.5) {
      //   printf("phoebe");
      // } else if (net->target[17] > 0.5) {
      //   printf("saavik");
      // } else if (net->target[18] > 0.5) {
      //   printf("steffi");
      // } else if (net->target[19] > 0.5) {
      //   printf("sz24");
      // } else if (net->target[20] > 0.5) {
      //   printf("tammo");
      // }
      
      // printf("\n");

      // // 打印输出层单元输出值
      // printf("输出值：");
      // for (j = 1; j <= net->output_n; j++) {
      //   printf("%.3f ", net->output_units[j]);
      // }

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
      
      // 输出预测人物 
      if (net->output_units[1] > 0.5 && net->output_units[2] <= 0.5 &&
          net->output_units[3] <= 0.5 && net->output_units[4] <= 0.5 &&
          net->output_units[5] <= 0.5 && net->output_units[6] <= 0.5 &&
          net->output_units[7] <= 0.5 && net->output_units[8] <= 0.5 &&
          net->output_units[9] <= 0.5 && net->output_units[10] <= 0.5 &&
          net->output_units[11] <= 0.5 && net->output_units[12] <= 0.5 &&
          net->output_units[13] <= 0.5 && net->output_units[14] <= 0.5 &&
          net->output_units[15] <= 0.5 && net->output_units[16] <= 0.5 &&
          net->output_units[17] <= 0.5 && net->output_units[18] <= 0.5 &&
          net->output_units[19] <= 0.5 && net->output_units[20] <= 0.5) {
        printf("我猜他是 an2i");
      } else if (net->output_units[1] <= 0.5 && net->output_units[2] > 0.5 &&
          net->output_units[3] <= 0.5 && net->output_units[4] <= 0.5 &&
          net->output_units[5] <= 0.5 && net->output_units[6] <= 0.5 &&
          net->output_units[7] <= 0.5 && net->output_units[8] <= 0.5 &&
          net->output_units[9] <= 0.5 && net->output_units[10] <= 0.5 &&
          net->output_units[11] <= 0.5 && net->output_units[12] <= 0.5 &&
          net->output_units[13] <= 0.5 && net->output_units[14] <= 0.5 &&
          net->output_units[15] <= 0.5 && net->output_units[16] <= 0.5 &&
          net->output_units[17] <= 0.5 && net->output_units[18] <= 0.5 &&
          net->output_units[19] <= 0.5 && net->output_units[20] <= 0.5) {
        printf("我猜他是 at33");
      } else if(net->output_units[1] <= 0.5 && net->output_units[2] <= 0.5 &&
          net->output_units[3] > 0.5 && net->output_units[4] <= 0.5 &&
          net->output_units[5] <= 0.5 && net->output_units[6] <= 0.5 &&
          net->output_units[7] <= 0.5 && net->output_units[8] <= 0.5 &&
          net->output_units[9] <= 0.5 && net->output_units[10] <= 0.5 &&
          net->output_units[11] <= 0.5 && net->output_units[12] <= 0.5 &&
          net->output_units[13] <= 0.5 && net->output_units[14] <= 0.5 &&
          net->output_units[15] <= 0.5 && net->output_units[16] <= 0.5 &&
          net->output_units[17] <= 0.5 && net->output_units[18] <= 0.5 &&
          net->output_units[19] <= 0.5 && net->output_units[20] <= 0.5) {
        printf("我猜他是 boland");
      } else if (net->output_units[1] <= 0.5 && net->output_units[2] <= 0.5 &&
          net->output_units[3] <= 0.5 && net->output_units[4] > 0.5 &&
          net->output_units[5] <= 0.5 && net->output_units[6] <= 0.5 &&
          net->output_units[7] <= 0.5 && net->output_units[8] <= 0.5 &&
          net->output_units[9] <= 0.5 && net->output_units[10] <= 0.5 &&
          net->output_units[11] <= 0.5 && net->output_units[12] <= 0.5 &&
          net->output_units[13] <= 0.5 && net->output_units[14] <= 0.5 &&
          net->output_units[15] <= 0.5 && net->output_units[16] <= 0.5 &&
          net->output_units[17] <= 0.5 && net->output_units[18] <= 0.5 &&
          net->output_units[19] <= 0.5 && net->output_units[20] <= 0.5) {
        printf("我猜他是 bpm");
      } else if (net->output_units[1] <= 0.5 && net->output_units[2] <= 0.5 &&
          net->output_units[3] <= 0.5 && net->output_units[4] <= 0.5 &&
          net->output_units[5] > 0.5 && net->output_units[6] <= 0.5 &&
          net->output_units[7] <= 0.5 && net->output_units[8] <= 0.5 &&
          net->output_units[9] <= 0.5 && net->output_units[10] <= 0.5 &&
          net->output_units[11] <= 0.5 && net->output_units[12] <= 0.5 &&
          net->output_units[13] <= 0.5 && net->output_units[14] <= 0.5 &&
          net->output_units[15] <= 0.5 && net->output_units[16] <= 0.5 &&
          net->output_units[17] <= 0.5 && net->output_units[18] <= 0.5 &&
          net->output_units[19] <= 0.5 && net->output_units[20] <= 0.5) {
        printf("我猜他是 ch4f");
      } else if (net->output_units[1] <= 0.5 && net->output_units[2] <= 0.5 &&
          net->output_units[3] <= 0.5 && net->output_units[4] <= 0.5 &&
          net->output_units[5] <= 0.5 && net->output_units[6] > 0.5 &&
          net->output_units[7] <= 0.5 && net->output_units[8] <= 0.5 &&
          net->output_units[9] <= 0.5 && net->output_units[10] <= 0.5 &&
          net->output_units[11] <= 0.5 && net->output_units[12] <= 0.5 &&
          net->output_units[13] <= 0.5 && net->output_units[14] <= 0.5 &&
          net->output_units[15] <= 0.5 && net->output_units[16] <= 0.5 &&
          net->output_units[17] <= 0.5 && net->output_units[18] <= 0.5 &&
          net->output_units[19] <= 0.5 && net->output_units[20] <= 0.5) {
        printf("我猜他是 cheyer");
      } else if (net->output_units[1] <= 0.5 && net->output_units[2] <= 0.5 &&
          net->output_units[3] <= 0.5 && net->output_units[4] <= 0.5 &&
          net->output_units[5] <= 0.5 && net->output_units[6] <= 0.5 &&
          net->output_units[7] > 0.5 && net->output_units[8] <= 0.5 &&
          net->output_units[9] <= 0.5 && net->output_units[10] <= 0.5 &&
          net->output_units[11] <= 0.5 && net->output_units[12] <= 0.5 &&
          net->output_units[13] <= 0.5 && net->output_units[14] <= 0.5 &&
          net->output_units[15] <= 0.5 && net->output_units[16] <= 0.5 &&
          net->output_units[17] <= 0.5 && net->output_units[18] <= 0.5 &&
          net->output_units[19] <= 0.5 && net->output_units[20] <= 0.5) {
        printf("我猜他是 choon");
      } else if (net->output_units[1] <= 0.5 && net->output_units[2] <= 0.5 &&
          net->output_units[3] <= 0.5 && net->output_units[4] <= 0.5 &&
          net->output_units[5] <= 0.5 && net->output_units[6] <= 0.5 &&
          net->output_units[7] <= 0.5 && net->output_units[8] > 0.5 &&
          net->output_units[9] <= 0.5 && net->output_units[10] <= 0.5 &&
          net->output_units[11] <= 0.5 && net->output_units[12] <= 0.5 &&
          net->output_units[13] <= 0.5 && net->output_units[14] <= 0.5 &&
          net->output_units[15] <= 0.5 && net->output_units[16] <= 0.5 &&
          net->output_units[17] <= 0.5 && net->output_units[18] <= 0.5 &&
          net->output_units[19] <= 0.5 && net->output_units[20] <= 0.5) {
        printf("我猜他是 danieln");
      } else if (net->output_units[1] <= 0.5 && net->output_units[2] <= 0.5 &&
          net->output_units[3] <= 0.5 && net->output_units[4] <= 0.5 &&
          net->output_units[5] <= 0.5 && net->output_units[6] <= 0.5 &&
          net->output_units[7] <= 0.5 && net->output_units[8] <= 0.5 &&
          net->output_units[9] > 0.5 && net->output_units[10] <= 0.5 &&
          net->output_units[11] <= 0.5 && net->output_units[12] <= 0.5 &&
          net->output_units[13] <= 0.5 && net->output_units[14] <= 0.5 &&
          net->output_units[15] <= 0.5 && net->output_units[16] <= 0.5 &&
          net->output_units[17] <= 0.5 && net->output_units[18] <= 0.5 &&
          net->output_units[19] <= 0.5 && net->output_units[20] <= 0.5) {
        printf("我猜他是 glickman");
      } else if (net->output_units[1] <= 0.5 && net->output_units[2] <= 0.5 &&
          net->output_units[3] <= 0.5 && net->output_units[4] <= 0.5 &&
          net->output_units[5] <= 0.5 && net->output_units[6] <= 0.5 &&
          net->output_units[7] <= 0.5 && net->output_units[8] <= 0.5 &&
          net->output_units[9] <= 0.5 && net->output_units[10] > 0.5 &&
          net->output_units[11] <= 0.5 && net->output_units[12] <= 0.5 &&
          net->output_units[13] <= 0.5 && net->output_units[14] <= 0.5 &&
          net->output_units[15] <= 0.5 && net->output_units[16] <= 0.5 &&
          net->output_units[17] <= 0.5 && net->output_units[18] <= 0.5 &&
          net->output_units[19] <= 0.5 && net->output_units[20] <= 0.5) {
        printf("我猜他是 karyadi");
      } else if (net->output_units[1] <= 0.5 && net->output_units[2] <= 0.5 &&
          net->output_units[3] <= 0.5 && net->output_units[4] <= 0.5 &&
          net->output_units[5] <= 0.5 && net->output_units[6] <= 0.5 &&
          net->output_units[7] <= 0.5 && net->output_units[8] <= 0.5 &&
          net->output_units[9] <= 0.5 && net->output_units[10] <= 0.5 &&
          net->output_units[11] > 0.5 && net->output_units[12] <= 0.5 &&
          net->output_units[13] <= 0.5 && net->output_units[14] <= 0.5 &&
          net->output_units[15] <= 0.5 && net->output_units[16] <= 0.5 &&
          net->output_units[17] <= 0.5 && net->output_units[18] <= 0.5 &&
          net->output_units[19] <= 0.5 && net->output_units[20] <= 0.5) {
        printf("我猜他是 kawamura");
      } else if (net->output_units[1] <= 0.5 && net->output_units[2] <= 0.5 &&
          net->output_units[3] <= 0.5 && net->output_units[4] <= 0.5 &&
          net->output_units[5] <= 0.5 && net->output_units[6] <= 0.5 &&
          net->output_units[7] <= 0.5 && net->output_units[8] <= 0.5 &&
          net->output_units[9] <= 0.5 && net->output_units[10] <= 0.5 &&
          net->output_units[11] <= 0.5 && net->output_units[12] > 0.5 &&
          net->output_units[13] <= 0.5 && net->output_units[14] <= 0.5 &&
          net->output_units[15] <= 0.5 && net->output_units[16] <= 0.5 &&
          net->output_units[17] <= 0.5 && net->output_units[18] <= 0.5 &&
          net->output_units[19] <= 0.5 && net->output_units[20] <= 0.5) {
        printf("我猜他是 kk49");
      } else if (net->output_units[1] <= 0.5 && net->output_units[2] <= 0.5 &&
          net->output_units[3] <= 0.5 && net->output_units[4] <= 0.5 &&
          net->output_units[5] <= 0.5 && net->output_units[6] <= 0.5 &&
          net->output_units[7] <= 0.5 && net->output_units[8] <= 0.5 &&
          net->output_units[9] <= 0.5 && net->output_units[10] <= 0.5 &&
          net->output_units[11] <= 0.5 && net->output_units[12] <= 0.5 &&
          net->output_units[13] > 0.5 && net->output_units[14] <= 0.5 &&
          net->output_units[15] <= 0.5 && net->output_units[16] <= 0.5 &&
          net->output_units[17] <= 0.5 && net->output_units[18] <= 0.5 &&
          net->output_units[19] <= 0.5 && net->output_units[20] <= 0.5) {
        printf("我猜他是 megak");
      } else if (net->output_units[1] <= 0.5 && net->output_units[2] <= 0.5 &&
          net->output_units[3] <= 0.5 && net->output_units[4] <= 0.5 &&
          net->output_units[5] <= 0.5 && net->output_units[6] <= 0.5 &&
          net->output_units[7] <= 0.5 && net->output_units[8] <= 0.5 &&
          net->output_units[9] <= 0.5 && net->output_units[10] <= 0.5 &&
          net->output_units[11] <= 0.5 && net->output_units[12] <= 0.5 &&
          net->output_units[13] <= 0.5 && net->output_units[14] > 0.5 &&
          net->output_units[15] <= 0.5 && net->output_units[16] <= 0.5 &&
          net->output_units[17] <= 0.5 && net->output_units[18] <= 0.5 &&
          net->output_units[19] <= 0.5 && net->output_units[20] <= 0.5) {
        printf("我猜他是 mitchell");
      } else if (net->output_units[1] <= 0.5 && net->output_units[2] <= 0.5 &&
          net->output_units[3] <= 0.5 && net->output_units[4] <= 0.5 &&
          net->output_units[5] <= 0.5 && net->output_units[6] <= 0.5 &&
          net->output_units[7] <= 0.5 && net->output_units[8] <= 0.5 &&
          net->output_units[9] <= 0.5 && net->output_units[10] <= 0.5 &&
          net->output_units[11] <= 0.5 && net->output_units[12] <= 0.5 &&
          net->output_units[13] <= 0.5 && net->output_units[14] <= 0.5 &&
          net->output_units[15] > 0.5 && net->output_units[16] <= 0.5 &&
          net->output_units[17] <= 0.5 && net->output_units[18] <= 0.5 &&
          net->output_units[19] <= 0.5 && net->output_units[20] <= 0.5) {
        printf("我猜他是 night");
      } else if (net->output_units[1] <= 0.5 && net->output_units[2] <= 0.5 &&
          net->output_units[3] <= 0.5 && net->output_units[4] <= 0.5 &&
          net->output_units[5] <= 0.5 && net->output_units[6] <= 0.5 &&
          net->output_units[7] <= 0.5 && net->output_units[8] <= 0.5 &&
          net->output_units[9] <= 0.5 && net->output_units[10] <= 0.5 &&
          net->output_units[11] <= 0.5 && net->output_units[12] <= 0.5 &&
          net->output_units[13] <= 0.5 && net->output_units[14] <= 0.5 &&
          net->output_units[15] <= 0.5 && net->output_units[16] > 0.5 &&
          net->output_units[17] <= 0.5 && net->output_units[18] <= 0.5 &&
          net->output_units[19] <= 0.5 && net->output_units[20] <= 0.5) {
        printf("我猜他是 phoebe");
      } else if (net->output_units[1] <= 0.5 && net->output_units[2] <= 0.5 &&
          net->output_units[3] <= 0.5 && net->output_units[4] <= 0.5 &&
          net->output_units[5] <= 0.5 && net->output_units[6] <= 0.5 &&
          net->output_units[7] <= 0.5 && net->output_units[8] <= 0.5 &&
          net->output_units[9] <= 0.5 && net->output_units[10] <= 0.5 &&
          net->output_units[11] <= 0.5 && net->output_units[12] <= 0.5 &&
          net->output_units[13] <= 0.5 && net->output_units[14] <= 0.5 &&
          net->output_units[15] <= 0.5 && net->output_units[16] <= 0.5 &&
          net->output_units[17] > 0.5 && net->output_units[18] <= 0.5 &&
          net->output_units[19] <= 0.5 && net->output_units[20] <= 0.5) {
        printf("我猜他是 saavik");
      } else if (net->output_units[1] <= 0.5 && net->output_units[2] <= 0.5 &&
          net->output_units[3] <= 0.5 && net->output_units[4] <= 0.5 &&
          net->output_units[5] <= 0.5 && net->output_units[6] <= 0.5 &&
          net->output_units[7] <= 0.5 && net->output_units[8] <= 0.5 &&
          net->output_units[9] <= 0.5 && net->output_units[10] <= 0.5 &&
          net->output_units[11] <= 0.5 && net->output_units[12] <= 0.5 &&
          net->output_units[13] <= 0.5 && net->output_units[14] <= 0.5 &&
          net->output_units[15] <= 0.5 && net->output_units[16] <= 0.5 &&
          net->output_units[17] <= 0.5 && net->output_units[18] > 0.5 &&
          net->output_units[19] <= 0.5 && net->output_units[20] <= 0.5) {
        printf("我猜他是 steffi");
      } else if (net->output_units[1] <= 0.5 && net->output_units[2] <= 0.5 &&
          net->output_units[3] <= 0.5 && net->output_units[4] <= 0.5 &&
          net->output_units[5] <= 0.5 && net->output_units[6] <= 0.5 &&
          net->output_units[7] <= 0.5 && net->output_units[8] <= 0.5 &&
          net->output_units[9] <= 0.5 && net->output_units[10] <= 0.5 &&
          net->output_units[11] <= 0.5 && net->output_units[12] <= 0.5 &&
          net->output_units[13] <= 0.5 && net->output_units[14] <= 0.5 &&
          net->output_units[15] <= 0.5 && net->output_units[16] <= 0.5 &&
          net->output_units[17] <= 0.5 && net->output_units[18] <= 0.5 &&
          net->output_units[19] > 0.5 && net->output_units[20] <= 0.5) {
        printf("我猜他是 sz24");
      } else if (net->output_units[1] <= 0.5 && net->output_units[2] <= 0.5 &&
          net->output_units[3] <= 0.5 && net->output_units[4] <= 0.5 &&
          net->output_units[5] <= 0.5 && net->output_units[6] <= 0.5 &&
          net->output_units[7] <= 0.5 && net->output_units[8] <= 0.5 &&
          net->output_units[9] <= 0.5 && net->output_units[10] <= 0.5 &&
          net->output_units[11] <= 0.5 && net->output_units[12] <= 0.5 &&
          net->output_units[13] <= 0.5 && net->output_units[14] <= 0.5 &&
          net->output_units[15] <= 0.5 && net->output_units[16] <= 0.5 &&
          net->output_units[17] <= 0.5 && net->output_units[18] <= 0.5 &&
          net->output_units[19] <= 0.5 && net->output_units[20] > 0.5) {
        printf("我猜他是 tammo");
      } else {
        printf("我不知道他是谁");
      }

      printf(" ");

      /*** See if it got it right. ***/
      if (evaluate_performance(net, &val)) {
        correct++;
        printf("😎\n");
      } else {
        printf("😡\n");
      }

      printf("\n");

      err += val;
    }

    err = err / (double)n;

    // 输出 匹配准确率 和 平均误差
    if (!list_errors)
      printf("准确率：%g%%  平均误差：%g 👻\n\n",
             ((double)correct / (double)n) * 100.0, err);
  } else {
    if (!list_errors)
      printf("0.0 0.0 ");
  }
  return 0;
}

int backprop_face(trainlist, test1list, test2list, epochs, savedelta, netname,
                  list_errors) IMAGELIST *trainlist,
    *test1list, *test2list;
int epochs, savedelta, list_errors;
char *netname;
{
  IMAGE *iimg;
  BPNN *net;
  int train_n, epoch, i, imgsize;
  double out_err, hid_err, sumerr;

  train_n = trainlist->n;

  /*** Read network in if it exists, otherwise make one from scratch ***/
  if ((net = bpnn_read(netname)) == NULL) {
    if (train_n > 0) {
      printf("Creating new network '%s'\n", netname);
      iimg = trainlist->list[0];
      imgsize = ROWS(iimg) * COLS(iimg);
      /* bthom ===========================
        make a net with:
          imgsize inputs, 20 hiden units, and 20 output unit
    图片规模的输入层单元个数，20个隐藏层单元，20个输出层单元
          */
      net = bpnn_create(imgsize, 20, 20);
    } else {
      printf("Need some images to train on, use -t\n");
      return -1;
    }
  }

  if (epochs > 0) {
    /*** 训练进行中（epochs次） ***/
    printf("Training underway (going to %d epochs)\n", epochs);
    /*** 每epochs次保存网络 ***/
    printf("Will save network every %d epochs\n", savedelta);
    fflush(stdout);
  }

  /*** 迭代前输出测试表现 ***/
  /*** Print out performance before any epochs have been completed. ***/
  printf("\n迭代前：\n");
  printf("训练集误差和：0.0\n");
  printf("评估训练集的表现： ");
  performance_on_imagelist(net, trainlist, 0);
  printf("评估测试集1的表现：");
  performance_on_imagelist(net, test1list, 0);
  printf("评估测试集2的表现：");
  performance_on_imagelist(net, test2list, 0);
  printf("\n");
  fflush(stdout);
  if (list_errors) {
    printf(
        "\n训练集中的这些图片分类失败:\n");
    performance_on_imagelist(net, trainlist, 1);
    printf("\n测试集1中的这些图片分类失败:\n");
    performance_on_imagelist(net, test1list, 1);
    printf("\n测试集2中的这些图片分类失败:\n");
    performance_on_imagelist(net, test2list, 1);
  }

  /************** 开始训练！ ****************************/
  /************** Train it *****************************/
  for (epoch = 1; epoch <= epochs; epoch++) {

    // 输出迭代次数
    printf("迭代次数： %d \n", epoch);
    fflush(stdout);

    sumerr = 0.0;
    for (i = 0; i < train_n; i++) {

      /** Set up input units on net with image i **/
      // 用训练集中图片i来设置输入层单元
      load_input_with_image(trainlist->list[i], net);

      /** Set up target vector for image i **/
      // 为图片i设置目标向量
      load_target(trainlist->list[i], net);

      /** Run backprop, learning rate 0.3, momentum 0.3 **/
      /** 运行反向传播算法，学习速率0.3，冲量0.3 **/
      bpnn_train(net, 0.3, 0.3, &out_err, &hid_err);

      sumerr += (out_err + hid_err);  // 训练集中所有图片作为输入，网络的 输出层 和 隐藏层 的误差之和
    }
    printf("训练集误差和: %g \n", sumerr);

    // 评估测试集，测试集1，测试集2 的表现
    /*** Evaluate performance on train, test, test2, and print perf ***/
    printf("评估训练集的表现： ");
    performance_on_imagelist(net, trainlist, 0);
    printf("评估测试集1的表现：");
    performance_on_imagelist(net, test1list, 0);
    printf("评估测试集2的表现：");
    performance_on_imagelist(net, test2list, 0);
    printf("\n");
    fflush(stdout);

    /*** Save network every 'savedelta' epochs ***/
    if (!(epoch % savedelta)) {
      bpnn_save(net, netname);
    }
  }
  printf("\n");
  fflush(stdout);
  /************** 迭代结束 ****************************/

  /************** 预测结果 ****************************/

  // 输出测试集中每张图片的匹配情况
  printf("迭代结束后的匹配情况：\n\n");
  printf("测试集1：\n\n");
  result_on_imagelist(net, test1list, 0);
  printf("测试集2：\n\n");
  result_on_imagelist(net, test2list, 0);

  /** Save the trained network **/
  if (epochs > 0) {
    bpnn_save(net, netname);
  }
  return 0;
}

void printusage(prog) char *prog;
{
  printf("USAGE: %s\n", prog);
  printf("       -n <network file>\n");
  printf("       [-e <number of epochs>]\n");
  printf("       [-s <random number generator seed>]\n");
  printf("       [-S <number of epochs between saves of network>]\n");
  printf("       [-t <training set list>]\n");
  printf("       [-1 <testing set 1 list>]\n");
  printf("       [-2 <testing set 2 list>]\n");
  printf("       [-T]\n");
}

int main(argc, argv) int argc;
char *argv[];
{
  char netname[256], trainname[256], test1name[256], test2name[256];
  IMAGELIST *trainlist, *test1list, *test2list;
  int ind, epochs, seed, savedelta, list_errors;

  // 种子
  seed = 102194; /*** today's date seemed like a good default ***/

  epochs = 100;
  savedelta = 100;
  list_errors = 0;
  netname[0] = trainname[0] = test1name[0] = test2name[0] = '\0';

  if (argc < 2) {
    printusage(argv[0]);
    exit(-1);
  }

  /*** Create imagelists ***/
  trainlist = imgl_alloc();
  test1list = imgl_alloc();
  test2list = imgl_alloc();

  /*** 扫描命令行 ***/
  /*** Scan command line ***/
  for (ind = 1; ind < argc; ind++) {

    /*** 分析选择了那种操作 ***/
    /*** Parse switches ***/
    if (argv[ind][0] == '-') {
      switch (argv[ind][1]) {
      case 'n':
        strcpy(netname, argv[++ind]);
        break;
      case 'e':
        epochs = atoi(argv[++ind]);
        break;
      case 's':
        seed = atoi(argv[++ind]);
        break;
      case 'S':
        savedelta = atoi(argv[++ind]);
        break;
      case 't':
        strcpy(trainname, argv[++ind]);
        break;
      case '1':
        strcpy(test1name, argv[++ind]);
        break;
      case '2':
        strcpy(test2name, argv[++ind]);
        break;
      case 'T':
        list_errors = 1;
        epochs = 0;
        break;
      default:
        printf("Unknown switch '%c'\n", argv[ind][1]);
        break;
      }
    }
  }

  /*** If any train, test1, or test2 sets have been specified, then
       load them in. ***/
  if (trainname[0] != '\0')
    imgl_load_images_from_textfile(trainlist, trainname);
  if (test1name[0] != '\0')
    imgl_load_images_from_textfile(test1list, test1name);
  if (test2name[0] != '\0')
    imgl_load_images_from_textfile(test2list, test2name);

  /*** If we haven't specified a network save file, we should... ***/
  if (netname[0] == '\0') {
    printf("%s: Must specify an output file, i.e., -n <network file>\n",
           argv[0]);
    exit(-1);
  }

  /*** Don't try to train if there's no training data ***/
  if (trainname[0] == '\0') {
    epochs = 0;
  }

  /*** 初始化神经网络包 ***/
  /*** Initialize the neural net package ***/
  bpnn_initialize(seed);

  /*** 显示训练集，测试集1，测试集2中图片数量 ***/
  /*** Show number of images in train, test1, test2 ***/
  printf("%d images in training set\n", trainlist->n);
  printf("%d images in test1 set\n", test1list->n);
  printf("%d images in test2 set\n", test2list->n);

  /*** If we've got at least one image to train on, go train the net ***/
  // 假如我们至少有1张图片来训练，那么就开始训练吧！
  backprop_face(trainlist, test1list, test2list, epochs, savedelta, netname,
                list_errors);

  exit(0);
}
