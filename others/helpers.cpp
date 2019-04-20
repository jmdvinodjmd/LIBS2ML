/**
 * helpers.cpp
 * Usage: This file contains some auxiliary definitions, classes and functions.
 * This file is part of LIBS2ML.
 * Created by V. K. Chauhan on Sept. 15, 2018
 * Last Modified by V. K. Chauhan on Apr. 15, 2019
 **/

#include<stdio.h>
#include<math.h>
#include "helpers.h"

 double nrm2_sq(const feature_node *x) {
    double ret = 0;
    while(x->index != -1)
    {
        ret += x->value*x->value;
        x++;
    }
    return (ret);
}

 double dot(const double *s, const feature_node *x) {
    double ret = 0;
    while(x->index != -1)
    {
        ret += s[x->index-1]*x->value;
        x++;
    }
    return (ret);
}

 void axpy(const double a, const feature_node *x, double *y) {
    while(x->index != -1)
    {
        y[x->index-1] += a*x->value;
        x++;
    }
}

double dot_(size_t p, double *x, double *y) {
    double temp = 0;
    for(size_t i=0; i<p; ++i) {
        temp += x[i]*y[i];
    }
    return temp;
}
double norm_(size_t p, double *y) {
    return sqrt(dot_(p, y, y));
}
void axpy_(size_t p, const double a, double *x, double *y) {
    for(size_t i=0; i<p; ++i) {
        y[i] = a*x[i] + y[i];
    }
}