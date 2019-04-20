/**
 * method_base.cpp
 * Usage: This file defines base method class for different solvers.
 *
 * This file is part of LIBS2ML.
 * Created by V. K. Chauhan on Sept. 15, 2018
 * Last Modified by V. K. Chauhan on Apr. 15, 2019
 **/

#ifndef METHOD_BASE_H
#define METHOD_BASE_H

#include <stdio.h>
#include <math.h>
// #include "helpers.h"


// this is base class for all the methods. Each new method should extend this class.
class method {
public:
    virtual ~method() {}
    virtual void solve(double *w, outputs *results) = 0;
};

#endif