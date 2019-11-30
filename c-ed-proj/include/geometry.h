#ifndef GEOMETRY_INCLUDED
#define GEOMETRY_INCLUDED

#include <stdbool.h>

int create_geometry(int xsites, int ysites, int periodic_x, int periodic_y, int number_left2right_up2down, double * xlocs, double * ylocs, bool * connect_mat);

int center_locs(double * xlocs, double * ylocs, int nsites);

#endif
