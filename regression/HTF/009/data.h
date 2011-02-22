#ifndef _DATA_H
#define _DATA_H

typedef	struct {
	int id;
	double x;
	double y;
	double z;
} s_node;

typedef	struct {
	int id;
	double dx;
	double dy;
	double dz;
} s_disp;

typedef	struct {
	double x;
	double y;
	double z;	
	double dx;
	double dy;
	double dz;
} s_node_disp;

typedef struct {
	int nodes[4];
} s_element;

#endif
