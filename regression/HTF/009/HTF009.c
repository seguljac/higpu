
/*****************************************************************************
 *
 * This program consists of multiple files, which share a header file that
 * defines multiple structs. The hiCUDA compiler should always generate
 * correct code regardless of the order of files put in the command line.
 *
 * Here, correctness means that the unamed struct types must be given unique
 * names.
 *
 ****************************************************************************/

#include "HTF009.h"

int main (int argc, char * argv[])
{
    s_node* nodes;
    s_disp* disps;
    s_node_disp* node_disps;
    s_element* elements;
    unsigned char *mask;

    func1(node_disps, elements, mask);
    func2(nodes, disps, elements, node_disps);

    return 0;
}

