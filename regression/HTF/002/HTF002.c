
/*****************************************************************************
 *
 * This program tests the generation of the struct in the header file.
 * It should contain all the fields (as opposed to "filler"s).
 *
 ****************************************************************************/

#include <stdlib.h>

typedef unsigned long long pb_Timestamp;

struct pb_Timer
{
    pb_Timestamp init;
    pb_Timestamp elapsed;
};

int main(int argc, char ** argv)
{
    struct pb_Timer timer;

    timer.init = atoi(argv[0]);
    timer.elapsed = atoi(argv[1]);

    return (timer.init + timer.elapsed);
}

