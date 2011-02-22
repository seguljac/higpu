
/*****************************************************************************
 *
 * Stress test for the code generator
 *
 ****************************************************************************/

#include <stdlib.h>
#include <math.h>
#include <stdio.h>


/*
 * (c) 2007 The Board of Trustees of the University of Illinois.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <unistd.h>

/* Command line parameters for benchmarks */
struct pb_Parameters {
 char *outFile;                /* If not NULL, the raw output of the
                                * computation should be saved to this
                                * file. The string is owned. */
 char **inpFiles;              /* A NULL-terminated array of strings
                                * holding the input file(s) for the
                                * computation.  The array and strings
                                * are owned. */
 int synchronizeGpu;           /* Controls behavior of CUDA benchmarks.
                                * If nonzero, a CUDA runtime
                                * synchronization call should happen
                                * after each data transfer to the GPU
                                * and after each kernel call.  This
                                * is necessary for accurate timing
                                * measurement. */
};

/* Read command-line parameters.
 *
 * The argc and argv parameters to main are read, and any parameters
 * interpreted by this function are removed from the argument list.
 *
 * A new instance of struct pb_Parameters is returned.
 * If there is an error, then an error message is printed on stderr
 * and NULL is returned.
 */
struct pb_Parameters *
pb_ReadParameters(int *_argc, char **argv);

/* Free an instance of struct pb_Parameters.
 */
void
pb_FreeParameters(struct pb_Parameters *p);

/* Count the number of input files in a pb_Parameters instance.
 */
int
pb_Parameters_CountInputs(struct pb_Parameters *p);

/* A time or duration. */
#if _POSIX_VERSION >= 200112L
typedef unsigned long long pb_Timestamp; /* time in microseconds */
#else
# error "Timestamps not implemented"
#endif

enum pb_TimerState {
 pb_Timer_STOPPED,
 pb_Timer_RUNNING,
};

struct pb_Timer {
 enum pb_TimerState state;
 pb_Timestamp elapsed;         /* Amount of time elapsed so far */
 pb_Timestamp init;            /* Beginning of the current time interval,
                                * if state is RUNNING.  Undefined
                                * otherwise.  */
};

/* Reset a timer.
 * Use this to initialize a timer or to clear
 * its elapsed time.  The reset timer is stopped.
 */
void
pb_ResetTimer(struct pb_Timer *timer);

/* Start a timer.  The timer is set to RUNNING mode and
 * time elapsed while the timer is running is added to
 * the timer.
 * The timer should not already be running.
 */
void
pb_StartTimer(struct pb_Timer *timer);

/* Stop a timer.
 * This stops adding elapsed time to the timer.
 * The timer should not already be stopped.
 */
void
pb_StopTimer(struct pb_Timer *timer);

/* Get the elapsed time in seconds. */
double
pb_GetElapsedTime(struct pb_Timer *timer);

/* Execution time is assigned to one of these categories. */
enum pb_TimerID {
 pb_TimerID_NONE = 0,
 pb_TimerID_IO,                /* Time spent in input/output */
 pb_TimerID_GPU,               /* Time spent computing on the GPU */
 pb_TimerID_COPY,              /* Time spent moving data to/from GPU and
                                * allocating/freeing memory on the GPU */
 pb_TimerID_COMPUTE,           /* Time for all program execution other
                                * than parsing command line arguments,
                                * I/O, GPU, and copy */
 pb_TimerID_LAST               /* Number of timer IDs */
};

/* A set of timers for recording execution times. */
struct pb_TimerSet {
 enum pb_TimerID current;
 struct pb_Timer timers[pb_TimerID_LAST];
};

/* Reset all timers in the set. */
void
pb_InitializeTimerSet(struct pb_TimerSet *timers);

/* Select which timer the next interval of time should be accounted
 * to. The selected timer is started and other timers are stopped.
 * Using pb_TimerID_NONE stops all timers. */
void
pb_SwitchToTimer(struct pb_TimerSet *timers, enum pb_TimerID timer);

/* Print timer values to standard output. */
void
pb_PrintTimerSet(struct pb_TimerSet *timers);

#ifdef __cplusplus
}
#endif

#define VOLSIZEX 512
#define VOLSIZEY 512
#define ATOMCOUNT 40000
#define MAX_ATOMS 1024

/* Size of a thread block */
#define BLOCKSIZEX 16
#define BLOCKSIZEY 8

/* Number of grid points processed by a thread */
#define UNROLLX 1

/* Number of atoms processed by a kernel */
#define MAXATOMS 4000

typedef struct {
 int x;
 int y;
 int z;
} voldim3i;



/* The main compute kernel. */
void cpuenergy(voldim3i grid,
              int numatoms,
              float gridspacing,
              int k,
              const float *atoms,
              float *energygrid);



void cpuenergy(voldim3i grid,
              int numatoms,
              float gridspacing,
              int k,
              const float *atoms,
              float *energygrid)
{
 float energy;                   /* Energy of current grid point */
 float x,y,z;                    /* Coordinates of current grid point */
 int i,j,n;                      /* Loop counters */
 int atomarrdim = numatoms * 4;
 float atominfo[MAX_ATOMS];
 //int UNROLLX = 1;
 //dim3 Gsz, Bsz;

 // printf("\tWorking on plane %i of %ld\n", k, grid.z);
 z = gridspacing * (float) k;


#pragma hicuda constant copyin atominfo[*]

 int energygrid_sz = grid.x * grid.y * grid.z;

#pragma hicuda shape energygrid[energygrid_sz]
#pragma hicuda shape atoms[MAX_ATOMS]

#pragma hicuda global alloc numatoms copyin
#pragma hicuda global alloc gridspacing copyin
#pragma hicuda global alloc energygrid[*] copyin
#pragma hicuda global alloc atoms[*] copyin


 //int Gsz_x = grid.x / (BLOCKSIZEX * UNROLLX);
 //int Gsz_y = grid.y / (BLOCKSIZEY * UNROLLX);
 //int Gsz_z = grid.z / (1* UNROLLX);
int nn;
 for (nn=0; nn < atomarrdim; nn += 4*MAX_ATOMS)
     {
#pragma hicuda kernel addVector tblock(64, 32) thread(8, 16)
//#pragma hicuda kernel addVector tblock(Gsz_z, Gsz_y, Gsz_x) thread(16)

#pragma hicuda loop_partition over_tblock over_thread
 /* For each x, y grid point in this plane */
 for (j=0; j<grid.y; j++) {
   y = gridspacing * (float) j;

#pragma hicuda loop_partition over_tblock over_thread
   for (i=0; i<grid.x; i++) {
     x = gridspacing * (float) i;
     energy = 0.0f;

/* help the vectorizer make the right decision */
#if defined(__INTEL_COMPILER)
#pragma vector always
#endif

//#pragma hicuda loop_partition over_tblock over_thread
     /* Calculate the interaction with each atom */
     for (n=nn; n< nn+4*MAX_ATOMS; n+=4) {
       float dx = x - atoms[n  ];
       float dy = y - atoms[n+1];
       float dz = z - atoms[n+2];
       energy += atoms[n+3] / sqrtf(dx*dx + dy*dy + dz*dz);


     }

     energygrid[grid.x*grid.y*k + grid.x*j + i] = energy;
   }
 }
#pragma hicuda kernel_end
}
#pragma hicuda global copyout energygrid[*]

}


/* initatoms()
 * Store a pseudorandom arrangement of point charges in *atombuf.
 */
static int
initatoms(float **atombuf, int count, voldim3i volsize, float gridspacing) {
 voldim3i size;
 int i;
 float *atoms;

 srand(54321);                 // Ensure that atom placement is repeatable

 atoms = (float *) malloc(count * 4 * sizeof(float));
 *atombuf = atoms;

 // compute grid dimensions in angstroms
 size.x = gridspacing * volsize.x;
 size.y = gridspacing * volsize.y;
 size.z = gridspacing * volsize.z;

 for (i=0; i<count; i++) {
   int addr = i * 4;
   atoms[addr    ] = (rand() / (float) RAND_MAX) * size.x;
   atoms[addr + 1] = (rand() / (float) RAND_MAX) * size.y;
   atoms[addr + 2] = (rand() / (float) RAND_MAX) * size.z;
   atoms[addr + 3] = ((rand() / (float) RAND_MAX) * 2.0) - 1.0;  // charge
 }

 return 0;
}

/* writeenergy()
 * Write part of the energy array to an output file for verification.
 */
static int
writeenergy(char *filename, float *energy, voldim3i volsize)
{
 FILE *outfile;
 int x, y;

 outfile = fopen(filename, "w");
 if (outfile == NULL) {
   fputs("Cannot open output file\n", stderr);
   return -1;
   }

 /* Print the execution parameters */
 fprintf(outfile, "%d %d %d %d\n", volsize.x, volsize.y, volsize.z, ATOMCOUNT);

 /* Print a checksum */
 {
   double sum = 0.0;
    for (y = 0; y < volsize.y; y++) {
     for (x = 0; x < volsize.x; x++) {
       double t = energy[y*volsize.x+x];
       t = fmax(-20.0, fmin(20.0, t));
       sum += t;
     }
   }
   fprintf(outfile, "%.4g\n", sum);
 }

 /* Print several rows of the computed data */
 for (y = 0; y < 17; y++) {
   for (x = 0; x < volsize.x; x++) {
     int addr = y * volsize.x + x;
     fprintf(outfile, "%.4g ", energy[addr]);
   }
   fprintf(outfile, "\n");
 }

 fclose(outfile);

 return 0;
}

int main(int argc, char** argv) {
 struct pb_TimerSet timers;
 struct pb_Parameters *parameters;

 float *energy = NULL;         // Output of calculation
 float *atoms = NULL;
 voldim3i volsize;

 // number of atoms to simulate
 int atomcount = ATOMCOUNT;

 // voxel spacing
 const float gridspacing = 0.1;

 // Size of buffer on GPU
 int volmemsz;

 printf("CUDA accelerated coulombic potential microbenchmark\n");
 printf("Original version by John E. Stone <johns@ks.uiuc.edu>\n");
 printf("This version maintained by Chris Rodrigues\n");

 parameters = pb_ReadParameters(&argc, argv);
 if (!parameters)
   return -1;

 if (parameters->inpFiles[0]) {
   fputs("No input files expected\n", stderr);
   return -1;
 }

 pb_InitializeTimerSet(&timers);
 pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

 // setup energy grid size
 volsize.x = VOLSIZEX;
 volsize.y = VOLSIZEY;
 volsize.z = 1;

 // allocate and initialize atom coordinates and charges
 if (initatoms(&atoms, atomcount, volsize, gridspacing))
   return -1;

 // allocate the output array
 volmemsz = sizeof(float) * volsize.x * volsize.y * volsize.z;
 energy = (float *) malloc(volmemsz);

 // Main computation
 cpuenergy(volsize, atomcount, gridspacing, 0, atoms, energy);

#if 0
   printf("Done\n");
#endif

 pb_SwitchToTimer(&timers, pb_TimerID_IO);

 /* Print a subset of the results to a file */
 if (parameters->outFile) {
   if (writeenergy(parameters->outFile, energy, volsize) == -1)
     return -1;
 }

 pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

 free(atoms);
 free(energy);

 pb_SwitchToTimer(&timers, pb_TimerID_NONE);

 pb_PrintTimerSet(&timers);
 pb_FreeParameters(parameters);

 return 0;
}


/*
 * (c) 2007 The Board of Trustees of the University of Illinois.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if _POSIX_VERSION >= 200112L
#include <sys/time.h>
#endif

/* Free an array of owned strings. */
static void
free_string_array(char **string_array)
{
 char **p;

 if (!string_array) return;
 for (p = string_array; *p; p++) free(*p);
 free(string_array);
}

/* Parse a comma-delimited list of strings into an
 * array of strings. */
static char **
read_string_array(char *in)
{
 char **ret;
 int i;
 int count;                    /* Number of items in the input */
 char *substring;              /* Current substring within 'in' */

 /* Count the number of items in the string */
 count = 1;
 for (i = 0; in[i]; i++) if (in[i] == ',') count++;

 /* Allocate storage */
 ret = malloc((count + 1) * sizeof(char *));

 /* Create copies of the strings from the list */
 substring = in;
 for (i = 0; i < count; i++) {
   char *substring_end;
   int substring_length;

   /* Find length of substring */
   for (substring_end = substring;
        (*substring_end != ',') && (*substring_end != 0);
        substring_end++);

   substring_length = substring_end - substring;

   /* Allocate memory and copy the substring */
   ret[i] = malloc(substring_length + 1);
   memcpy(ret[i], substring, substring_length);
   ret[i][substring_length] = 0;

   /* go to next substring */
   substring = substring_end + 1;
 }
 ret[i] = NULL;                /* Write the sentinel value */

 return ret;
}

struct argparse {
 int argc;                     /* Number of arguments.  Mutable. */
 char **argv;                  /* Argument values.  Immutable. */

 int argn;                     /* Current argument number. */
 char **argv_get;              /* Argument value being read. */
 char **argv_put;              /* Argument value being written.
                                * argv_put <= argv_get. */
};

static void
initialize_argparse(struct argparse *ap, int argc, char **argv)
{
 ap->argc = argc;
 ap->argn = 0;
 ap->argv_get = ap->argv_put = ap->argv = argv;
}

static void
finalize_argparse(struct argparse *ap)
{
 /* Move the remaining arguments */
 for(; ap->argn < ap->argc; ap->argn++)
   *ap->argv_put++ = *ap->argv_get++;
}

/* Delete the current argument. */
static void
delete_argument(struct argparse *ap)
{
 if (ap->argn >= ap->argc) {
   fprintf(stderr, "delete_argument\n");
 }
 ap->argc--;
 ap->argv_get++;
}

/* Go to the next argument.  Also, move the current argument to its
 * final location in argv. */
static void
next_argument(struct argparse *ap)
{
 if (ap->argn >= ap->argc) {
   fprintf(stderr, "next_argument\n");
 }
 /* Move argument to its new location. */
 *ap->argv_put++ = *ap->argv_get++;
 ap->argn++;
}

static int
is_end_of_arguments(struct argparse *ap)
{
 return ap->argn == ap->argc;
}

static char *
get_argument(struct argparse *ap)
{
 return *ap->argv_get;
}

static char *
consume_argument(struct argparse *ap)
{
 char *ret = get_argument(ap);
 delete_argument(ap);
 return ret;
}

struct pb_Parameters *
pb_ReadParameters(int *_argc, char **argv)
{
 char *err_message;
 struct argparse ap;
 struct pb_Parameters *ret = malloc(sizeof(struct pb_Parameters));

 /* Initialize the parameters structure */
 ret->outFile = NULL;
 ret->inpFiles = malloc(sizeof(char *));
 ret->inpFiles[0] = NULL;
 ret->synchronizeGpu = 0;

 /* Each argument */
 initialize_argparse(&ap, *_argc, argv);
 while(!is_end_of_arguments(&ap)) {
   char *arg = get_argument(&ap);

   /* Single-character flag */
   if ((arg[0] == '-') && (arg[1] != 0) && (arg[2] == 0)) {
     delete_argument(&ap);     /* This argument is consumed here */

     switch(arg[1]) {
     case 'o':                 /* Output file name */
       if (is_end_of_arguments(&ap))
         {
           err_message = "Expecting file name after '-o'\n";
           goto error;
         }
       free(ret->outFile);
       ret->outFile = strdup(consume_argument(&ap));
       break;
     case 'i':                 /* Input file name */
       if (is_end_of_arguments(&ap))
         {
           err_message = "Expecting file name after '-i'\n";
           goto error;
         }
       ret->inpFiles = read_string_array(consume_argument(&ap));
       break;
     case 'S':                 /* Synchronize */
       ret->synchronizeGpu = 1;
       break;
     case '-':                 /* End of options */
       goto end_of_options;
     default:
       err_message = "Unexpected command-line parameter\n";
       goto error;
     }
   }
   else {
     /* Other parameters are ignored */
     next_argument(&ap);
   }
 } /* end for each argument */

 end_of_options:
 *_argc = ap.argc;             /* Save the modified argc value */
 finalize_argparse(&ap);

 return ret;

 error:
 fputs(err_message, stderr);
 pb_FreeParameters(ret);
 return NULL;
}

void
pb_FreeParameters(struct pb_Parameters *p)
{
 char **cpp;

 free(p->outFile);
 free_string_array(p->inpFiles);
 free(p);
}

int
pb_Parameters_CountInputs(struct pb_Parameters *p)
{
 int n;

 for (n = 0; p->inpFiles[n]; n++);
 return n;
}

/*****************************************************************************/
/* Timer routines */

static void
accumulate_time(pb_Timestamp *accum,
               pb_Timestamp start,
               pb_Timestamp end)
{
#if _POSIX_VERSION >= 200112L
 *accum += end - start;
#else
# error "Timestamps not implemented for this system"
#endif
}

void
pb_ResetTimer(struct pb_Timer *timer)
{
 timer->state = pb_Timer_STOPPED;

#if _POSIX_VERSION >= 200112L
 timer->elapsed = 0;
#else
# error "pb_ResetTimer: not implemented for this system"
#endif
}

void
pb_StartTimer(struct pb_Timer *timer)
{
 if (timer->state != pb_Timer_STOPPED) {
   fputs("Ignoring attempt to start a running timer\n", stderr);
   return;
 }

 timer->state = pb_Timer_RUNNING;

#if _POSIX_VERSION >= 200112L
 {
   struct timeval tv;
   gettimeofday(&tv, NULL);
   timer->init = tv.tv_sec * 1000000LL + tv.tv_usec;
 }
#else
# error "pb_StartTimer: not implemented for this system"
#endif
}

void
pb_StopTimer(struct pb_Timer *timer)
{
 pb_Timestamp fini;

 if (timer->state != pb_Timer_RUNNING) {
   fputs("Ignoring attempt to stop a stopped timer\n", stderr);
   return;
 }

 timer->state = pb_Timer_STOPPED;

#if _POSIX_VERSION >= 200112L
 {
   struct timeval tv;
   gettimeofday(&tv, NULL);
   fini = tv.tv_sec * 1000000LL + tv.tv_usec;
 }
#else
# error "pb_StopTimer: not implemented for this system"
#endif

 accumulate_time(&timer->elapsed, timer->init, fini);
}

/* Get the elapsed time in seconds. */
double
pb_GetElapsedTime(struct pb_Timer *timer)
{
 double ret;

 if (timer->state != pb_Timer_STOPPED) {
   fputs("Elapsed time from a running timer is inaccurate\n", stderr);
 }

#if _POSIX_VERSION >= 200112L
 ret = timer->elapsed / 1e6;
#else
# error "pb_GetElapsedTime: not implemented for this system"
#endif
 return ret;
}

void
pb_InitializeTimerSet(struct pb_TimerSet *timers)
{
 int n;

 timers->current = pb_TimerID_NONE;

 for (n = 0; n < pb_TimerID_LAST; n++)
 pb_ResetTimer(&timers->timers[n]);
}

void
pb_SwitchToTimer(struct pb_TimerSet *timers, enum pb_TimerID timer)
{
 /* Stop the currently running timer */
 if (timers->current != pb_TimerID_NONE)
   pb_StopTimer(&timers->timers[timers->current]);

 timers->current = timer;

 /* Start the new timer */
 if (timer != pb_TimerID_NONE)
   pb_StartTimer(&timers->timers[timer]);
}

void
pb_PrintTimerSet(struct pb_TimerSet *timers)
{
 struct pb_Timer *t = timers->timers;

 double time, totTime;

 time = pb_GetElapsedTime(&t[pb_TimerID_IO]);
 totTime = time;
 printf("IO:      %f\n", time);

 time = pb_GetElapsedTime(&t[pb_TimerID_GPU]);
 totTime += time;
 printf("GPU:     %f\n", time);

 time = pb_GetElapsedTime(&t[pb_TimerID_COPY]);
 totTime += time;
 printf("Copy:    %f\n", time);

 time = pb_GetElapsedTime(&t[pb_TimerID_COMPUTE]);
 totTime += time;
 printf("Compute: %f\n", time);

 printf("Total:   %f\n", totTime);
}

