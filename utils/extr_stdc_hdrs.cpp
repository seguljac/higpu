//
// Extract standard C headers from the given pre-processed source files.
//
// extr_stdc_hdrs [std_c_include_dirs] [[input .i files]]
//
// Output the list of #include's to stdout.
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#define MAX_NUM_SYS_INC_DIRS 32
#define MAX_LINE_LEN 2048

static const char* g_stdc_hdr_names[] = {
    "assert.h",
    "complex.h",
    "ctype.h",
    "errno.h",
    "fenv.h",
    "float.h",
    "inttypes.h",
    "limits.h",
//  "math.h",
    "locale.h",
    "setjump.h",
    "signal.h",
    "stdarg.h",
    "stdbool.h",
    "stdint.h",
    "stddef.h",
    "stdio.h",
    "stdlib.h",
//  "string.h",
    "tgmath.h",
    "time.h",
// The following are not ISO C headers.
    "unistd.h",
    "sys/time.h"
};

static int g_num_stdc_hdrs = sizeof(g_stdc_hdr_names) / sizeof(const char*);

// Assume at most 32 system include directories.
static char* g_sys_inc_dirs[MAX_NUM_SYS_INC_DIRS] = {
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
};

static int g_num_sys_inc_dirs = 0;

//////////////////////////////////////////////////////////////////////////////

// Read the list of system include directories from the given input file.
//
static void read_system_include_dirs(FILE *fin)
{
    char line[MAX_LINE_LEN];

    while (fgets(line, MAX_LINE_LEN, fin) != NULL)
    {
        if (g_num_sys_inc_dirs == MAX_NUM_SYS_INC_DIRS)
        {
            fprintf(stderr,
                    "WARNING: more than %d system include directories!\n",
                    MAX_NUM_SYS_INC_DIRS);
            break;
        }

        // Remove the trailing \n.
        int line_len = strlen(line);
        line[line_len-1] = '\0';

        // Remove the trailing '/', if any.
        if (line[line_len-2] == '/')
        {
            --line_len;
            line[line_len-1] = '\0';
        }

        char *dir = (char*)malloc(line_len);
        strcpy(dir, line);
        g_sys_inc_dirs[g_num_sys_inc_dirs++] = dir;
    }
}

static void free_system_include_dirs()
{
    for (int i = 0; i < g_num_sys_inc_dirs; ++i)
    {
        free(g_sys_inc_dirs[i]);
        g_sys_inc_dirs[i] = NULL;
    }
}

//////////////////////////////////////////////////////////////////////////////

static void parse_input_file(FILE *fin, bool *stdc_hdr_present)
{
    char line[MAX_LINE_LEN];

    typedef enum {
        SYS_INC_DIR,
        NON_SYS_INC_DIR,
        OTHERS
    } SRC_LINE_INFO;

    SRC_LINE_INFO prev_line_info, curr_line_info = OTHERS;

    while (fgets(line, MAX_LINE_LEN, fin) != NULL)
    {
        // Skip empty lines.
        // Sometime the previous non-system-include path is separated from
        // this one by empty lines.
        if (line[0] == '\n') continue;

        prev_line_info = curr_line_info;
        curr_line_info = OTHERS;

        // We only care about lines that start with "#".
        if (line[0] != '#') continue;

        // Tokenize the line.
        // Read the 3rd token, which is the header file path.
        char *token = strtok(line, " \n");
        if (token == NULL || strcmp(token, "#") != 0) continue;
        token = strtok(NULL, " \n");
        if (token == NULL) continue;
        token = strtok(NULL, " \n");
        if (token == NULL || strlen(token) <= 2) continue;

        // The 3rd token is surrounded by double-quotes.
        char *token_end = &token[strlen(token)-1];
        if (token[0] != '"' || *token_end != '"') continue;
        // Trim the quotes.
        ++token; --token_end;

        // Is this path under a system include directory?
        // NOTE: we cannot determine which system include directory this
        // header belongs to because the system include directories may
        // contain each other.
        curr_line_info = NON_SYS_INC_DIR;
        for (int i = 0; i < g_num_sys_inc_dirs; ++i)
        {
            if (strncmp(token, g_sys_inc_dirs[i],
                        strlen(g_sys_inc_dirs[i])) == 0)
            {
                curr_line_info = SYS_INC_DIR;
                break;
            }
        }
        if (curr_line_info == NON_SYS_INC_DIR) continue;

        // We only check the header file name if the previous line is a
        // non-system include directory.
        if (prev_line_info != NON_SYS_INC_DIR) continue;

        // Check it against the list of standard C header files.
        for (int i = 0; i < g_num_stdc_hdrs; ++i)
        {
            // No need to check for one that is already present.
            if (stdc_hdr_present[i]) continue;

            // NOTE: some header file names have '/' in them.
            char *c1 = token_end;
            const char *hdr_name = g_stdc_hdr_names[i];
            const char *c2 = &hdr_name[strlen(hdr_name)-1];
            while (c1 >= token && c2 >= hdr_name && *c1 == *c2)
            {
                --c1; --c2;
            }
            if (c2 >= hdr_name) continue;

            // After matching the header file name, we still need to make sure
            // that the directory that contains this file IS one of the system
            // include paths.
            assert(*c1 == '/');
            *c1 = '\0';
            for (int j = 0; j < g_num_sys_inc_dirs; ++j)
            {
                if (strcmp(token, g_sys_inc_dirs[j]) == 0)
                {
                    stdc_hdr_present[i] = true;
                    break;
                }
            }
            *c1 = '/';
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

static void print_usage(const char *prog_name)
{
    fprintf(stderr,
            "\nUsage: %s [system_include_dir_file] [input .i files]\n\n",
            prog_name);
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        print_usage(argv[0]);
        return 1;
    }

    // Read the system-include-directory file.
    FILE *fin = fopen(argv[1], "r");
    if (fin == NULL)
    {
        fprintf(stderr, "Failed to read the system include dir file.\n");
        return 1;
    }
    read_system_include_dirs(fin);
    fclose(fin);

    bool stdc_hdr_present[g_num_stdc_hdrs];
    for (int i = 0; i < g_num_stdc_hdrs; ++i) stdc_hdr_present[i] = false;

    // Process the input files one by one.
    for (int i = 2; i < argc; ++i)
    {
        fin = fopen(argv[i], "r");
        if (fin == NULL)
        {
            fprintf(stderr, "Failed to open <%s>! Skip it.\n", argv[i]);
            continue;
        }
        parse_input_file(fin, stdc_hdr_present);
        fclose(fin);
    }

    // Print out the standard C headers that are referenced.
    for (int i = 0; i < g_num_stdc_hdrs; ++i)
    {
        if (!stdc_hdr_present[i]) continue;
        printf("#include <%s>\n", g_stdc_hdr_names[i]);
    }

    // Clean up.
    free_system_include_dirs();

    return 0;
}
