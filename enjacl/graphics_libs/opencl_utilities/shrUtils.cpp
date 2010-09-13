/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */
 
// *********************************************************************
// Generic Utilities for NVIDIA GPU Computing SDK 
// *********************************************************************

// includes
#include <shrUtils.h>
#include <cmd_arg_reader.h>
#include <fstream>
#include <iostream>

// size of PGM file header 
const unsigned int PGMHeaderSize = 0x40;
#define MIN_EPSILON_ERROR 1e-3f

// Deallocate memory allocated within shrUtils
// *********************************************************************
void shrFree(void* ptr) 
{
  if( NULL != ptr) free( ptr);
}

// Helper function to init data arrays 
// *********************************************************************
void shrFillArray(float* pfData, int iSize)
{
    int i; 
    const float fScale = 1.0f / (float)RAND_MAX;
    for (i = 0; i < iSize; ++i) 
    {
        pfData[i] = fScale * rand();
    }
}

// Helper function to print data arrays 
// *********************************************************************
void shrPrintArray(float* pfData, int iSize)
{
    int i;
    for (i = 0; i < iSize; ++i) 
    {
        shrLog("%d: %.3f\n", i, pfData[i]);
    }
}

// Helper function to return precision delta time for 3 counters since last call based upon host high performance counter
// *********************************************************************
double shrDeltaT(int iCounterID = 0)
{
    // local var for computation of microseconds since last call
    double DeltaT;

    #ifdef _WIN32 // Windows version of precision host timer

        // Variables that need to retain state between calls
        static LARGE_INTEGER liOldCount0 = {0, 0};
        static LARGE_INTEGER liOldCount1 = {0, 0};
        static LARGE_INTEGER liOldCount2 = {0, 0};

        // locals for new count, new freq and new time delta 
	    LARGE_INTEGER liNewCount, liFreq;
	    if (QueryPerformanceFrequency(&liFreq))
	    {
		    // Get new counter reading
		    QueryPerformanceCounter(&liNewCount);

            // Update the requested timer
		    switch (iCounterID)
		    {
			    case 0: 
			    {
				    // Calculate time difference for timer 0.  (zero when called the first time) 
				    DeltaT = liOldCount0.LowPart ? (((double)liNewCount.QuadPart - (double)liOldCount0.QuadPart) / (double)liFreq.QuadPart) : 0.0;

				    // Reset old count to new
				    liOldCount0 = liNewCount;

				    break;
			    }
			    case 1:
			    {
				    // Calculate time difference for timer 1.  (zero when called the first time) 
				    DeltaT = liOldCount1.LowPart ? (((double)liNewCount.QuadPart - (double)liOldCount1.QuadPart) / (double)liFreq.QuadPart) : 0.0;

				    // Reset old count to new
				    liOldCount1 = liNewCount;

				    break;
			    }
			    case 2:
			    {
				    // Calculate time difference for timer 2.  (zero when called the first time) 
				    DeltaT = liOldCount2.LowPart ? (((double)liNewCount.QuadPart - (double)liOldCount2.QuadPart) / (double)liFreq.QuadPart) : 0.0;

				    // Reset old count to new
				    liOldCount2 = liNewCount;

				    break;
			    }
			    default: 
			    {
		            // Requested counter ID out of range
		            return -9999.0;
			    }
		    }

		    // Returns time difference in seconds sunce the last call
		    return DeltaT;
	    }
	    else
	    {
		    // No high resolution performance counter
		    return -9999.0;
	    }
    #else // Linux version of precision host timer. See http://www.informit.com/articles/article.aspx?p=23618&seqNum=8
        static struct timeval _NewTime;  // new wall clock time (struct representation in seconds and microseconds)
        static struct timeval _OldTime0; // old wall clock time 0(struct representation in seconds and microseconds)
        static struct timeval _OldTime1; // old wall clock time 1(struct representation in seconds and microseconds)
        static struct timeval _OldTime2; // old wall clock time 2(struct representation in seconds and microseconds)

        // Get new counter reading
        gettimeofday(&_NewTime, NULL);

	    switch (iCounterID)
	    {
		    case 0: 
		    {
			    // Calculate time difference for timer 0.  (zero when called the first time) 
                DeltaT =  ((double)_NewTime.tv_sec + 1.0e-6 * (double)_NewTime.tv_usec) - ((double)_OldTime0.tv_sec + 1.0e-6 * (double)_OldTime0.tv_usec);

			    // Reset old time 0 to new
			    _OldTime0.tv_sec = _NewTime.tv_sec;
			    _OldTime0.tv_usec = _NewTime.tv_usec;

			    break;
		    }
		    case 1:
		    {
			    // Calculate time difference for timer 1.  (zero when called the first time) 
                DeltaT =  ((double)_NewTime.tv_sec + 1.0e-6 * (double)_NewTime.tv_usec) - ((double)_OldTime1.tv_sec + 1.0e-6 * (double)_OldTime1.tv_usec);

			    // Reset old time 1 to new
			    _OldTime1.tv_sec = _NewTime.tv_sec;
			    _OldTime1.tv_usec = _NewTime.tv_usec;

			    break;
		    }
		    case 2:
		    {
			    // Calculate time difference for timer 2.  (zero when called the first time) 
                DeltaT =  ((double)_NewTime.tv_sec + 1.0e-6 * (double)_NewTime.tv_usec) - ((double)_OldTime2.tv_sec + 1.0e-6 * (double)_OldTime2.tv_usec);

			    // Reset old time 2 to new
			    _OldTime2.tv_sec = _NewTime.tv_sec;
			    _OldTime2.tv_usec = _NewTime.tv_usec;

			    break;
		    }
		    default: 
		    {
	            // Requested counter ID out of range
	            return -9999.0;
		    }
	    }

	    // Returns time difference in seconds sunce the last call
	    return DeltaT;
    #endif
} 

// Optional LogFileName Override function
// *********************************************************************
char* cLogFilePathAndName = NULL;
void shrSetLogFileName (const char* cOverRideName)
{
    if( cLogFilePathAndName != NULL ) {
        free(cLogFilePathAndName);
    }
    cLogFilePathAndName = (char*) malloc(strlen(cOverRideName) + 1);
    #ifdef WIN32
        strcpy_s(cLogFilePathAndName, strlen(cOverRideName) + 1, cOverRideName);
    #else
        strcpy(cLogFilePathAndName, cOverRideName);
    #endif
    return;
}

// Function to log standardized information to console, file or both
// *********************************************************************
static int shrLogV(int iLogMode, int iErrNum, const char* cFormatString, va_list vaArgList)
{
    static FILE* pFileStream0 = NULL;
    static FILE* pFileStream1 = NULL;
    size_t szNumWritten = 0;
    char cFileMode [3];
    const char*     pStr; 
    const char*     cArg;
    int             iArg;
    double          dArg;
    unsigned int    uiArg;

    // if the sample log file is closed and the call incudes a "write-to-file", open file for writing
    if ((pFileStream0 == NULL) && (iLogMode & LOGFILE))
    {
        // if the default filename has not been overriden, set to default
        if (cLogFilePathAndName == NULL)
        {
            shrSetLogFileName(DEFAULTLOGFILE); 
        }

        #ifdef _WIN32   // Windows version
            // set the file mode
            if (iLogMode & APPENDMODE)  // append to prexisting file contents
            {
                sprintf_s (cFileMode, 3, "a+");  
            }
            else                        // replace prexisting file contents
            {
                sprintf_s (cFileMode, 3, "w"); 
            }

            // open the individual sample log file in the requested mode
            errno_t err = fopen_s(&pFileStream0, cLogFilePathAndName, cFileMode);
            
            // if error on attempt to open, be sure the file is null or close it, then return negative error code            
            if (err != 0)
            {
                if (pFileStream0)
                {
                    fclose (pFileStream0);
                }
                return -err;
            }
        #else           // Linux & Mac version
            // set the file mode
            if (iLogMode & APPENDMODE)  // append to prexisting file contents
            {
                sprintf (cFileMode, "a+");  
            }
            else                        // replace prexisting file contents
            {
                sprintf (cFileMode, "w"); 
            }

            // open the file in the requested mode
            if ((pFileStream0 = fopen(cLogFilePathAndName, cFileMode)) == 0)
            {
                // if error on attempt to open, be sure the file is null or close it, then return negative error code
                if (pFileStream0)
                {
                    fclose (pFileStream0);
                }
                return -1;
            }
        #endif
    }
    
    // if the master log file is closed and the call incudes a "write-to-file" and MASTER, open master logfile file for writing
    if ((pFileStream1 == NULL) && (iLogMode & LOGFILE) && (iLogMode & MASTER))
    {

        #ifdef _WIN32   // Windows version
            // open the master log file in append mode
            errno_t err = fopen_s(&pFileStream1, MASTERLOGFILE, "a+");

            // if error on attempt to open, be sure the file is null or close it, then return negative error code
            if (err != 0)
            {
                if (pFileStream1)
                {
                    fclose (pFileStream1);
                }
                return -err;
            }
        #else           // Linux & Mac version

            // open the file in the requested mode
            if ((pFileStream1 = fopen(MASTERLOGFILE, "a+")) == 0)
            {
                // if error on attempt to open, be sure the file is null or close it, then return negative error code
                if (pFileStream1)
                {
                    fclose (pFileStream1);
                }
                return -1;
            }
        #endif
        
        // If master log file length has become excessive, empty/reopen
        fseek(pFileStream1, 0L, SEEK_END);            
        if (ftell(pFileStream1) > 50000L)
        {
            fclose (pFileStream1);
        #ifdef _WIN32   // Windows version
            fopen_s(&pFileStream1, MASTERLOGFILE, "w");
	    #else
	        pFileStream1 = fopen(MASTERLOGFILE, "w");
	    #endif
        }
    }

    // Handle special Error Message code
    if (iLogMode & ERRORMSG)  
    {   
        // print string to console if flagged
        if (iLogMode & LOGCONSOLE) 
        {
            szNumWritten = printf ("\n !!! Error # %i at ", iErrNum);
        }
        // print string to file if flagged
        if (iLogMode & LOGFILE) 
        {
            szNumWritten = fprintf (pFileStream0, "\n !!! Error # %i at ", iErrNum);
        }
    }

    // Start at the head of the string and scan to the null at the end
    for (pStr = cFormatString; *pStr; ++pStr)
    {
        // Check if the current character is not a formatting specifier ('%') 
        if (*pStr != '%')
        {
            // not "%", so print verbatim to console and/or files as flagged
            if (iLogMode & LOGCONSOLE) // to console if flagged
            {
                szNumWritten = putc(*pStr, stdout);
            }
            if (iLogMode & LOGFILE)    
            {
                szNumWritten  = putc(*pStr, pFileStream0);      // sample log file
                if (iLogMode & MASTER)                          
                {
                    szNumWritten  = putc(*pStr, pFileStream1);  // master log file
                }
            }
        } 
        else 
        {
            // character was %, so handle the next arg according to next character
            switch (*++pStr)
            {
                case 's':   // string with default precision (max length for string)
                {
                    // Set cArg as the next value in list
                    cArg = va_arg(vaArgList, char*);

                    // print string to console and/or files if flagged
                    if (iLogMode & LOGCONSOLE) 
                    {
                        szNumWritten = printf("%s", cArg);
                    }
                    if (iLogMode & LOGFILE)
                    {
                        szNumWritten = fprintf (pFileStream0, "%s", cArg);
                        if (iLogMode & MASTER)                          
                        {
                            szNumWritten  = fprintf(pFileStream1, "%s", cArg);  // master log file
                        }
                    }
                    continue;
                }
                case 'd':   // integer with default precision
                case 'i':   // integer with default precision
                {
                    // set iArg as the next value in list
                    iArg = va_arg(vaArgList, int);

                    // print string to console and/or file if flagged
                    if (iLogMode & LOGCONSOLE) 
                    {
                        szNumWritten = printf("%i", iArg);
                    }
                    if (iLogMode & LOGFILE)
                    {
                        szNumWritten = fprintf (pFileStream0, "%i", iArg);
                        if (iLogMode & MASTER)                          
                        {
                            szNumWritten  = fprintf(pFileStream1, "%i", iArg);  // master log file
                        }
                    }
                    continue;
                }
                case 'u':   // unsigned integer with default precision
                {
                    // set uiArg as the next value in list
                    uiArg = va_arg(vaArgList, unsigned int);

                    // print string to console and/or file if flagged
                    if (iLogMode & LOGCONSOLE) 
                    {
                        szNumWritten = printf("%u", uiArg);
                    }
                    if (iLogMode & LOGFILE)
                    {
                        szNumWritten = fprintf (pFileStream0, "%u", uiArg);
                        if (iLogMode & MASTER)                          
                        {
                            szNumWritten  = fprintf(pFileStream1, "%u", uiArg);  // master log file
                        }
                    }
                    continue;
                }
                case 'e':   // scientific double/float with default precision
                case 'E':   // scientific double/float with default precision
                {
                    // set dArg as the next value in list
                    dArg = va_arg(vaArgList, double);

                    // print string to console and/or file if flagged
                    if (iLogMode & LOGCONSOLE) 
                    {
                        szNumWritten = printf("%e", dArg);
                    }
                    if (iLogMode & LOGFILE)
                    {
                        szNumWritten = fprintf (pFileStream0, "%e", dArg);
                        if (iLogMode & MASTER)                          
                        {
                            szNumWritten  = fprintf(pFileStream1, "%e", dArg);  // master log file
                        }
                    }
                    continue;
                }
                case 'f':  // float/double with default precision
                {
                    // set dArg as the next value in list
                    dArg = va_arg(vaArgList, double);

                    // print string to console and/or file if flagged
                    if (iLogMode & LOGCONSOLE) 
                    {
                        szNumWritten = printf("%f", dArg);  
                    }
                    if (iLogMode & LOGFILE)
                    {
                        szNumWritten = fprintf (pFileStream0, "%f", dArg);
                        if (iLogMode & MASTER)                          
                        {
                            szNumWritten  = fprintf(pFileStream1, "%f", dArg);  // master log file
                        }
                    }

                    continue;
                }
                case '.':  // Type and Precision are specified (f, e, E, i, d, u, s)
                {
                    // format with specified type and precision 
                    char cFormat [5] = "%.3f";
                    cFormat[2] = *++pStr;   // jump past the decimal pt and get the precision width
                    cFormat[3] = *++pStr;   // jump past the precision val and get type field (f, e, E, i, d, u, s)   

                    switch(cFormat[3])
                    {
                        case 'f':
                        case 'e':
                        case 'E':
                        {
                            // set dArg as the next value in list
                            dArg = va_arg(vaArgList, double);

                            // print string to console if flagged
                            if (iLogMode & LOGCONSOLE) 
                            {
                                szNumWritten = printf(cFormat, dArg);
                            }

                            // print string to file if flagged
                            if (iLogMode & LOGFILE)
                            {
                                szNumWritten = fprintf (pFileStream0, cFormat, dArg);
                                if (iLogMode & MASTER)                          
                                {
                                    szNumWritten  = fprintf(pFileStream1, cFormat, dArg);  // master log file
                                }
                            }
                            break;
                        }
                        case 'd':
                        case 'i':
                        {
                            // set iArg as the next value in list
                            iArg = va_arg(vaArgList, int);

                            // print string to console if flagged
                            if (iLogMode & LOGCONSOLE) 
                            {
                                szNumWritten = printf(cFormat, iArg);
                            }

                            // print string to file if flagged
                            if (iLogMode & LOGFILE)
                            {
                                szNumWritten = fprintf (pFileStream0, cFormat, iArg);
                                if (iLogMode & MASTER)                          
                                {
                                    szNumWritten  = fprintf(pFileStream1, cFormat, iArg);  // master log file
                                }
                            }
                            break;
                        }
                        case 'u':
                        {
                            // set uiArg as the next value in list
                            uiArg = va_arg(vaArgList, unsigned int);

                            // print string to console if flagged
                            if (iLogMode & LOGCONSOLE) 
                            {
                                szNumWritten = printf(cFormat, uiArg);
                            }

                            // print string to file if flagged
                            if (iLogMode & LOGFILE)
                            {
                                szNumWritten = fprintf (pFileStream0, cFormat, uiArg);
                                if (iLogMode & MASTER)                          
                                {
                                    szNumWritten  = fprintf(pFileStream1, cFormat, uiArg);  // master log file
                                }
                            }
                            break;
                        }
                        case 's':
                        {
                            // Set cArg as the next value in list
                            cArg = va_arg(vaArgList, char*);

                            // print string to console if flagged
                            if (iLogMode & LOGCONSOLE) 
                            {
                                szNumWritten = printf(cFormat, cArg);
                            }

                            // print string to file if flagged
                            if (iLogMode & LOGFILE)
                            {
                                szNumWritten = fprintf (pFileStream0, cFormat, cArg);
                                if (iLogMode & MASTER)                          
                                {
                                    szNumWritten  = fprintf(pFileStream1, cFormat, cArg);  // master log file
                                }
                            }
                            break;
                        }
                    }
                    continue;
                }
                default: 
                {
                    if (iLogMode & LOGCONSOLE) // to console if flagged
                    {
                        szNumWritten = putc(*pStr, stdout);
                    }
                    if (iLogMode & LOGFILE)    
                    {
                        szNumWritten  = putc(*pStr, pFileStream0);      // sample log file
                        if (iLogMode & MASTER)                          
                        {
                            szNumWritten  = putc(*pStr, pFileStream1);  // master log file
                        }
                    }
                }
            }
        }
    }

    // end the sample log(s) with a horizontal line if closing
    if (iLogMode & CLOSELOG) 
    {
        if (iLogMode & LOGCONSOLE) 
        {
            printf(HDASHLINE);
        }
        if (iLogMode & LOGFILE)
        {
            fprintf(pFileStream0, HDASHLINE);
        }
    }

    // flush console and/or file buffers if updated
    if (iLogMode & LOGCONSOLE) 
    {
        fflush(stdout);
    }
    if (iLogMode & LOGFILE)
    {
        fflush (pFileStream0);

        // if the master log file has been updated, flush it too
        if (iLogMode & MASTER)
        {
            fflush (pFileStream1);
        }
    }

    // If the log file is open and the caller requests "close file", then close and NULL file handle
    if ((pFileStream0) && (iLogMode & CLOSELOG))
    {
        fclose (pFileStream0);
        pFileStream0 = NULL;
    }
    if ((pFileStream1) && (iLogMode & CLOSELOG))
    {
        fclose (pFileStream1);
        pFileStream1 = NULL;
    }

    // return error code or OK 
    if (iLogMode & ERRORMSG)
    {
        return iErrNum;
    }
    else 
    {
        return 0;
    }
}

// Function to log standardized information to console, file or both
// *********************************************************************
int shrLogEx(int iLogMode = LOGCONSOLE, int iErrNum = 0, const char* cFormatString = "", ...)
{
    va_list vaArgList;

    // Prepare variable agument list 
    va_start(vaArgList, cFormatString);
    int ret = shrLogV(iLogMode, iErrNum, cFormatString, vaArgList);

    // end variable argument handler
    va_end(vaArgList);

    return ret;
}

// Function to log standardized information to console, file or both
// *********************************************************************
int shrLog(const char* cFormatString = "", ...)
{
    va_list vaArgList;

    // Prepare variable agument list 
    va_start(vaArgList, cFormatString);
    int ret = shrLogV(LOGBOTH, 0, cFormatString, vaArgList);

    // end variable argument handler
    va_end(vaArgList);

    return ret;
}

//////////////////////////////////////////////////////////////////////////////
//! Find the path for a file assuming that
//! files are found in the searchPath.
//!
//! @return the path if succeeded, otherwise 0
//! @param filename         name of the file
//! @param executable_path  optional absolute path of the executable
//////////////////////////////////////////////////////////////////////////////
char* shrFindFilePath(const char* filename, const char* executable_path) 
{
    // <executable_name> defines a variable that is replaced with the name of the executable

    // Typical relative search paths to locate needed companion files (e.g. sample input data, or JIT source files)
    // The origin for the relative search may be the .exe file, a .bat file launching an .exe, a browser .exe launching the .exe or .bat, etc
    const char* searchPath[] = {
        "./",                                       // same dir 
        "./data/",                                  // "/data/" subdir 
        "./src/",                                   // "/src/" subdir
        "./inc/",                                   // "/inc/" subdir
        "../",                                      // up 1 in tree 
        "../data/",                                 // up 1 in tree, "/data/" subdir 
        "../src/",                                  // up 1 in tree, "/src/" subdir 
        "../inc/",                                  // up 1 in tree, "/inc/" subdir 
        "../OpenCL/src/<executable_name>/",         // up 1 in tree, "/OpenCL/src/<executable_name>/" subdir 
        "../OpenCL/src/<executable_name>/data/",    // up 1 in tree, "/OpenCL/src/<executable_name>/data/" subdir 
        "../OpenCL/src/<executable_name>/src/",     // up 1 in tree, "/OpenCL/src/<executable_name>/src/" subdir 
        "../OpenCL/src/<executable_name>/inc/",     // up 1 in tree, "/OpenCL/src/<executable_name>/inc/" subdir 
        "../C/src/<executable_name>/",              // up 1 in tree, "/C/src/<executable_name>/" subdir 
        "../C/src/<executable_name>/data/",         // up 1 in tree, "/C/src/<executable_name>/data/" subdir 
        "../C/src/<executable_name>/src/",          // up 1 in tree, "/C/src/<executable_name>/src/" subdir 
        "../C/src/<executable_name>/inc/",          // up 1 in tree, "/C/src/<executable_name>/inc/" subdir 
        "../DirectCompute/src/<executable_name>/",      // up 1 in tree, "/DirectCompute/src/<executable_name>/" subdir 
        "../DirectCompute/src/<executable_name>/data/", // up 1 in tree, "/DirectCompute/src/<executable_name>/data/" subdir 
        "../DirectCompute/src/<executable_name>/src/",  // up 1 in tree, "/DirectCompute/src/<executable_name>/src/" subdir 
        "../DirectCompute/src/<executable_name>/inc/",  // up 1 in tree, "/DirectCompute/src/<executable_name>/inc/" subdir 
        "../../",                                   // up 2 in tree 
        "../../data/",                              // up 2 in tree, "/data/" subdir 
        "../../src/",                               // up 2 in tree, "/src/" subdir 
        "../../inc/",                               // up 2 in tree, "/inc/" subdir 
        "../../../",                                // up 3 in tree 
        "../../../src/<executable_name>/",          // up 3 in tree, "/src/<executable_name>/" subdir 
        "../../../src/<executable_name>/data/",     // up 3 in tree, "/src/<executable_name>/data/" subdir 
        "../../../src/<executable_name>/src/",      // up 3 in tree, "/src/<executable_name>/src/" subdir 
        "../../../src/<executable_name>/inc/",      // up 3 in tree, "/src/<executable_name>/inc/" subdir 
        "../../../sandbox/<executable_name>/",      // up 3 in tree, "/sandbox/<executable_name>/" subdir
        "../../../sandbox/<executable_name>/data/", // up 3 in tree, "/sandbox/<executable_name>/data/" subdir
        "../../../sandbox/<executable_name>/src/",  // up 3 in tree, "/sandbox/<executable_name>/src/" subdir
        "../../../sandbox/<executable_name>/inc/"   // up 3 in tree, "/sandbox/<executable_name>/inc/" subdir
    };
    
    // Extract the executable name
    std::string executable_name;
    if (executable_path != 0) 
    {
        executable_name = std::string(executable_path);

    #ifdef _WIN32        
        // Windows path delimiter
        size_t delimiter_pos = executable_name.find_last_of('\\');        
        executable_name.erase(0,delimiter_pos+1);

		if (executable_name.rfind(".exe") != string::npos) {
			// we strip .exe, only if the .exe is found
			executable_name.resize(executable_name.size() - 4);        
		}
    #else
        // Linux & OSX path delimiter
        size_t delimiter_pos = executable_name.find_last_of('/');        
        executable_name.erase(0,delimiter_pos+1);
    #endif
        
    }
    
    // Loop over all search paths and return the first hit
    for( unsigned int i=0; i<sizeof(searchPath)/sizeof(char*); ++i )
    {
        std::string path(searchPath[i]);        
        size_t executable_name_pos = path.find("<executable_name>");

        // If there is executable_name variable in the searchPath 
        // replace it with the value
        if( executable_name_pos != std::string::npos )
        {
            if( executable_path != 0 ) {
                
                path.replace(executable_name_pos, strlen("<executable_name>"), executable_name);

            } 
            else 
            {
                // Skip this path entry if no executable argument is given
                continue;
            }
        }
        
        // Test if the file exists
        path.append(filename);
        std::fstream fh(path.c_str(), std::fstream::in);
        if (fh.good())
        {
            // File found
            // returning an allocated array here for backwards compatibility reasons
            char* file_path = (char*) malloc(path.length() + 1);
        #ifdef _WIN32  
            strcpy_s(file_path, path.length() + 1, path.c_str());
        #else
            strcpy(file_path, path.c_str());
        #endif                
            return file_path;
        }
    }    

    // File not found
    return 0;
}

//////////////////////////////////////////////////////////////////////////////
//! Read file \filename and return the data
//! @return shrTRUE if reading the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
//////////////////////////////////////////////////////////////////////////////
#if 0
template<class T>
shrBOOL
shrReadFile( const char* filename, T** data, unsigned int* len, bool verbose) 
{
    // check input arguments
    ARGCHECK(NULL != filename);
    ARGCHECK(NULL != len);

    // intermediate storage for the data read
    std::vector<T>  data_read;

    // open file for reading
    std::fstream fh( filename, std::fstream::in);
    // check if filestream is valid
    if(!fh.good()) 
    {
        if (verbose)
            std::cerr << "shrReadFile() : Opening file failed." << std::endl;
        return shrFALSE;
    }

    // read all data elements 
    T token;
    while( fh.good()) 
    {
        fh >> token;   
        data_read.push_back( token);
    }

    // the last element is read twice
    data_read.pop_back();

    // check if reading result is consistent
    if( ! fh.eof()) 
    {
        if (verbose)
            std::cerr << "WARNING : readData() : reading file might have failed." 
            << std::endl;
    }

    fh.close();

    // check if the given handle is already initialized
    if( NULL != *data) 
    {
        if( *len != data_read.size()) 
        {
            std::cerr << "shrReadFile() : Initialized memory given but "
                      << "size  mismatch with signal read "
                      << "(data read / data init = " << (unsigned int)data_read.size()
                      <<  " / " << *len << ")" << std::endl;

            return shrFALSE;
        }
    }
    else 
    {
        // allocate storage for the data read
		*data = (T*) malloc( sizeof(T) * data_read.size());
        // store signal size
        *len = static_cast<unsigned int>( data_read.size());
    }

    // copy data
    memcpy( *data, &data_read.front(), sizeof(T) * data_read.size());

    return shrTRUE;
}
#endif

//////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename 
//! @return shrTRUE if writing the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
//////////////////////////////////////////////////////////////////////////////
template<class T>
shrBOOL
shrWriteFile( const char* filename, const T* data, unsigned int len,
              const T epsilon, bool verbose) 
{
    ARGCHECK(NULL != filename);
    ARGCHECK(NULL != data);

    // open file for writing
    std::fstream fh( filename, std::fstream::out);
    // check if filestream is valid
    if(!fh.good()) 
    {
        if (verbose)
            std::cerr << "shrWriteFile() : Opening file failed." << std::endl;
        return shrFALSE;
    }

    // first write epsilon
    fh << "# " << epsilon << "\n";

    // write data
    for( unsigned int i = 0; (i < len) && (fh.good()); ++i) 
    {
        fh << data[i] << ' ';
    }

    // Check if writing succeeded
    if( ! fh.good()) 
    {
        if (verbose)
            std::cerr << "shrWriteFile() : Writing file failed." << std::endl;
        return shrFALSE;
    }

    // file ends with nl
    fh << std::endl;

    return shrTRUE;
}

////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg single precision floating point data 
//! @return shrTRUEif reading the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
#if 0
shrBOOL shrReadFilef( const char* filename, float** data, unsigned int* len, bool verbose) 
{
    return shrReadFile( filename, data, len, verbose);
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg double precision floating point data 
//! @return shrTRUEif reading the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
#if 0
shrBOOL shrReadFiled( const char* filename, double** data, unsigned int* len, bool verbose) 
{
    return shrReadFile( filename, data, len, verbose);
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg integer data 
//! @return shrTRUEif reading the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
#if 0
shrBOOL shrReadFilei( const char* filename, int** data, unsigned int* len, bool verbose) 
{
    return shrReadFile( filename, data, len, verbose);
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg unsigned integer data 
//! @return shrTRUEif reading the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
#if 0
shrBOOL shrReadFileui( const char* filename, unsigned int** data, unsigned int* len, bool verbose) 
{
    return shrReadFile( filename, data, len, verbose);
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg char / byte data 
//! @return shrTRUEif reading the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
#if 0
shrBOOL shrReadFileb( const char* filename, char** data, unsigned int* len, bool verbose) 
{
    return shrReadFile( filename, data, len, verbose);
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg unsigned char / byte data 
//! @return shrTRUEif reading the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
#if 0
shrBOOL shrReadFileub( const char* filename, unsigned char** data, unsigned int* len, bool verbose) 
{
    return shrReadFile( filename, data, len, verbose);
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for single precision floating point data
//! @return shrTRUEif writing the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
#if 0
shrBOOL shrWriteFilef( const char* filename, const float* data, unsigned int len,
               const float epsilon, bool verbose) 
{
    return shrWriteFile( filename, data, len, epsilon, verbose);
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for double precision floating point data
//! @return shrTRUEif writing the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
#if 0
shrBOOL shrWriteFiled( const char* filename, const double* data, unsigned int len,
               const double epsilon, bool verbose) 
{
    return shrWriteFile( filename, data, len, epsilon, verbose);
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for integer data
//! @return shrTRUEif writing the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
#if 0
shrBOOL shrWriteFilei( const char* filename, const int* data, unsigned int len, bool verbose) 
{
    return shrWriteFile( filename, data, len, 0, verbose);
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for unsigned integer data
//! @return shrTRUEif writing the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
#if 0
shrBOOL shrWriteFileui( const char* filename,const unsigned int* data,unsigned int len, bool verbose)
{
    return shrWriteFile( filename, data, len, static_cast<unsigned int>(0), verbose);
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for byte / char data
//! @return shrTRUEif writing the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
#if 0
shrBOOL shrWriteFileb( const char* filename, const char* data, unsigned int len, bool verbose) 
{  
    return shrWriteFile( filename, data, len, static_cast<char>(0), verbose);
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for byte / char data
//! @return shrTRUEif writing the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
#if 0
shrBOOL shrWriteFileub( const char* filename, const unsigned char* data, 
                unsigned int len, bool verbose) 
{  
    return shrWriteFile( filename, data, len, static_cast<unsigned char>(0), verbose);
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for unsigned byte / char data
//! @return shrTRUEif writing the file succeeded, otherwise shrFALSE
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
#if 0
shrBOOL shrWriteFileb( const char* filename,const unsigned char* data,unsigned int len, bool verbose)
{
    return shrWriteFile( filename, data, len, static_cast<unsigned char>(0), verbose);
}
#endif

//////////////////////////////////////////////////////////////////////////////
//! Load PGM or PPM file
//! @note if data == NULL then the necessary memory is allocated in the 
//!       function and w and h are initialized to the size of the image
//! @return shrTRUE if the file loading succeeded, otherwise shrFALSE
//! @param file        name of the file to load
//! @param data        handle to the memory for the image file data
//! @param w        width of the image
//! @param h        height of the image
//! @param channels number of channels in image
//////////////////////////////////////////////////////////////////////////////
shrBOOL loadPPM(const char* file, unsigned char** data, 
            unsigned int *w, unsigned int *h, unsigned int *channels) 
{
    FILE* fp = 0;

    #ifdef _WIN32
        // open the file for binary read
        errno_t err;
        if ((err = fopen_s(&fp, file, "rb")) != 0)
    #else
        // open the file for binary read
        if ((fp = fopen(file, "rb")) == 0)
    #endif
        {
            // if error on attempt to open, be sure the file is null or close it, then return negative error code
            if (fp)
            {
                fclose (fp);
            }
            std::cerr << "loadPPM() : Failed to open file: " << file << std::endl;
            return shrFALSE;
        }

    // check header
    char header[PGMHeaderSize];
    if ((fgets( header, PGMHeaderSize, fp) == NULL) && ferror(fp))
    {
        if (fp)
        {
            fclose (fp);
        }
        std::cerr << "loadPPM() : File is not a valid PPM or PGM image" << std::endl;
        *channels = 0;
        return shrFALSE;
    }

    if (strncmp(header, "P5", 2) == 0)
    {
        *channels = 1;
    }
    else if (strncmp(header, "P6", 2) == 0)
    {
        *channels = 3;
    }
    else
    {
        std::cerr << "loadPPM() : File is not a PPM or PGM image" << std::endl;
        *channels = 0;
        return shrFALSE;
    }

    // parse header, read maxval, width and height
    unsigned int width = 0;
    unsigned int height = 0;
    unsigned int maxval = 0;
    unsigned int i = 0;
    while(i < 3) 
    {
        if ((fgets(header, PGMHeaderSize, fp) == NULL) && ferror(fp))
        {
            if (fp)
            {
                fclose (fp);
            }
            std::cerr << "loadPPM() : File is not a valid PPM or PGM image" << std::endl;
            return shrFALSE;
        }
        if(header[0] == '#') continue;

        #ifdef _WIN32
            if(i == 0) 
            {
                i += sscanf_s(header, "%u %u %u", &width, &height, &maxval);
            }
            else if (i == 1) 
            {
                i += sscanf_s(header, "%u %u", &height, &maxval);
            }
            else if (i == 2) 
            {
                i += sscanf_s(header, "%u", &maxval);
            }
        #else
            if(i == 0) 
            {
                i += sscanf(header, "%u %u %u", &width, &height, &maxval);
            }
            else if (i == 1) 
            {
                i += sscanf(header, "%u %u", &height, &maxval);
            }
            else if (i == 2) 
            {
                i += sscanf(header, "%u", &maxval);
            }
        #endif
    }

    // check if given handle for the data is initialized
    if(NULL != *data) 
    {
        if (*w != width || *h != height) 
        {
            fclose(fp);
            std::cerr << "loadPPM() : Invalid image dimensions." << std::endl;
            return shrFALSE;
        }
    } 
    else 
    {
        *data = (unsigned char*)malloc( sizeof(unsigned char) * width * height * *channels);
        *w = width;
        *h = height;
    }

    // read and close file
    if (fread(*data, sizeof(unsigned char), width * height * *channels, fp) != width * height * *channels)
    {
        fclose(fp);
        std::cerr << "loadPPM() : Invalid image." << std::endl;
        return shrFALSE;
    }
    fclose(fp);

    return shrTRUE;
}

//////////////////////////////////////////////////////////////////////////////
//! Write / Save PPM or PGM file
//! @note Internal usage only
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
//////////////////////////////////////////////////////////////////////////////  
shrBOOL savePPM( const char* file, unsigned char *data, 
             unsigned int w, unsigned int h, unsigned int channels) 
{
    ARGCHECK(NULL != data);
    ARGCHECK(w > 0);
    ARGCHECK(h > 0);

    std::fstream fh( file, std::fstream::out | std::fstream::binary );
    if( fh.bad()) 
    {
        std::cerr << "savePPM() : Opening file failed." << std::endl;
        return shrFALSE;
    }

    if (channels == 1)
    {
        fh << "P5\n";
    }
    else if (channels == 3) {
        fh << "P6\n";
    }
    else {
        std::cerr << "savePPM() : Invalid number of channels." << std::endl;
        return shrFALSE;
    }

    fh << w << "\n" << h << "\n" << 0xff << std::endl;

    for( unsigned int i = 0; (i < (w*h*channels)) && fh.good(); ++i) 
    {
        fh << data[i];
    }
    fh.flush();

    if( fh.bad()) 
    {
        std::cerr << "savePPM() : Writing data failed." << std::endl;
        return shrFALSE;
    } 
    fh.close();

    return shrTRUE;
}

////////////////////////////////////////////////////////////////////////////////
//! Load PPM image file (with unsigned char as data element type), padding 4th component
//! @return shrTrue if reading the file succeeded, otherwise shrFALSE
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
////////////////////////////////////////////////////////////////////////////////
shrBOOL shrLoadPPM4ub( const char* file, unsigned char** data, 
                unsigned int *w,unsigned int *h)
{
    unsigned char *idata = 0;
    unsigned int channels;
    
    if (loadPPM( file, &idata, w, h, &channels)) {
        // pad 4th component
        int size = *w * *h;

        // keep the original pointer
        unsigned char* idata_orig = idata;
        *data = (unsigned char*) malloc( sizeof(unsigned char) * size * 4);
        unsigned char *ptr = *data;
        for(int i=0; i<size; i++) 
        {
            *ptr++ = *idata++;
            *ptr++ = *idata++;
            *ptr++ = *idata++;
            *ptr++ = 0;
        }
        free( idata_orig);
        return shrTRUE;
    }
    else
    {
        free(idata);
        return shrFALSE;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Save PPM image file (with unsigned char as data element type, padded to 4 byte)
//! @return shrTrue if reading the file succeeded, otherwise shrFALSE
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
////////////////////////////////////////////////////////////////////////////////
shrBOOL shrSavePPM4ub( const char* file, unsigned char *data, 
               unsigned int w, unsigned int h) 
{
    // strip 4th component
    int size = w * h;
    unsigned char *ndata = (unsigned char*) malloc( sizeof(unsigned char) * size*3);
    unsigned char *ptr = ndata;
    for(int i=0; i<size; i++) {
        *ptr++ = *data++;
        *ptr++ = *data++;
        *ptr++ = *data++;
        data++;
    }
    
    return savePPM(file, ndata, w, h, 3);
}

////////////////////////////////////////////////////////////////////////////////
//! Save PGM image file (with unsigned char as data element type)
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
////////////////////////////////////////////////////////////////////////////////
shrBOOL shrSavePGMub( const char* file, unsigned char *data, 
              unsigned int w, unsigned int h) 
{
    return savePPM( file, data, w, h, 1);
}

////////////////////////////////////////////////////////////////////////////////
//! Load PGM image file (with unsigned char as data element type)
//! @return shrTRUE if reading the file succeeded, otherwise shrFALSE
//! @param file  name of the image file
//! @param data  handle to the data read
//! @param w     width of the image
//! @param h     height of the image
////////////////////////////////////////////////////////////////////////////////
shrBOOL shrLoadPGMub( const char* file, unsigned char** data, 
              unsigned int *w,unsigned int *h)
{
    unsigned int channels;
    return loadPPM( file, data, w, h, &channels);
}

////////////////////////////////////////////////////////////////////////////////
//! Check if command line argument \a flag-name is given
//! @return shrTRUE if command line argument \a flag_name has been given, otherwise shrFALSE
//! @param argc  argc as passed to main()
//! @param argv  argv as passed to main()
//! @param flag_name  name of command line flag
////////////////////////////////////////////////////////////////////////////////
#if 0
shrBOOL shrCheckCmdLineFlag( const int argc, const char** argv, const char* flag_name) 
{
    shrBOOL ret_val = shrFALSE;

    try 
    {
        // initalize 
        CmdArgReader::init( argc, argv);

        // check if the command line argument exists
        if( CmdArgReader::existArg( flag_name)) 
        {
            ret_val = shrTRUE;
        }
    }
    catch( const std::exception& /*ex*/) 
    {    
        std::cerr << "Error when parsing command line argument string." << std::endl;
    } 

    return ret_val;
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! Get the value of a command line argument of type int
//! @return shrTRUE if command line argument \a arg_name has been given and
//!         is of the requested type, otherwise shrFALSE
//! @param argc  argc as passed to main()
//! @param argv  argv as passed to main()
//! @param arg_name  name of the command line argument
//! @param val  value of the command line argument
////////////////////////////////////////////////////////////////////////////////
#if 0
shrBOOL shrGetCmdLineArgumenti( const int argc, const char** argv, 
                        const char* arg_name, int* val) 
{
    shrBOOL ret_val = shrFALSE;

    try 
    {
        // initialize
        CmdArgReader::init( argc, argv);

        // access argument
        const int* v = CmdArgReader::getArg<int>( arg_name);
        if( NULL != v) 
        {
            // assign value
            *val = *v;
            ret_val = shrTRUE;
        }		
		else {
			// fail safe
			val = NULL;
		}
    }
    catch( const std::exception& /*ex*/) 
    {    
        std::cerr << "Error when parsing command line argument string." << std::endl;
    } 

    return ret_val;
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! Get the value of a command line argument of type unsigned int
//! @return shrTRUE if command line argument \a arg_name has been given and
//!         is of the requested type, otherwise shrFALSE
//! @param argc  argc as passed to main()
//! @param argv  argv as passed to main()
//! @param arg_name  name of the command line argument
//! @param val  value of the command line argument
////////////////////////////////////////////////////////////////////////////////
#if 0
shrBOOL shrGetCmdLineArgumentu( const int argc, const char** argv, 
                        const char* arg_name, unsigned int* val) 
{
    shrBOOL ret_val = shrFALSE;

    try 
    {
        // initialize
        CmdArgReader::init( argc, argv);

        // access argument
        const int* v = CmdArgReader::getArg<int>( arg_name);
        if( NULL != v) 
        {
            // assign value
            *val = *v;
            ret_val = shrTRUE;
        }		
		else {
			// fail safe
			val = NULL;
		}
    }
    catch( const std::exception& /*ex*/) 
    {    
        std::cerr << "Error when parsing command line argument string." << std::endl;
    } 

    return ret_val;
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! Get the value of a command line argument of type float
//! @return shrTRUE if command line argument \a arg_name has been given and
//!         is of the requested type, otherwise shrFALSE
//! @param argc  argc as passed to main()
//! @param argv  argv as passed to main()
//! @param arg_name  name of the command line argument
//! @param val  value of the command line argument
////////////////////////////////////////////////////////////////////////////////
#if 0
shrBOOL shrGetCmdLineArgumentf( const int argc, const char** argv, 
                       const char* arg_name, float* val) 
{
    shrBOOL ret_val = shrFALSE;

    try 
    {
        // initialize
        CmdArgReader::init( argc, argv);

        // access argument
        const float* v = CmdArgReader::getArg<float>( arg_name);
        if( NULL != v) 
        {
            // assign value
            *val = *v;
            ret_val = shrTRUE;
        }
		else {
			// fail safe
			val = NULL;
		}
    }
    catch( const std::exception& /*ex*/) 
    {    
        std::cerr << "Error when parsing command line argument string." << std::endl;
    } 

    return ret_val;
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! Get the value of a command line argument of type string
//! @return shrTRUE if command line argument \a arg_name has been given and
//!         is of the requested type, otherwise shrFALSE
//! @param argc  argc as passed to main()
//! @param argv  argv as passed to main()
//! @param arg_name  name of the command line argument
//! @param val  value of the command line argument
////////////////////////////////////////////////////////////////////////////////
#if 0
shrBOOL shrGetCmdLineArgumentstr( const int argc, const char** argv, 
                         const char* arg_name, char** val) 
{
    shrBOOL ret_val = shrFALSE;

    try 
    {
        // initialize
        CmdArgReader::init( argc, argv);

        // access argument
        const std::string* v = CmdArgReader::getArg<std::string>( arg_name);
        if( NULL != v) 
        {

            // allocate memory for the string
            *val = (char*)malloc(sizeof(char) * (v->length() + 1));

            // copy from string to c_str
            #ifdef WIN32
                strcpy_s(*val, v->length() + 1, v->c_str());
            #else
                strcpy(*val, v->c_str());
            #endif
            ret_val = shrTRUE;
        }		
		else {
			// fail safe
			*val = NULL;
		}
    }
    catch( const std::exception& /*ex*/) 
    {    
        std::cerr << "Error when parsing command line argument string."<< 
        std::endl;
    } 

    return ret_val;

}
#endif

////////////////////////////////////////////////////////////////////////////// 
//! Compare two arrays of arbitrary type       
//! @return shrTRUE if \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param epsilon    epsilon to use for the comparison
//////////////////////////////////////////////////////////////////////////////
template<class T, class S>
shrBOOL  
compareData( const T* reference, const T* data, const unsigned int len, 
             const S epsilon, const float threshold) 
{
    ARGCHECK( epsilon >= 0);

    bool result = true;
    unsigned int error_count = 0;

    for( unsigned int i = 0; i < len; ++i) {

        T diff = reference[i] - data[i];
        bool comp = (diff <= epsilon) && (diff >= -epsilon);
        result &= comp;

        error_count += !comp;

#ifdef _DEBUG
        if( ! comp) 
        {
            std::cerr << "ERROR, i = " << i << ",\t " 
                << reference[i] << " / "
                << data[i] 
                << " (reference / data)\n";
        }
#endif
    }

    if (threshold == 0.0f) {
        return (result) ? shrTRUE : shrFALSE;
    } else {
        return (len*threshold > error_count) ? shrTRUE : shrFALSE;
    }
}

////////////////////////////////////////////////////////////////////////////// 
//! Compare two arrays of arbitrary type       
//! @return shrTRUE if \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param epsilon    epsilon to use for the comparison
//////////////////////////////////////////////////////////////////////////////
template<class T, class S>
shrBOOL  
compareDataAsFloat( const T* reference, const T* data, const unsigned int len, 
                    const S epsilon) 
{
    ARGCHECK(epsilon >= 0);

    // If we set epsilon to be 0, let's set a minimum threshold
    float max_error = MAX( (float)epsilon, MIN_EPSILON_ERROR );
    int error_count = 0;
    bool result = true;

    for( unsigned int i = 0; i < len; ++i) {
        float diff = fabs((float)reference[i] - (float)data[i]);
        bool comp = (diff < max_error);
        result &= comp;

        if( ! comp) 
        {
            error_count++;
#ifdef _DEBUG
			if (error_count < 50) {
                shrLog("\n    ERROR(epsilon=%4.3f), i=%d, (ref)0x%02x / (data)0x%02x / (diff)%d\n", max_error, i, reference[i], data[i], (unsigned int)diff);
			}
#endif
        }
    }
    if (error_count) {
        shrLog("\n    Total # of errors = %d\n", error_count);
    }
    return (error_count == 0) ? shrTRUE : shrFALSE;
}

////////////////////////////////////////////////////////////////////////////// 
//! Compare two arrays of arbitrary type       
//! @return shrTRUE if \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param epsilon    epsilon to use for the comparison
//! @param epsilon    threshold % of (# of bytes) for pass/fail
//////////////////////////////////////////////////////////////////////////////
template<class T, class S>
shrBOOL  
compareDataAsFloatThreshold( const T* reference, const T* data, const unsigned int len, 
                    const S epsilon, const float threshold) 
{
    ARGCHECK(epsilon >= 0);

    // If we set epsilon to be 0, let's set a minimum threshold
    float max_error = MAX( (float)epsilon, MIN_EPSILON_ERROR);
    int error_count = 0;
    bool result = true;

    for( unsigned int i = 0; i < len; ++i) {
        float diff = fabs((float)reference[i] - (float)data[i]);
        bool comp = (diff < max_error);
        result &= comp;

        if( ! comp) 
        {
            error_count++;
#ifdef _DEBUG
			if (error_count < 50) {
                shrLog("\n    ERROR(epsilon=%4.3f), i=%d, (ref)0x%02x / (data)0x%02x / (diff)%d\n", max_error, i, reference[i], data[i], (unsigned int)diff);
			}
#endif
        }
    }

    if (threshold == 0.0f) {
        if (error_count) {
            shrLog("\n    Total # of errors = %d\n", error_count);
        }
        return (error_count == 0) ? shrTRUE : shrFALSE;
    } else {

        if (error_count) {
            shrLog("\n    %4.2f(%%) of bytes mismatched (count=%d)\n", (float)error_count*100/(float)len, error_count);
        }

        return ((len*threshold > error_count) ? shrTRUE : shrFALSE);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two float arrays
//! @return shrTRUE if \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
////////////////////////////////////////////////////////////////////////////////
shrBOOL shrComparef( const float* reference, const float* data,
            const unsigned int len ) 
{
    const float epsilon = 0.0;
    return compareData( reference, data, len, epsilon, 0.0f );
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two integer arrays
//! @return shrTRUE if \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
////////////////////////////////////////////////////////////////////////////////
shrBOOL shrComparei( const int* reference, const int* data,
            const unsigned int len ) 
{
    const int epsilon = 0;
    return compareData( reference, data, len, epsilon, 0.0f);
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two unsigned integer arrays, with epsilon and threshold
//! @return shrTRUE if \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
////////////////////////////////////////////////////////////////////////////////
shrBOOL shrCompareuit( const unsigned int* reference, const unsigned int* data,
            const unsigned int len, const float epsilon, const float threshold )
{
//    const int epsilon = 0;
    return compareData( reference, data, len, epsilon, threshold );
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two integer arrays
//! @return shrTRUE if \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
////////////////////////////////////////////////////////////////////////////////
shrBOOL shrCompareub( const unsigned char* reference, const unsigned char* data,
             const unsigned int len ) 
{
    const int epsilon = 0;
    return compareData( reference, data, len, epsilon, 0.0f);
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two integer arrays (inc Threshold for # of pixel we can have errors)
//! @return  shrTRUE if \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
////////////////////////////////////////////////////////////////////////////////
shrBOOL shrCompareubt( const unsigned char* reference, const unsigned char* data,
             const unsigned int len, const float epsilon, const float threshold ) 
{
    return compareDataAsFloatThreshold( reference, data, len, epsilon, threshold );
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two integer arrays
//! @return  shrTRUE if \a reference and \a data are identical, 
//!          otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
////////////////////////////////////////////////////////////////////////////////
shrBOOL shrCompareube( const unsigned char* reference, const unsigned char* data,
             const unsigned int len, const float epsilon ) 
{
    return compareDataAsFloat( reference, data, len, epsilon );
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two float arrays with an epsilon tolerance for equality
//! @return shrTRUE if \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param epsilon    epsilon to use for the comparison
////////////////////////////////////////////////////////////////////////////////
shrBOOL shrComparefe( const float* reference, const float* data,
             const unsigned int len, const float epsilon ) 
{
    return compareData( reference, data, len, epsilon, 0.0f);
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two float arrays with an epsilon tolerance for equality and a 
//!     threshold for # pixel errors
//! @return  shrTRUE if \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param epsilon    epsilon to use for the comparison
////////////////////////////////////////////////////////////////////////////////
shrBOOL shrComparefet( const float* reference, const float* data,
             const unsigned int len, const float epsilon, const float threshold ) 
{
    return compareDataAsFloatThreshold( reference, data, len, epsilon, threshold );
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two float arrays using L2-norm with an epsilon tolerance for equality
//! @return shrTRUE if \a reference and \a data are identical, otherwise shrFALSE
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param epsilon    epsilon to use for the comparison
////////////////////////////////////////////////////////////////////////////////
shrBOOL shrCompareL2fe( const float* reference, const float* data,
                const unsigned int len, const float epsilon ) 
{
    ARGCHECK(epsilon >= 0);

    float error = 0;
    float ref = 0;

    for( unsigned int i = 0; i < len; ++i) {

        float diff = reference[i] - data[i];
        error += diff * diff;
        ref += reference[i] * reference[i];
    }

    float normRef = sqrtf(ref);
    if (fabs(ref) < 1e-7) {
#ifdef _DEBUG
        std::cerr << "ERROR, reference l2-norm is 0\n";
#endif
        return shrFALSE;
    }
    float normError = sqrtf(error);
    error = normError / normRef;
    bool result = error < epsilon;
#ifdef _DEBUG
    if( ! result) 
    {
        std::cerr << "ERROR, l2-norm error " 
            << error << " is greater than epsilon " << epsilon << "\n";
    }
#endif

    return result ? shrTRUE : shrFALSE;
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two PPM image files with an epsilon tolerance for equality
//! @return shrTRUE if \a reference and \a data are identical, otherwise shrFALSE
//! @param src_file   filename for the image to be compared
//! @param data       filename for the reference data / gold image
//! @param epsilon    epsilon to use for the comparison
//! @param threshold  threshold of pixels that can still mismatch to pass (i.e. 0.15f = 15% must pass)
//! @param verboseErrors output details of image mismatch to std::cerr
////////////////////////////////////////////////////////////////////////////////
shrBOOL shrComparePPM( const char *src_file, const char *ref_file, const float epsilon, const float threshold)
{
	unsigned char *src_data, *ref_data;
	unsigned long error_count = 0;
	unsigned int ref_width, ref_height;
	unsigned int src_width, src_height;

    // Check sample and reference file pointers
	if (src_file == NULL || ref_file == NULL) {
		shrLog("\n> shrComparePGM: src_file or ref_file is NULL\n  Aborting comparison !!!\n\n");
		return shrFALSE;
	}
    shrLog("\n> shrComparePPM:\n    (a)rendered:  <%s>\n    (b)reference: <%s>\n", src_file, ref_file);

    // Load the ref image file
	if (shrLoadPPM4ub(ref_file, &ref_data, &ref_width, &ref_height) != shrTRUE) 
	{
		shrLog("\n    Unable to load ref image file: %s\n    Aborting comparison !!!\n\n", ref_file);
		return shrFALSE;
	}

    // Load the sample image file
	if (shrLoadPPM4ub(src_file, &src_data, &src_width, &src_height) != shrTRUE) 
	{
		shrLog("\n    Unable to load src image file: %s\n    Aborting comparison !!!\n\n", src_file);
		return shrFALSE;
	}

    // check to see if image dimensions match
	if(src_height != ref_height || src_width != ref_width)
	{
		shrLog("\n    Source and ref size mismatch (%u x %u) vs (%u x %u)\n    Aborting Comparison !!!\n\n ", 
		    src_width, src_height, ref_width, ref_height);
		return shrFALSE;
	}

    // compare the images
	if (shrCompareubt(ref_data, src_data, src_width*src_height*4, epsilon, threshold ) == shrFALSE) 
	{
		error_count=1;
	}

    shrLog("    Images %s\n\n", (error_count == 0) ? "Match" : "Don't Match !!!"); 
	return (error_count == 0) ? shrTRUE : shrFALSE;  // returns true if all pixels pass
}

////////////////////////////////////////////////////////////////////////////////
//! Compare two PGM image files with an epsilon tolerance for equality
//! @return shrTRUE if \a reference and \a data are identical, otherwise shrFALSE
//! @param src_file   filename for the image to be compared
//! @param data       filename for the reference data / gold image
//! @param epsilon    epsilon to use for the comparison
//! @param threshold  threshold of pixels that can still mismatch to pass (i.e. 0.15f = 15% must pass)
////////////////////////////////////////////////////////////////////////////////
shrBOOL shrComparePGM( const char *src_file, const char *ref_file, const float epsilon, const float threshold)
{
	unsigned char *src_data, *ref_data;
	unsigned long error_count = 0;
	unsigned int ref_width, ref_height;
	unsigned int src_width, src_height;

    // Check sample and reference file pointers
	if (src_file == NULL || ref_file == NULL) {
		shrLog("\n> shrComparePGM: src_file or ref_file is NULL\n  Aborting comparison !!!\n\n");
		return shrFALSE;
	}
    shrLog("\n> shrComparePGM:\n    (a)rendered:  <%s>\n    (b)reference: <%s>\n", src_file, ref_file);

    // Load the ref image file
	if (shrLoadPPM4ub(ref_file, &ref_data, &ref_width, &ref_height) != shrTRUE) 
	{
		shrLog("\n    Unable to load ref image file: %s\n    Aborting comparison !!!\n\n", ref_file);
		return shrFALSE;
	}

    // Load the sample image file
	if (shrLoadPPM4ub(src_file, &src_data, &src_width, &src_height) != shrTRUE) 
	{
		shrLog("\n    Unable to load src image file: %s\n    Aborting comparison !!!\n\n", src_file);
		return shrFALSE;
	}

    // check to see if image dimensions match
	if(src_height != ref_height || src_width != ref_width)
	{
		shrLog("\n    Source and ref size mismatch (%u x %u) vs (%u x %u)\n    Aborting Comparison !!!\n\n ", 
		    src_width, src_height, ref_width, ref_height);
		return shrFALSE;
	}

    // compare the images
	if (shrCompareubt(ref_data, src_data, src_width*src_height*4, epsilon, threshold ) == shrFALSE) 
	{
		error_count=1;
	}

    shrLog("    Images %s\n\n", (error_count == 0) ? "Match" : "Don't Match !!!"); 
	return (error_count == 0) ? shrTRUE : shrFALSE;  // returns true if all pixels pass
}

// Load raw data from disk
unsigned char* shrLoadRawFile(const char* filename, size_t size)
{
    FILE *fp = NULL;
    #ifdef WIN32
        errno_t err;
        if ((err = fopen_s(&fp, filename, "rb")) != 0)
    #else
        if ((fp = fopen(filename, "rb")) == NULL) 
    #endif
        {
            shrLog(" Error opening file '%s' !!!\n", filename);
            return 0;
        }

    unsigned char* data = (unsigned char*)malloc(size);
    size_t read = fread(data, 1, size, fp);
    fclose(fp);

    shrLog(" Read '%s', %d bytes\n", filename, read);

    return data;
}

// Round Up Division function
size_t shrRoundUp(int group_size, int global_size) 
{
    int r = global_size % group_size;
    if(r == 0) 
    {
        return global_size;
    } else 
    {
        return global_size + group_size - r;
    }
}
