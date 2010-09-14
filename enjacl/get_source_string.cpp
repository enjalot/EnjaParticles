char*  source CL::getSourceString(const char* path_to_source_file)
{
	// Must find a way to only compile a single time. 
	// Define all programs before starting the code? 

	//printf("enter addProgramR\n");

    FILE* fd =  fopen(path_to_source_file, "r");
	if (fd == 0) {
		printf("cannot open file: %s\n", path_to_source_file);
	}
// should not limit string size
	int max_len = 300000;
    char* source = new char [max_len];
    int nb = fread(source, 1, max_len, fd);    

	if (nb > (max_len-2)) { 
        printf("cannot read program from %s\n", path_to_source_file);
        printf("   buffer size too small\n");
    }    
	source[nb] = '\0';

	return source;
}
//----------------------------------------------------------------------
