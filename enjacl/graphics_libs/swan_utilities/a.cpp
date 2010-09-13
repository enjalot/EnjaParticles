
typedef struct dim3 {
  int x; int y; int z;

  dim3(int a, int b, int c) {
  	x = a; y = b; z = c;
  }
} dim3;

int main()
{
	dim3 a(1,2,3);
	return 0;
}
