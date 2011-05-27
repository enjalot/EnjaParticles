const float pi = 3.141592654;

uniform sampler2D depthTex;
uniform float del_x;
uniform float del_y;
uniform float sig;
const float maxDepth = 0.9999999;

void main(void)
{
	float depth=texture2D(depthTex, gl_TexCoord[0].st).x;
	if(depth>maxDepth)
	{
		discard;
	}
   float sum = 0.0;	
   float denom = (2.*sig*sig);
   float gauss =(1./(2.*pi*sig*sig));
   int width = int(2.*sig)+1;
   for(int i=-width; i<width; i++ )
   {
	   for(int j=-width; j<width; j++ )
	   {
			float tmp = texture2D(depthTex,gl_TexCoord[0].st+vec2(float(i)*del_x,float(j)*del_y)).x;
			sum += tmp * gauss *exp(-(pow(float(i),2.)+pow(float(j),2.))/denom);
		}
   }
   gl_FragData[0] = vec4(sum,sum,sum,1.0);
   gl_FragDepth = sum;
}
