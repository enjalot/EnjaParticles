uniform sampler2D depthTex; // the texture with the scene you want to blur
uniform float del_y;
uniform float sig;
const float pi = 3.141592654;
 
void main(void)
{
	float depth=texture2D(depthTex, gl_TexCoord[0].st).x;
	float maxDepth=0.9999999;
	if(depth>maxDepth)
	{
		discard;
	}
    float sigmasq = sig*sig;
    float gauss = (1./(2.*pi*sigmasq));
    float denom = (2.*sigmasq);
    int width = int(3*sig);
   float sum = 0.0;	
   for(int i=-width; i<width; i++ )
   {
		float tmp = texture2D(depthTex,gl_TexCoord[0].st+vec2(0.0,float(i)*del_y)).x;
		sum += tmp * gauss *exp(-(pow(float(i),2.))/denom);
   }
   gl_FragData[0] = vec4(sum,sum,sum,1.0);
   gl_FragDepth = sum;
}
