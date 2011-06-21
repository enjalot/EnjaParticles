uniform sampler2D depthTex; // the texture with the scene you want to blur
uniform float del_x;
uniform float del_y;
uniform float sig;
const float pi = 3.141592654;
const float maxDepth=0.9999999;
 
void main(void)
{
	float depth=texture2D(depthTex, gl_TexCoord[0].st).x;
	if(depth>maxDepth)
	{
		discard;
	}
    float gauss = (1./(2.*pi*sig*sig));
    float denom = 1./(2.*sig*sig);
    int width = int(2.*sig)+1;
   float sum_x = 0.0;	
   float sum_y = 0.0;
   for(int i=-width; i<width; i++ )
   {
       float ex = exp(-(float(i*i))*denom);
       float depthx = texture2D(depthTex,gl_TexCoord[0].st+vec2(float(i)*del_x,0.0)).x;
       float depthy = texture2D(depthTex,gl_TexCoord[0].st+vec2(0.0,float(i)*del_y)).x;
       if(abs(depthx-depth)>0.01)
       {
           depthx = depth;
       }
       if(abs(depthy-depth)>0.01)
       {
           depthy = depth;
       }
		sum_x += depthx * ex;
        sum_y += depthy * ex;
   }
   float tmp = gauss * sum_x*sum_y;
   gl_FragData[0] = vec4(tmp,tmp,tmp,1.0);
   gl_FragDepth = tmp;
}
