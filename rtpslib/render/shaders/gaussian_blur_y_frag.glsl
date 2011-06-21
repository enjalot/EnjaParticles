uniform sampler2D depthTex; // the texture with the scene you want to blur
uniform float del_y;
uniform float falloff;
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
    float gauss = (1./(sqrt(2.*pi)*sig));
    float denom = 1./(2.*sig*sig);
    int width = int(2.*sig)+1;
   float sum = 0.0;	
   for(int i=-width; i<width; i++ )
   {
        float d = texture2D(depthTex,gl_TexCoord[0].st+vec2(0.0,float(i)*del_y)).x;
        if(abs(depth-d)>falloff)
           d =depth;
		sum += d * exp(-(float(i*i))*denom);
   }
   sum*=gauss;
   gl_FragData[0] = vec4(sum,sum,sum,1.0);
   gl_FragDepth = sum;
}
