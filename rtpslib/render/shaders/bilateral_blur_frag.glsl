//#define KERNEL_SIZE 10000
//#define KERNEL_DIAMETER 31 
//const float sigmasq = 64. ;//0.84089642*0.84089642;
const float pi = 3.141592654;

//float kernel[KERNEL_SIZE];

uniform sampler2D depthTex;
uniform float del_x;
uniform float del_y;
uniform float sig;

//vec2 offset[KERNEL_SIZE];

void main(void)
{
	float depth=texture2D(depthTex, gl_TexCoord[0].st).x;
	float maxDepth=0.9999999;
	if(depth>maxDepth)
	{
		discard;
		//return;
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
/*uniform sampler2D depthTex; // the texture with the scene you want to blur
uniform vec2 blurDir;
uniform float blurScale;
uniform float blurDepthFallOff;

float filterwidth(vec2 v)
{
  vec2 fw = max(abs(dFdx(v)), abs(dFdy(v)));
  return max(fw.x, fw.y);
}

void main(void)
{
	float depth = texture2D(depthTex,gl_TexCoord[0].xy).x;
	float sum = 0.0;
	float wsum = 0.0;
	float filterRadius = 3.0;
	float blurDir = vec2(1.0,0.0)*1/800;//*dFdy(gl_TexCoord[0].xy);
	float blurDepthFallOff = 1.0;
	float blurScale = 1./5.;
	for(float x=-filterRadius; x<=filterRadius; x+=1.0)
	{
		float sample = texture2D(depthTex,gl_TexCoord[0].xy+x*blurDir).x;
		
		float r = x * blurScale;
		float w = exp(-r*r);

		float r2 = (sample - depth) * blurDepthFallOff;
		float g = exp(-r2*r2);
		sum += sample * w * g;
		wsum += w * g;
	}
	if(wsum > 0.0)
	{
		sum /= wsum;
	}

	gl_FragData[0] = vec4(sum,sum,sum,1.0);
	gl_FragDepth=sum;
}*/
