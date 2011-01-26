#define KERNEL_SIZE 49
#define KERNEL_DIAMETER 7
const float sigmasq = 4.;//0.84089642*0.84089642;
const float pi = 3.141592654;

float kernel[KERNEL_SIZE];

uniform sampler2D depthTex;

float step_w = 1.0/800.;
float step_h = 1.0/600.;
vec2 offset[KERNEL_SIZE];

void main(void)
{
	/*kernel[0]=1.0/16.0; kernel[1] = 2.0/16.0; kernel[2] = 1.0/16.0;
	kernel[3]=2.0/16.0; kernel[4] = 4.0/16.0; kernel[5] = 2.0/16.0;
	kernel[6]=1.0/16.0; kernel[7] = 2.0/16.0; kernel[8] = 1.0/16.0;
	offset[0]=vec2(-step_w, -step_h); offset[1] = vec2(0.0, -step_h); offset[2] = vec2(step_w, -step_h);
	offset[3]=vec2(-step_w, 0.0); offset[4] = vec2(0.0, 0.0); offset[5] = vec2(step_w, 0.0);
	offset[6]=vec2(-step_w, step_h); offset[7] = vec2(0.0, step_h); offset[8] = vec2(step_w, step_h);*/
   for(int i=0; i<KERNEL_DIAMETER; i++ )
	{
	   for(int j=0; j<KERNEL_DIAMETER; j++ )
	   {
	    	kernel[i*KERNEL_DIAMETER+j] = (1./(2.*pi*sigmasq))*exp(-(pow(i-3.,2.)+pow(j-3.,2.))/(2.*sigmasq));
			offset[i*KERNEL_DIAMETER+j] = vec2((i-3.)*step_w,(j-3.)*step_h);
	   }
	}
   float sum = 0.0;	
   for(int i=0; i<KERNEL_DIAMETER; i++ )
   {
	   for(int j=0; j<KERNEL_DIAMETER; j++ )
	   {
			float tmp = texture2D(depthTex, gl_TexCoord[0].st + offset[i*KERNEL_DIAMETER+j]).x;
			if(tmp>0.95)
				continue;
			sum += tmp * kernel[i*KERNEL_DIAMETER+j];
		}
   }
   if(sum==0.0)
	    sum = 1.;
   gl_FragColor = vec4(sum,sum,sum,1.0);
   gl_FragDepth = sum;
}
/*uniform sampler2D depthTex; // the texture with the scene you want to blur
uniform vec2 blurDir;
uniform float blurScale;
uniform float blurDepthFallOff;
 
void main(void)
{
	float depth = texture2D(depthTex,gl_TexCoord[0].xy).x;
	float sum = 0.0;
	float wsum = 0.0;
	float filterRadius = 3.0;
	float blurDir = vec2(1.0,0.0)*dFdx(gl_TexCoord[0].xy);//*dFdy(gl_TexCoord[0].xy);
	float blurDepthFallOff = 1.0;
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
