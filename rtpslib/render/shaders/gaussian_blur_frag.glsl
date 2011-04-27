#version 150
uniform sampler2D depthTex;
uniform float del_x;
uniform float del_y;
uniform float gaussian[]

void main(void)
{
	float depth=texture2D(depthTex, gl_TexCoord[0].st).x;
	float maxDepth=0.9999999;
	//float threshold=0.01;
	if(depth>maxDepth)
	{
		discard;
		//return;
	}
   float sum = 0.0;	
   for(int i=0; i<KERNEL_DIAMETER; i++ )
   {
	   for(int j=0; j<KERNEL_DIAMETER; j++ )
	   {
			float tmp = texture2D(depthTex,gl_TexCoord[0].st+vec2(float(i-(KERNEL_DIAMETER/2))*del_x,float(j-(KERNEL_DIAMETER/2))*del_y)).x;//texture2D(depthTex, gl_TexCoord[0].st + offset[(i*KERNEL_DIAMETER)+j]).x;
			sum += tmp * (1./(2.*pi*sigmasq))*exp(-(pow(float(i-(KERNEL_DIAMETER/2)),2.)+pow(float(j-(KERNEL_DIAMETER/2)),2.))/(2.*sigmasq));
		}
   }
   gl_FragData[0] = vec4(sum,sum,sum,1.0);
   gl_FragDepth = sum;
}
