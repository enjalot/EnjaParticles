uniform float pointRadius;  // point size in world space
uniform float pointScale;   // scale to calculate size in pixels

void main()
{
	gl_PointSize = pointRadius;
    vec4 viewPos = gl_ModelViewMatrix * gl_Vertex;
    gl_Position = ftransform();

/*	int Vy=800;
	int Vx=600;

	//make h a uniform parameter
	int h=1;
	depth = float[(Vy*Vx)/(h*h)](1.0/0.0);
	//for(int i = 0; i<(Vy*Vx)/(h*h); i++)
	//{
	//	depth[i] = 1.0/0.0;
	//}

    gl_PointSize = pointRadius;


    gl_TexCoord[0] = gl_MultiTexCoord0;
	gl_Position = ftransform();
	gl_Position.z = gl_Position.z*gl_Position.w
	vec3 r = vec3((pointRadius*Vx*sqrt(pow(gl_ProjectionMatrix[0][0],2.)+pow(gl_ProjectionMatrix[0][1],2.)+pow(gl_ProjectionMatrix[0][2],2.))/gl_Position.w),
					pointRadius*Vy*sqrt(pow(gl_ProjectionMatrix[1][0],2.)+pow(gl_ProjectionMatrix[1][1],2.)+pow(gl_ProjectionMatrix[1][2],2.))/gl_Position.w),
					pointRadius);
	for(int i = 0; i<600; i++)
	{
		for(int j=0;j<800; j++)
		{
			float sumSq = pow(i*h-gl_Position.x,2)+pow(j*h-gl_Position.y,2);
			if(sumSq<pow(r.x,2))
			{
				float d = gl_Position.z-r.z*sqrt(1-(sumSq/pow(r.x,2)));
				depth[j+(i*800)]=min(depth[j+(i*800)],d);
			}
		}
	}*/
}
