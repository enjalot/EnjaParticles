uniform sampler2D col;
uniform sampler2D depth;

void main()
{
	/*gl_FragColor.rgb = vec3(depth);
    vec4 tex = texture2D(col, gl_TexCoord[0].st);
	//This should come from the application and not be hardcoded.
	float Vx = 600;
	float Vy = 800;

	vec3 Pxy;
	//This should come from the application and not be hardcoded.
	float f_near = 0.1;
	float Fx = (2.0 - f_near)/gl_ProjectionMatrix[0][0];
	float Fy = (2.0 - f_near)/gl_ProjectionMatrix[1][1];
	Pxy.x = ((((2.0*gl_FragCoord.x)/Vx)-1.0)/Fx);
	Pxy.y = ((((2.0*gl_FragCoord.y)/Vy)-1.0)/Fy);
	Pxy.z = gl_FragCoord.z;
	Pxy = Pxy*gl_FragCoord.z;
	if(tex.a>0.0)   										  
	{
		gl_FragColor = vec4(gl_FragCoord.zzz,tex.a);
	}
    else
	{
		gl_FragColor = vec4(0.0);
	}*/
}
