uniform sampler2D depthTex;

vec3 uvToEye(vec2 texCoord,float z)
{
	// convert texture coordinate to homogeneous space
	vec2 xyPos = texCoord * 2.0 - 1.0;

	// construct clip-space position
	vec4 clipPos = vec4( xyPos, z, 1.0 );

	// transform from clip space to view (eye) space
	// NOTE: this assumes that you've precomputed the
	// inverse of the view->clip transform matrix and
	// provided it to the shader as a constant.
	vec4 viewPos =  (gl_ProjectionMatrixInverse * clipPos);
	return(viewPos.xyz/viewPos.w);
}

void main()
{
	//make this a uniform variable;
	float maxDepth = 0.5;
	float depth = texture2D(depthTex,gl_TexCoord[0].xy).x;
	/*if(depth>maxDepth)
	{
		discard;
		return;
	}*/	
	float del_x = 1.0/800.0;
	float del_y = 1.0/600.0;

	vec3 posEye = uvToEye(gl_TexCoord[0].xy,depth);
	vec2 texCoord1 = vec2(gl_TexCoord[0].x+del_x,gl_TexCoord[0].y);
	vec2 texCoord2 = vec2(gl_TexCoord[0].x-del_x,gl_TexCoord[0].y);

	vec3 ddx = uvToEye(texCoord1, texture2D(depthTex,texCoord1.xy).x)-posEye;
	/*vec3 ddx2 = posEye-uvToEye(texCoord2, texture2D(depthTex,texCoord2.xy).x);
	if(abs(ddx.z)>abs(ddx2.z))
	{
		ddx = ddx2;
	}*/

	texCoord1 = vec2(gl_TexCoord[0].x,gl_TexCoord[0].y+del_y);
	texCoord2 = vec2(gl_TexCoord[0].x,gl_TexCoord[0].y-del_y);

	vec3 ddy = uvToEye(texCoord1, texture2D(depthTex,texCoord1.xy).x)-posEye;
	/*vec3 ddy2 = posEye-uvToEye(texCoord2, texture2D(depthTex,texCoord2.xy).x);
	if(abs(ddy.z)>abs(ddy2.z))
	{
		ddy = ddy2;
	}*/
	vec3 n = cross(ddx,ddy);
	n = normalize(n);
	/*const vec3 lightDir = vec3(0.577, 0.577, 0.577);
	float diffuse = max(0.0,dot(lightDir,n));
	gl_FragData[0] = vec4(0.,0.,1.,1.0)*diffuse;*/
	gl_FragData[0] = vec4(n,1.0);
	//gl_FragDepth = depth;
}
