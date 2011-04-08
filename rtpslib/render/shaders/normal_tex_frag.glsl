uniform sampler2D depthTex;
uniform sampler2D colorTex;
uniform float del_x;
uniform float del_y;
//varying float w;

vec3 uvToEye(vec2 texCoord,float z)
{
	// convert texture coordinate to homogeneous space
	vec2 xyPos = (texCoord*2. -1.);

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
	float maxDepth = 0.999;
	//float maxDepth = 0.95;
	float depth = texture2D(depthTex,gl_TexCoord[0].xy).x;
	if(depth>maxDepth)
	{
		discard;
		//return;
	}	

	vec3 posEye = uvToEye(gl_TexCoord[0].xy,depth);
	vec2 texCoord1 = vec2(gl_TexCoord[0].x+del_x,gl_TexCoord[0].y);
	vec2 texCoord2 = vec2(gl_TexCoord[0].x-del_x,gl_TexCoord[0].y);

	vec3 ddx = uvToEye(texCoord1, texture2D(depthTex,texCoord1.xy).x)-posEye;
	vec3 ddx2 = posEye-uvToEye(texCoord2, texture2D(depthTex,texCoord2.xy).x);
	if(abs(ddx.z)>abs(ddx2.z))
	{
		ddx = ddx2;
	}

	texCoord1 = vec2(gl_TexCoord[0].x,gl_TexCoord[0].y+del_y);
	texCoord2 = vec2(gl_TexCoord[0].x,gl_TexCoord[0].y-del_y);

	vec3 ddy = uvToEye(texCoord1, texture2D(depthTex,texCoord1.xy).x)-posEye;
	vec3 ddy2 = posEye-uvToEye(texCoord2, texture2D(depthTex,texCoord2.xy).x);
	if(abs(ddy.z)>abs(ddy2.z))
	{
		ddy = ddy2;
	}

	vec3 n = cross(ddx,ddy);
	n = normalize(n);
	const vec3 lightDir = vec3(-0.5, 1.5, 5.);
	//const vec3 lightDir = vec3(-0.5, -0.5, 2.);
	//const vec3 lightDir = vec3(-1.0,.0,-2.0);
	const float shininess = 300.0;
	float diffuse = max(0.0,dot(lightDir,n));

	vec3 v = normalize(-posEye);
    vec3 h = normalize(lightDir + v);
    float specular = pow(max(0.0, dot(n, h)), shininess);
	
    //gl_FragData[0] = vec4(1.,0.0,0.0,0.5)*diffuse+specular;
	gl_FragData[0] = texture2D(colorTex,gl_TexCoord[0].xy)+specular;//*diffuse;//+specular;
	//gl_FragData[0] = vec4((n+vec3(1.0))/2.,1.0);
	//gl_FragData[0] = vec4(depth,depth,depth,1.0);
	//gl_FragDepth = depth;
}
